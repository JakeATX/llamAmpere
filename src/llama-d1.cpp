// [D1] opt-in exact rescore of the coarse output head. See llama-d1.h.

#include "llama-d1.h"

#include "llama-impl.h"

#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>

#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
#include <immintrin.h>
#define LLAMA_D1_AVX2 1
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static constexpr size_t LLAMA_D1_END_BYTES = 1u << 20;

static uint64_t fnv64_update(uint64_t h, const uint8_t * p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

uint64_t llama_d1_fnv64_ends(const uint8_t * data, size_t nbytes) {
    const size_t k = std::min(LLAMA_D1_END_BYTES, nbytes);
    uint64_t h = 0xcbf29ce484222325ULL;
    h = fnv64_update(h, data, k);
    h = fnv64_update(h, data + (nbytes - k), k);
    return h;
}

bool llama_d1_src_from_file(const std::string & path, size_t offs, size_t nbytes, int type, llama_d1_src & out) {
    out = {};
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    const size_t k = std::min(LLAMA_D1_END_BYTES, nbytes);
    std::vector<uint8_t> buf(2 * k);
    bool ok = true;
#ifdef _WIN32
    ok = ok && _fseeki64(f, (long long) offs, SEEK_SET) == 0;
#else
    ok = ok && fseeko(f, (off_t) offs, SEEK_SET) == 0;
#endif
    ok = ok && std::fread(buf.data(), 1, k, f) == k;
#ifdef _WIN32
    ok = ok && _fseeki64(f, (long long) (offs + nbytes - k), SEEK_SET) == 0;
#else
    ok = ok && fseeko(f, (off_t) (offs + nbytes - k), SEEK_SET) == 0;
#endif
    ok = ok && std::fread(buf.data() + k, 1, k, f) == k;
    std::fclose(f);
    if (!ok) {
        return false;
    }
    uint64_t h = 0xcbf29ce484222325ULL;
    h = fnv64_update(h, buf.data(), 2 * k);
    out.valid  = true;
    out.type   = type;
    out.nbytes = nbytes;
    out.fnv    = h;
    return true;
}

// ---- exact rows -------------------------------------------------------------------------------------------------

llama_d1_head::~llama_d1_head() {
#ifdef _WIN32
    if (map_base) { UnmapViewOfFile(map_base); }
    if (h_map)    { CloseHandle((HANDLE) h_map); }
    if (h_file)   { CloseHandle((HANDLE) h_file); }
#else
    if (map_base) { munmap(map_base, map_size); }
#endif
}

float llama_d1_head::dot(const float * h, int64_t id) const {
    const uint8_t * row = rows + (size_t) id * row_bytes;
    const int64_t n = n_embd;
    if (type == GGML_TYPE_F16) {
        const ggml_fp16_t * w = (const ggml_fp16_t *) row;
        int64_t k = 0;
        float sum = 0.0f;
#ifdef LLAMA_D1_AVX2
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        for (; k + 32 <= n; k += 32) {
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (w + k +  0))), _mm256_loadu_ps(h + k +  0), a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (w + k +  8))), _mm256_loadu_ps(h + k +  8), a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (w + k + 16))), _mm256_loadu_ps(h + k + 16), a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (w + k + 24))), _mm256_loadu_ps(h + k + 24), a3);
        }
        const __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        __m128 r = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        r = _mm_add_ps(r, _mm_movehl_ps(r, r));
        r = _mm_add_ss(r, _mm_movehdup_ps(r));
        sum = _mm_cvtss_f32(r);
#endif
        for (; k < n; ++k) {
            sum += ggml_fp16_to_fp32(w[k]) * h[k];
        }
        return sum;
    }
    if (type == GGML_TYPE_BF16) {
        // [EXACTHEAD] bf16 -> f32 is a 16-bit shift (exact); the served and audit paths share this dot
        const uint16_t * w = (const uint16_t *) row;
        int64_t k = 0;
        float sum = 0.0f;
#ifdef LLAMA_D1_AVX2
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        const auto ld = [&](int64_t o) {
            return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) (w + o))), 16));
        };
        for (; k + 32 <= n; k += 32) {
            a0 = _mm256_fmadd_ps(ld(k +  0), _mm256_loadu_ps(h + k +  0), a0);
            a1 = _mm256_fmadd_ps(ld(k +  8), _mm256_loadu_ps(h + k +  8), a1);
            a2 = _mm256_fmadd_ps(ld(k + 16), _mm256_loadu_ps(h + k + 16), a2);
            a3 = _mm256_fmadd_ps(ld(k + 24), _mm256_loadu_ps(h + k + 24), a3);
        }
        const __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        __m128 r = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        r = _mm_add_ps(r, _mm_movehl_ps(r, r));
        r = _mm_add_ss(r, _mm_movehdup_ps(r));
        sum = _mm_cvtss_f32(r);
#endif
        for (; k < n; ++k) {
            uint32_t u = (uint32_t) w[k] << 16;
            float f;
            std::memcpy(&f, &u, sizeof(f));
            sum += f * h[k];
        }
        return sum;
    }
    // Q8_0: blocks of { fp16 d; int8 qs[32] }
    constexpr int QK = 32;
    constexpr size_t BS = sizeof(ggml_fp16_t) + QK;
    const int64_t nb = n / QK;
    float sum = 0.0f;
#ifdef LLAMA_D1_AVX2
    __m256 acc = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = row + b * BS;
        ggml_fp16_t dh;
        std::memcpy(&dh, blk, sizeof(dh));
        const __m256 d = _mm256_set1_ps(ggml_fp16_to_fp32(dh));
        const int8_t * qs = (const int8_t *) (blk + sizeof(ggml_fp16_t));
        const float * hb = h + b * QK;
        __m256 s = _mm256_setzero_ps();
        for (int q = 0; q < QK; q += 8) {
            const __m128i q8 = _mm_loadl_epi64((const __m128i *) (qs + q));
            const __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
            s = _mm256_fmadd_ps(qf, _mm256_loadu_ps(hb + q), s);
        }
        acc = _mm256_fmadd_ps(d, s, acc);
    }
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    sum = _mm_cvtss_f32(r);
#else
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = row + b * BS;
        ggml_fp16_t dh;
        std::memcpy(&dh, blk, sizeof(dh));
        const float d = ggml_fp16_to_fp32(dh);
        const int8_t * qs = (const int8_t *) (blk + sizeof(ggml_fp16_t));
        float s = 0.0f;
        for (int q = 0; q < QK; ++q) {
            s += (float) qs[q] * h[b * QK + q];
        }
        sum += d * s;
    }
#endif
    return sum;
}

static bool d1_kv_u64(const gguf_context * g, const char * key, uint64_t & out) {
    const int64_t k = gguf_find_key(g, key);
    if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_UINT64) {
        return false;
    }
    out = gguf_get_val_u64(g, k);
    return true;
}

static bool d1_kv_str(const gguf_context * g, const char * key, std::string & out) {
    const int64_t k = gguf_find_key(g, key);
    if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_STRING) {
        return false;
    }
    out = gguf_get_val_str(g, k);
    return true;
}

// the sidecar's recorded source fingerprint, without mapping it (used to tell "target without the overlay")
static bool d1_peek_src(const char * path, uint64_t & bytes, uint64_t & fnv) {
    struct ggml_context * meta = nullptr;
    gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) {
        return false;
    }
    const bool ok = d1_kv_u64(g, "d1.source_bytes", bytes) && d1_kv_u64(g, "d1.source_fnv64", fnv);
    gguf_free(g);
    if (meta) {
        ggml_free(meta);
    }
    return ok;
}

std::shared_ptr<llama_d1_head> llama_d1_head_open(const char * path, const llama_d1_src & src,
                                                  int64_t n_embd, int64_t n_vocab, std::string & err) {
    struct ggml_context * meta = nullptr;
    gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) {
        err = format("cannot read sidecar GGUF %s", path);
        return nullptr;
    }
    struct cleanup_t {
        gguf_context * g; ggml_context * m;
        ~cleanup_t() { gguf_free(g); if (m) { ggml_free(m); } }
    } cleanup { g, meta };

    // (a) shape, type, rows_type KV
    const int64_t tid = gguf_find_tensor(g, "output_exact.weight");
    ggml_tensor * t = meta ? ggml_get_tensor(meta, "output_exact.weight") : nullptr;
    if (tid < 0 || !t) {
        err = format("sidecar %s has no tensor output_exact.weight", path);
        return nullptr;
    }
    if (t->ne[0] != n_embd || t->ne[1] != n_vocab || ggml_n_dims(t) != 2) {
        err = format("sidecar shape [%" PRId64 ", %" PRId64 "] != model head [%" PRId64 ", %" PRId64 "]",
                     t->ne[0], t->ne[1], n_embd, n_vocab);
        return nullptr;
    }
    if (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16 && t->type != GGML_TYPE_Q8_0) {
        err = format("sidecar rows type %s is not f16, bf16 or q8_0", ggml_type_name(t->type));
        return nullptr;
    }
    if (t->type == GGML_TYPE_Q8_0 && n_embd % 32 != 0) {
        err = "q8_0 sidecar needs n_embd % 32 == 0";
        return nullptr;
    }
    std::string rows_type;
    const char * want_rows_type = t->type == GGML_TYPE_F16 ? "f16" : t->type == GGML_TYPE_BF16 ? "bf16" : "q8_0";
    if (!d1_kv_str(g, "d1.rows_type", rows_type) || rows_type != want_rows_type) {
        err = format("sidecar KV d1.rows_type '%s' does not match tensor type %s", rows_type.c_str(), ggml_type_name(t->type));
        return nullptr;
    }
    uint64_t src_bytes = 0, src_fnv = 0, self_bytes = 0, self_fnv = 0;
    if (!d1_kv_u64(g, "d1.source_bytes", src_bytes) || !d1_kv_u64(g, "d1.source_fnv64", src_fnv) ||
        !d1_kv_u64(g, "d1.self_bytes", self_bytes) || !d1_kv_u64(g, "d1.self_fnv64", self_fnv)) {
        err = "sidecar is missing a d1.* fingerprint KV";
        return nullptr;
    }
    // (c) source fingerprint
    if (!src.valid) {
        err = "the base file's output.weight fingerprint was not recorded at load";
        return nullptr;
    }
    if (src.nbytes != src_bytes || src.fnv != src_fnv) {
        err = format("source fingerprint mismatch: model output.weight %zu B fnv %016" PRIx64 ", sidecar made from %" PRIu64 " B fnv %016" PRIx64,
                     src.nbytes, src.fnv, src_bytes, src_fnv);
        return nullptr;
    }

    auto head = std::make_shared<llama_d1_head>();
    head->path      = path;
    head->type      = t->type;
    head->n_embd    = n_embd;
    head->n_vocab   = n_vocab;
    head->row_bytes = ggml_row_size(t->type, n_embd);
    const size_t nbytes = ggml_nbytes(t);
    const size_t offs   = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid);
    if (nbytes != (size_t) n_vocab * head->row_bytes || nbytes != self_bytes) {
        err = format("sidecar tensor bytes %zu != d1.self_bytes %" PRIu64, nbytes, self_bytes);
        return nullptr;
    }

    // [EXACTHEAD] optional certificate stats
    const int64_t gid = gguf_find_tensor(g, "output_exact.gl2");
    size_t gl2_offs = 0;
    uint64_t gl2_fnv = 0;
    if (gid >= 0) {
        ggml_tensor * tg = meta ? ggml_get_tensor(meta, "output_exact.gl2") : nullptr;
        if (!tg || tg->type != GGML_TYPE_F32 || ggml_n_dims(tg) != 2 || n_embd % 32 != 0 ||
            tg->ne[0] != n_embd / 32 || tg->ne[1] != n_vocab) {
            err = "sidecar output_exact.gl2 is not F32 [n_embd/32, n_vocab]";
            return nullptr;
        }
        if (!d1_kv_u64(g, "d1.gl2_fnv64", gl2_fnv)) {
            err = "sidecar has output_exact.gl2 but no d1.gl2_fnv64";
            return nullptr;
        }
        gl2_offs = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, gid);
    }

#ifdef _WIN32
    HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (hf == INVALID_HANDLE_VALUE) { err = format("cannot open %s", path); return nullptr; }
    head->h_file = hf;
    LARGE_INTEGER fs; GetFileSizeEx(hf, &fs);
    head->map_size = (size_t) fs.QuadPart;
    HANDLE hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hm) { err = format("cannot map %s", path); return nullptr; }
    head->h_map = hm;
    head->map_base = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    if (!head->map_base) { err = format("cannot map %s", path); return nullptr; }
#else
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { err = format("cannot open %s", path); return nullptr; }
    struct stat st = {};
    if (fstat(fd, &st) != 0) { close(fd); err = format("cannot stat %s", path); return nullptr; }
    head->map_size = (size_t) st.st_size;
    void * base = mmap(nullptr, head->map_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) { err = format("cannot mmap %s", path); return nullptr; }
    head->map_base = base;
    {
        // [EXACTHEAD] pre-fault the add-on at load (LLAMA_D1_PREFAULT=0 skips): a row the sampler waits on must not
        // come from disk. LLAMA_D1_MLOCK=1 (opt-in) also pins it.
        const char * pf = getenv("LLAMA_D1_PREFAULT");
        if (!(pf && pf[0] == '0')) {
            const auto t0 = std::chrono::steady_clock::now();
            madvise(base, head->map_size, MADV_WILLNEED);
            volatile uint8_t sink = 0;
            const uint8_t * b = (const uint8_t *) base;
            for (size_t o = 0; o < head->map_size; o += 4096) { sink ^= b[o]; }
            (void) sink;
            LLAMA_LOG_WARN("d1: pre-faulted %s (%.1f MiB) in %.0f ms\n", path, head->map_size / 1048576.0,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const char * ml = getenv("LLAMA_D1_MLOCK");
        if (ml && ml[0] == '1') {
            if (mlock(base, head->map_size) != 0) {
                LLAMA_LOG_WARN("d1: LLAMA_D1_MLOCK=1: mlock of %s failed (%s); continuing unlocked\n", path, strerror(errno));
            } else {
                LLAMA_LOG_WARN("d1: LLAMA_D1_MLOCK=1: %s locked in RAM\n", path);
            }
        }
    }
#endif
    if (offs + nbytes > head->map_size) {
        err = "sidecar tensor data is outside the file";
        return nullptr;
    }
    head->rows = (const uint8_t *) head->map_base + offs;

    // (b) self fingerprint over the mapped bytes
    head->self_fnv = llama_d1_fnv64_ends(head->rows, nbytes);
    if (head->self_fnv != self_fnv) {
        err = format("sidecar self fingerprint mismatch: %016" PRIx64 " != d1.self_fnv64 %016" PRIx64, head->self_fnv, self_fnv);
        return nullptr;
    }
    head->src_fnv = src_fnv;

    if (gid >= 0) {
        const int64_t G = n_embd / 32;
        const size_t gbytes = (size_t) G * n_vocab * sizeof(float);
        if (gl2_offs + gbytes > head->map_size) {
            err = "sidecar output_exact.gl2 data is outside the file";
            return nullptr;
        }
        head->gl2     = (const float *) ((const uint8_t *) head->map_base + gl2_offs);
        head->n_group = G;
        if (llama_d1_fnv64_ends((const uint8_t *) head->gl2, gbytes) != gl2_fnv) {
            err = "sidecar output_exact.gl2 fingerprint mismatch";
            return nullptr;
        }
        head->gl2_rowmax.resize(n_vocab);
        float gmax = 0.0f;
        for (int64_t i = 0; i < n_vocab; ++i) {
            const float * e = head->gl2 + i * G;
            float m = 0.0f;
            for (int64_t gg = 0; gg < G; ++gg) {
                if (!(e[gg] >= 0.0f) || !std::isfinite(e[gg])) {
                    err = format("sidecar output_exact.gl2 row %" PRId64 " has a negative or non-finite value", i);
                    return nullptr;
                }
                m = std::max(m, e[gg]);
            }
            head->gl2_rowmax[i] = m;
            gmax = std::max(gmax, m);
        }
        head->gl2_max = gmax;
    }

#ifndef _WIN32
    // rows are read at random token ids: no read-around; optional async read-ahead of the whole tensor (default on)
    madvise((void *) ((uintptr_t) head->rows & ~(uintptr_t) 4095), nbytes + ((uintptr_t) head->rows & 4095), MADV_RANDOM);
    const char * pf = getenv("LLAMA_D1_PREFETCH");
    if (!pf || std::strcmp(pf, "0") != 0) {
        madvise((void *) ((uintptr_t) head->rows & ~(uintptr_t) 4095), nbytes + ((uintptr_t) head->rows & 4095), MADV_WILLNEED);
    }
#endif

    LLAMA_LOG_INFO("d1: fingerprint ok src=%016" PRIx64 " self=%016" PRIx64 " rows=%s (%.2f MiB, %s)%s\n",
                   src_fnv, head->self_fnv, rows_type.c_str(), nbytes / 1048576.0, path,
                   head->gl2 ? format("; certificate stats ok (max group error %.4g)", head->gl2_max).c_str() : "");
    return head;
}

llama_d1_scope_result llama_d1_scope(const llama_d1_scope_in & in, llama_d1_model_slot & slot,
                                     std::shared_ptr<llama_d1_head> & head_out, std::string & msg) {
    head_out.reset();
    if (!in.head_path || !in.head_path[0]) {
        return LLAMA_D1_SCOPE_UNSET;
    }
    if (!in.ctx_default) { msg = "not a DEFAULT context (draft / MTP)"; return LLAMA_D1_SCOPE_OFF; }
    if (in.embeddings)   { msg = "embeddings context";                  return LLAMA_D1_SCOPE_OFF; }
    if (in.mtp_chain)    { msg = "mtp_chain context";                   return LLAMA_D1_SCOPE_OFF; }
    if (in.no_alloc)     { msg = "no_alloc model (memory-fit probe)";   return LLAMA_D1_SCOPE_OFF; }

    // coarse card heads: the AIT2 overlay (type 128), or [EXACTHEAD] the 8 GB file's own AIF25_U2 head (type 125)
    const bool coarse = in.out_type == (int) GGML_TYPE_AIT2_U2_512 || in.out_type == (int) GGML_TYPE_AIF25_U2;
    if (!coarse || in.out_ne0 != in.n_embd || in.out_ne1 != in.n_vocab) {
        msg = format("head type %s [%" PRId64 ", %" PRId64 "], not AIT2_U2_512 or AIF25_U2 [%" PRId64 ", %" PRId64 "]",
                     in.out_type >= 0 ? ggml_type_name((ggml_type) in.out_type) : "none",
                     in.out_ne0, in.out_ne1, in.n_embd, in.n_vocab);
        // the model this sidecar was made from, still on its own base head: the type-128 overlay did not load
        uint64_t sb = 0, sf = 0;
        if (slot.src.valid && in.out_type == slot.src.type && d1_peek_src(in.head_path, sb, sf) &&
            sb == slot.src.nbytes && sf == slot.src.fnv) {
            msg += format("; LLAMA_D1_HEAD is set but this is the sidecar's own base head: the AIT2 overlay "
                          "(LLAMA_MODEL_OVERLAY_PATCH=...head-ait2.gguf) did not load, so the exact head runs on the card "
                          "(no rescore, no VRAM saving)");
            return LLAMA_D1_SCOPE_WARN;
        }
        return LLAMA_D1_SCOPE_OFF;
    }

    std::lock_guard<std::mutex> lk(slot.mtx);
    if (!slot.tried) {
        slot.tried = true;
        slot.head  = llama_d1_head_open(in.head_path, slot.src, in.n_embd, in.n_vocab, slot.err);
    }
    if (!slot.head) {
        msg = slot.err;
        return LLAMA_D1_SCOPE_ERROR;
    }
    head_out = slot.head;
    return LLAMA_D1_SCOPE_ON;
}

// ---- top-N select ------------------------------------------------------------------------------------------------

namespace {
using d1_cand = llama_d1_cand;
// heap order: "a before b" when a is better, so the heap front is the worst kept candidate
struct d1_better {
    bool operator()(const d1_cand & a, const d1_cand & b) const { return a.v > b.v || (a.v == b.v && a.id < b.id); }
};
}

void llama_d1_select_top(const float * row, int64_t n_vocab, int n, std::vector<int32_t> & ids) {
    ids.clear();
    if (n <= 0 || n_vocab <= 0) {
        return;
    }
    if (n >= n_vocab) {
        ids.resize(n_vocab);
        for (int64_t i = 0; i < n_vocab; ++i) { ids[i] = (int32_t) i; }
        return;
    }
    thread_local std::vector<d1_cand> heap;
    heap.resize(n);
    for (int i = 0; i < n; ++i) { heap[i] = { row[i], i }; }
    std::make_heap(heap.begin(), heap.end(), d1_better());
    float thr = heap.front().v;

    // later ids lose ties, so only a strictly larger value can enter
    auto push = [&](int64_t i) {
        if (row[i] > thr) {
            std::pop_heap(heap.begin(), heap.end(), d1_better());
            heap.back() = { row[i], (int32_t) i };
            std::push_heap(heap.begin(), heap.end(), d1_better());
            thr = heap.front().v;
        }
    };
    int64_t i = n;
#ifdef LLAMA_D1_AVX2
    for (; i < n_vocab && (i & 7); ++i) { push(i); }
    __m256 tv = _mm256_set1_ps(thr);
    for (; i + 8 <= n_vocab; i += 8) {
        int m = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(row + i), tv, _CMP_GT_OQ));
        if (m) {
            while (m) {
                const int b = __builtin_ctz(m);
                m &= m - 1;
                push(i + b);
            }
            tv = _mm256_set1_ps(thr);
        }
    }
#endif
    for (; i < n_vocab; ++i) { push(i); }

    ids.resize(n);
    for (int k = 0; k < n; ++k) { ids[k] = heap[k].id; }
}

float llama_d1_scan_chunk(const float * row, int64_t i0, int64_t i1, int n, float D, bool collect,
                          std::vector<llama_d1_cand> & heap, std::vector<int32_t> & buf) {
    heap.clear();
    buf.clear();
    const int64_t m = std::min<int64_t>(n, i1 - i0);
    for (int64_t i = i0; i < i0 + m; ++i) {
        heap.push_back({ row[i], (int32_t) i });
        if (collect) { buf.push_back((int32_t) i); }
    }
    if (i1 - i0 <= n) {
        return -INFINITY;
    }
    std::make_heap(heap.begin(), heap.end(), d1_better());
    float thr = heap.front().v;
    float lo  = collect ? thr - D : thr;
    const size_t cap = 4096;
    const auto visit = [&](int64_t i) {
        const float x = row[i];
        if (collect && x >= lo) { buf.push_back((int32_t) i); }
        if (x > thr) {
            std::pop_heap(heap.begin(), heap.end(), d1_better());
            heap.back() = { x, (int32_t) i };
            std::push_heap(heap.begin(), heap.end(), d1_better());
            thr = heap.front().v;
            if (collect) {
                lo = thr - D;
                // ids below the risen floor can never be needed (the floor only rises); keeps buf small
                if (buf.size() > cap) {
                    size_t w = 0;
                    for (const int32_t id : buf) { if (row[id] >= lo) { buf[w++] = id; } }
                    buf.resize(w);
                }
            }
        }
    };
    // collect: x >= lo (lo <= thr) is the superset test; else x > thr
    int64_t i = i0 + m;
#ifdef LLAMA_D1_AVX2
    for (; i < i1 && (i & 7); ++i) { visit(i); }
    __m256 tv = _mm256_set1_ps(collect ? lo : thr);
    for (; i + 8 <= i1; i += 8) {
        const __m256 x = _mm256_loadu_ps(row + i);
        int mk = _mm256_movemask_ps(collect ? _mm256_cmp_ps(x, tv, _CMP_GE_OQ) : _mm256_cmp_ps(x, tv, _CMP_GT_OQ));
        if (mk) {
            while (mk) {
                const int b = __builtin_ctz(mk);
                mk &= mk - 1;
                visit(i + b);
            }
            tv = _mm256_set1_ps(collect ? lo : thr);
        }
    }
#endif
    for (; i < i1; ++i) { visit(i); }
    return collect ? lo : thr;
}

// ---- per-context state --------------------------------------------------------------------------------------------

llama_d1_ctx::llama_d1_ctx(std::shared_ptr<llama_d1_head> head_, int n_top_, int eager_max_, int n_threads,
                           const llama_d1_cert & cert_)
    : head(std::move(head_)), n_top(n_top_), eager_max(eager_max_), cert(cert_) {
    GGML_ASSERT(!cert.on || head->gl2 != nullptr);
    for (int t = 1; t < n_threads; ++t) {
        pool.threads.emplace_back(&llama_d1_ctx::worker, this, t);
    }
}

llama_d1_ctx::~llama_d1_ctx() {
    drain();
    {
        std::lock_guard<std::mutex> lk(pool.mtx);
        pool.quit = true;
    }
    pool.cv_work.notify_all();
    for (auto & th : pool.threads) {
        th.join();
    }
}

std::string llama_d1_ctx::summary_delta() {
    const uint64_t v[14] = { n_calls.load(), n_rescored.load(), ns_rescore.load(), ns_select.load(), ns_dot.load(), ns_wait.load(),
                             n_minflt.load(), n_majflt.load(), n_read_coarse.load(), n_nvcsw.load(), n_nivcsw.load(),
                             n_cert_rows.load(), n_cert_sigma.load(), n_cert_forced.load() };
    uint64_t d[14];
    for (int i = 0; i < 14; ++i) { d[i] = v[i] - last[i]; last[i] = v[i]; }
    std::string out = format("calls=%" PRIu64 " rows=%" PRIu64 " rescore_ms=%.3f select_ms=%.3f dot_ms=%.3f wait_ms=%.3f minflt=%" PRIu64
                  " majflt=%" PRIu64 " coarse=%" PRIu64 " nvcsw=%" PRIu64 " nivcsw=%" PRIu64,
                  d[0], d[1], d[2] / 1e6, d[3] / 1e6, d[4] / 1e6, d[5] / 1e6, d[6], d[7], d[8], d[9], d[10]);
    if (cert.on) {
        out += format(" cert_rows=%" PRIu64 " cert_sigma=%" PRIu64 " cert_forced=%" PRIu64 " cert_maxrow=%" PRIu64,
                      d[11], d[12], d[13], n_cert_maxrow.load());
    }
    if (audit_every > 0) {
        out += " | " + audit_summary();
    }
    return out;
}

std::string llama_d1_ctx::summary() const {
    std::string out = format("d1: rows_rescored=%" PRIu64 " rows_read_distinct=%" PRIu64 " rows_read_coarse=%" PRIu64
                  " rescore_ms=%.3f select_ms=%.3f h_mismatch=%" PRIu64,
                  n_rescored.load(), n_read_distinct.load(), n_read_coarse.load(),
                  ns_rescore.load() / 1e6, ns_select.load() / 1e6, n_h_mismatch);
    if (cert.on) {
        out += format(" cert_rows=%" PRIu64 " cert_sigma=%" PRIu64 " cert_forced=%" PRIu64 " cert_maxrow=%" PRIu64,
                      n_cert_rows.load(), n_cert_sigma.load(), n_cert_forced.load(), n_cert_maxrow.load());
    }
    if (fused) {
        out += " pick=fused";
    }
    return out;
}

void llama_d1_ctx::reset(int64_t n_rows_max) {
    drain();
    const size_t need = (size_t) n_rows_max * head->n_embd;
    if (h.size() < need) {
        h.resize(need);
    }
    state.assign(n_rows_max, ROW_NONE);
    seen.assign(n_rows_max, 0);
    n_rows = n_rows_max;
}

void llama_d1_ctx::mark_copied(int64_t r0, int64_t n, bool has_h) {
    drain();
    GGML_ASSERT(r0 >= 0 && r0 + n <= n_rows);
    const uint8_t s = (has_h && n_top > 0) ? ROW_PENDING : ROW_COARSE;
    if (!has_h) {
        n_h_mismatch += n;
    }
    std::fill(state.begin() + r0, state.begin() + r0 + n, s);
    std::fill(seen.begin() + r0, seen.begin() + r0 + n, 0);
}

void llama_d1_ctx::swap_rows(int64_t i0, int64_t i1) {
    drain();
    if (i0 >= n_rows || i1 >= n_rows) {
        return;
    }
    std::swap(state[i0], state[i1]);
    std::swap(seen[i0], seen[i1]);
    float * a = h_row(i0);
    float * b = h_row(i1);
    for (int64_t k = 0; k < head->n_embd; ++k) {
        std::swap(a[k], b[k]);
    }
}

void llama_d1_ctx::rescore_one(float * logits, int64_t n_vocab, int64_t r, std::vector<int32_t> & ids) {
    float * row = logits + r * n_vocab;
    thread_local std::vector<float> coarse;
    const bool do_audit = audit_every > 0 && audit_tick.fetch_add(1) % (uint64_t) audit_every == 0;
    if (do_audit) {
        coarse.assign(row, row + n_vocab);
    }
    const auto t0 = std::chrono::steady_clock::now();
    llama_d1_select_top(row, n_vocab, n_top, ids);
    const auto t1 = std::chrono::steady_clock::now();
    const float * hr = h_row(r);
    for (const int32_t id : ids) {
        row[id] = head->dot(hr, id);
    }
    if (cert.on && !ids.empty()) {
        certify(row, n_vocab, hr, ids);
    }
    ns_select += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    ns_dot    += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t1).count();
    if (do_audit) {
        audit_row(row, coarse.data(), hr, n_vocab);
    }
}

// the preset sampler's distribution: top-k 20 (ties to the lower id), softmax at temp 1, top-p 0.95 (keep until the
// cumulative probability reaches p), renormalized. Returns (id, p) sorted by id.
static std::vector<std::pair<int32_t, double>> d1_audit_dist(const float * row, int64_t n_vocab) {
    std::vector<int32_t> ids;
    llama_d1_select_top(row, n_vocab, 20, ids);
    std::sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
    std::vector<double> p(ids.size());
    double z = 0;
    for (size_t i = 0; i < ids.size(); ++i) { p[i] = std::exp((double) row[ids[i]] - row[ids[0]]); z += p[i]; }
    double cum = 0;
    size_t keep = ids.size();
    for (size_t i = 0; i < ids.size(); ++i) {
        cum += p[i] / z;
        if (cum >= 0.95) { keep = i + 1; break; }
    }
    double zk = 0;
    for (size_t i = 0; i < keep; ++i) { zk += p[i]; }
    std::vector<std::pair<int32_t, double>> out;
    for (size_t i = 0; i < keep; ++i) { out.emplace_back(ids[i], p[i] / zk); }
    std::sort(out.begin(), out.end());
    return out;
}

static double d1_tv(const std::vector<std::pair<int32_t, double>> & a, const std::vector<std::pair<int32_t, double>> & b) {
    double tv = 0;
    size_t i = 0, j = 0;
    while (i < a.size() || j < b.size()) {
        if (j == b.size() || (i < a.size() && a[i].first < b[j].first)) { tv += a[i++].second; }
        else if (i == a.size() || b[j].first < a[i].first) { tv += b[j++].second; }
        else { tv += std::fabs(a[i++].second - b[j++].second); }
    }
    return 0.5 * tv;
}

void llama_d1_ctx::audit_row(const float * served, const float * coarse, const float * hr, int64_t n_vocab) {
    std::vector<float> x(n_vocab);
    for (int64_t i = 0; i < n_vocab; ++i) { x[i] = head->dot(hr, i); }
    std::vector<int32_t> ts, tx, tc;
    llama_d1_select_top(served, n_vocab, 20, ts);
    llama_d1_select_top(x.data(), n_vocab, 20, tx);
    llama_d1_select_top(coarse, n_vocab, 20, tc);
    std::sort(ts.begin(), ts.end()); std::sort(tx.begin(), tx.end()); std::sort(tc.begin(), tc.end());
    double dl = 0;
    for (const int32_t id : tx) { dl = std::max(dl, (double) std::fabs(served[id] - x[id])); }
    const auto dx = d1_audit_dist(x.data(), n_vocab);
    const auto ds = d1_audit_dist(served, n_vocab);
    const auto dc = d1_audit_dist(coarse, n_vocab);
    const double tv = d1_tv(ds, dx), tvc = d1_tv(dc, dx);
    // hidden-row alignment check, independent of the exact rows agreeing with themselves: the card's coarse logit
    // (computed on the card from its own h) minus the host exact logit (from the copied h) must look like the coarse
    // head's error, z = |coarse - exact| / s_i of order 1. A misaligned or wrong h gives z in the tens or more.
    double zmax = 0;
    uint64_t z6 = 0, zn = 0;
    if (head->gl2) {
        const int64_t G = head->n_group;
        std::vector<float> hg(G);
        for (int64_t gg = 0; gg < G; ++gg) {
            float s2 = 0.0f;
            for (int q = 0; q < 32; ++q) { s2 += hr[gg * 32 + q] * hr[gg * 32 + q]; }
            hg[gg] = s2 / 32.0f;
        }
        for (const int32_t id : tx) {
            const float * e = head->gl2 + (int64_t) id * G;
            double s2 = 0;
            for (int64_t gg = 0; gg < G; ++gg) { s2 += (double) hg[gg] * e[gg] * e[gg]; }
            const double z = std::fabs((double) coarse[id] - x[id]) / std::max(std::sqrt(s2), 1e-9);
            zmax = std::max(zmax, z);
            z6 += z > 6.0;
            ++zn;
        }
    }
    std::lock_guard<std::mutex> lk(audit_mtx);
    audit.z_max = std::max(audit.z_max, zmax);
    audit.z_gt6 += z6;
    audit.z_n   += zn;
    audit.rows++;
    audit.top_same        += ts == tx;
    audit.coarse_top_same += tc == tx;
    audit.dist_same       += tv == 0.0;
    audit.tv_sum += tv;         audit.tv_max        = std::max(audit.tv_max, tv);
    audit.coarse_tv_sum += tvc; audit.coarse_tv_max = std::max(audit.coarse_tv_max, tvc);
    audit.dlogit_max = std::max(audit.dlogit_max, dl);
    LLAMA_LOG_INFO("d1 audit row %" PRIu64 ": top20_same=%d dlogit_max=%.3g tv=%.3g coarse_top20_same=%d coarse_tv=%.4g z_max=%.3g\n",
                   audit.rows, (int) (ts == tx), dl, tv, (int) (tc == tx), tvc, zmax);
}

std::string llama_d1_ctx::audit_summary() {
    std::lock_guard<std::mutex> lk(audit_mtx);
    const double n = audit.rows ? (double) audit.rows : 1.0;
    return format("d1 audit: rows=%" PRIu64 " top20_same=%" PRIu64 " dist_same=%" PRIu64 " tv_mean=%.3g tv_max=%.3g "
                  "dlogit_max=%.3g | coarse: top20_same=%" PRIu64 " tv_mean=%.4g tv_max=%.4g | h check: z_max=%.3g z_gt6=%" PRIu64 "/%" PRIu64,
                  audit.rows, audit.top_same, audit.dist_same, audit.tv_sum / n, audit.tv_max, audit.dlogit_max,
                  audit.coarse_top_same, audit.coarse_tv_sum / n, audit.coarse_tv_max, audit.z_max, audit.z_gt6, audit.z_n);
}

// [EXACTHEAD] ids are exact in row; rescore every other row whose bound reaches the k-th exact logit (see llama_d1_cert)
float llama_d1_ctx::cert_slack(const float * hr) const {
    const int64_t G = head->n_group;
    double hn2 = 0.0;
    for (int64_t gg = 0; gg < G; ++gg) {
        float s2 = 0.0f;
        for (int q = 0; q < 32; ++q) { s2 += hr[gg * 32 + q] * hr[gg * 32 + q]; }
        hn2 += s2 / 32.0f;
    }
    // 1e-4 relative below the bound: the prefilter only admits more rows, check() decides as before
    return (cert.margin + cert.sigma * head->gl2_max * (float) std::sqrt(hn2)) * 1.0001f;
}

void llama_d1_ctx::cert_prepare(const float * row, const float * hr, const std::vector<int32_t> & ids, cert_prep_t & cp) const {
    const int64_t G = head->n_group;
    thread_local std::vector<float> top;
    top.resize(ids.size());
    for (size_t j = 0; j < ids.size(); ++j) { top[j] = row[ids[j]]; }
    const size_t k = (size_t) std::max(1, std::min<int>(cert.k, (int) ids.size()));
    std::nth_element(top.begin(), top.begin() + (k - 1), top.end(), std::greater<float>());
    cp.T = top[k - 1];
    cp.hg.resize(G);
    double hn2 = 0.0;
    for (int64_t gg = 0; gg < G; ++gg) {
        float s2 = 0.0f;
        for (int q = 0; q < 32; ++q) { s2 += hr[gg * 32 + q] * hr[gg * 32 + q]; }
        cp.hg[gg] = s2 / 32.0f;
        hn2 += cp.hg[gg];
    }
    cp.hn = (float) std::sqrt(hn2);  // s_i <= gl2_rowmax[i] * hn
    // a row below pre cannot pass even with the largest group error of the whole head
    cp.pre = cp.T - cert_slack(hr);
    cp.sids.assign(ids.begin(), ids.end());
    std::sort(cp.sids.begin(), cp.sids.end());
}

// rows in [i0, i1) whose bound reaches T are rescored; writes only row[i0, i1)
void llama_d1_ctx::cert_range(float * row, const float * hr, const cert_prep_t & cp, int64_t i0, int64_t i1,
                              uint64_t & n_sigma, uint64_t & n_forced) const {
    const int64_t G = head->n_group;
    const float T = cp.T, pre = cp.pre, hn = cp.hn;
    const auto check = [&](int64_t i) {
        if (std::binary_search(cp.sids.begin(), cp.sids.end(), (int32_t) i)) { return; } // exact already
        const float c = row[i] + cert.margin;
        if (c + cert.sigma * head->gl2_rowmax[i] * hn < T) { return; }
        const float * e = head->gl2 + i * G;
        float s2 = 0.0f;
        for (int64_t gg = 0; gg < G; ++gg) { s2 += cp.hg[gg] * e[gg] * e[gg]; }
        ++n_sigma;
        if (c + cert.sigma * std::sqrt(s2) >= T) {
            row[i] = head->dot(hr, i);
            ++n_forced;
        }
    };
    int64_t i = i0;
#ifdef LLAMA_D1_AVX2
    for (; i < i1 && (i & 7); ++i) { if (row[i] >= pre) { check(i); } }
    const __m256 pv = _mm256_set1_ps(pre);
    for (; i + 8 <= i1; i += 8) {
        // NaN coarse logits are never selected by the sampler's top-k either
        int m = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(row + i), pv, _CMP_GE_OQ));
        while (m) {
            const int b = __builtin_ctz(m);
            m &= m - 1;
            check(i + b);
        }
    }
#endif
    for (; i < i1; ++i) {
        if (row[i] >= pre) { check(i); }
    }
}

void llama_d1_ctx::cert_count(uint64_t n_sigma, uint64_t n_forced) {
    ++n_cert_rows;
    n_cert_sigma  += n_sigma;
    n_cert_forced += n_forced;
    uint64_t prev = n_cert_maxrow.load();
    while (n_forced > prev && !n_cert_maxrow.compare_exchange_weak(prev, n_forced)) {}
}

// [EXACTHEAD] ids are exact in row; rescore every other row whose bound reaches the k-th exact logit (see llama_d1_cert)
void llama_d1_ctx::certify(float * row, int64_t n_vocab, const float * hr, const std::vector<int32_t> & ids) {
    thread_local cert_prep_t cp;
    cert_prepare(row, hr, ids, cp);
    uint64_t n_sigma = 0, n_forced = 0;
    cert_range(row, hr, cp, 0, n_vocab, n_sigma, n_forced);
    cert_count(n_sigma, n_forced);
}

void llama_d1_ctx::fused_scan(int64_t k) {
    const int64_t ri = k / n_chunk, c = k % n_chunk;
    const int64_t nv = pool.n_vocab;
    const int64_t step = ((nv + n_chunk - 1) / n_chunk + 7) & ~(int64_t) 7;
    const int64_t i0 = std::min(nv, c * step), i1 = std::min(nv, i0 + step);
    const float * row = pool.logits + (*pool.rows)[ri] * nv;
    const auto t0 = std::chrono::steady_clock::now();
    fchunk_t & fc = fch[k];
    llama_d1_scan_chunk(row, i0, i1, n_top, 0.0f, false, fc.heap, fc.buf);
    ns_select += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    // the last chunk of the row merges (acq_rel: sees the other chunks' heaps and bufs)
    if (frow[ri].left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        fused_merge(ri);
    }
}

void llama_d1_ctx::fused_merge(int64_t ri) {
    const auto t0 = std::chrono::steady_clock::now();
    frow_t & fr = frow[ri];
    thread_local std::vector<d1_cand> all;
    all.clear();
    for (int c = 0; c < n_chunk; ++c) {
        const fchunk_t & fc = fch[ri * n_chunk + c];
        all.insert(all.end(), fc.heap.begin(), fc.heap.end());
    }
    // the same set as llama_d1_select_top: the global order (value desc, then id asc), first n_top
    if ((int64_t) all.size() > n_top) {
        std::nth_element(all.begin(), all.begin() + (n_top - 1), all.end(), d1_better());
        all.resize(n_top);
    }
    fr.ids.resize(all.size());
    for (size_t j = 0; j < all.size(); ++j) { fr.ids[j] = all[j].id; }
    fr.audit = audit_every > 0 && audit_tick.fetch_add(1) % (uint64_t) audit_every == 0;
    if (fr.audit) {
        const float * row = pool.logits + (*pool.rows)[ri] * pool.n_vocab;
        fr.coarse.assign(row, row + pool.n_vocab);
    }
    ns_select += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    fr.ready.store(1, std::memory_order_release);
}

void llama_d1_ctx::fused_dots(int64_t k) {
    const int64_t ri = k / n_chunk, c = k % n_chunk;
    frow_t & fr = frow[ri];
    // the row's chunks were all taken before any dot part (phase order), so this wait is short
    while (!fr.ready.load(std::memory_order_acquire)) {
#ifdef LLAMA_D1_AVX2
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }
    const auto t0 = std::chrono::steady_clock::now();
    float * row = pool.logits + (*pool.rows)[ri] * pool.n_vocab;
    const float * hr = h_row((*pool.rows)[ri]);
    for (size_t j = c; j < fr.ids.size(); j += n_chunk) {
        row[fr.ids[j]] = head->dot(hr, fr.ids[j]);
    }
    ns_dot += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    if (fr.dleft.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (cert.on && !fr.ids.empty()) {
            const auto t1 = std::chrono::steady_clock::now();
            cert_prepare(row, hr, fr.ids, fr.cp);
            fr.n_sigma = 0;
            fr.n_forced = 0;
            ns_dot += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t1).count();
            fr.cready.store(1, std::memory_order_release);
        } else {
            fr.cready.store(2, std::memory_order_release); // no certificate: cert parts only count down
        }
    }
}

void llama_d1_ctx::fused_cert(int64_t k) {
    const int64_t ri = k / n_chunk, c = k % n_chunk;
    frow_t & fr = frow[ri];
    // the row's dot parts were all taken before any certificate part (job order), so this wait is short
    int st;
    while (!(st = fr.cready.load(std::memory_order_acquire))) {
#ifdef LLAMA_D1_AVX2
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t nv = pool.n_vocab;
    float * row = pool.logits + (*pool.rows)[ri] * nv;
    const float * hr = h_row((*pool.rows)[ri]);
    if (st == 1) {
        // the same chunk bounds as the scans: each part reads and writes only its own range
        const int64_t step = ((nv + n_chunk - 1) / n_chunk + 7) & ~(int64_t) 7;
        const int64_t i0 = std::min(nv, c * step), i1 = std::min(nv, i0 + step);
        uint64_t ns = 0, nf = 0;
        cert_range(row, hr, fr.cp, i0, i1, ns, nf);
        fr.n_sigma  += ns;
        fr.n_forced += nf;
    }
    ns_dot += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    if (fr.cleft.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (st == 1) {
            cert_count(fr.n_sigma.load(), fr.n_forced.load());
        }
        if (fr.audit) {
            audit_row(row, fr.coarse.data(), hr, nv);
        }
        fr.done.store(1, std::memory_order_release);
    }
}

void llama_d1_ctx::fused_run(int64_t k) {
    const int64_t ri = k / (3 * n_chunk), p = k % (3 * n_chunk);
    if (p < n_chunk) {
        fused_scan(ri * n_chunk + p);
    } else if (p < 2 * n_chunk) {
        fused_dots(ri * n_chunk + p - n_chunk);
    } else {
        fused_cert(ri * n_chunk + p - 2 * n_chunk);
    }
}

void llama_d1_ctx::drain() {
    if (!async_pending) {
        return;
    }
    const auto tw = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lk(pool.mtx);
        pool.cv_done.wait(lk, [&] { return pool.n_busy == 0; });
        pool.rows = nullptr;
    }
    async_pending = false;
    ns_wait += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tw).count();
}

void llama_d1_ctx::run_job() {
    thread_local std::vector<int32_t> ids;
    if (fused) {
        // jobs in row order, each row's scans before its dot parts: a dot part only waits on scans already taken
        const int64_t n_jobs = (int64_t) pool.rows->size() * 3 * n_chunk;
        for (int64_t k; (k = pool.next.fetch_add(1)) < n_jobs; ) {
            fused_run(k);
        }
        return;
    }
    for (;;) {
        const int64_t k = pool.next.fetch_add(1);
        if (k >= (int64_t) pool.rows->size()) {
            break;
        }
        rescore_one(pool.logits, pool.n_vocab, (*pool.rows)[k], ids);
    }
}

void llama_d1_ctx::worker(int /*idx*/) {
    uint64_t seen_gen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(pool.mtx);
            pool.cv_work.wait(lk, [&] { return pool.quit || pool.gen != seen_gen; });
            if (pool.quit) {
                return;
            }
            seen_gen = pool.gen;
        }
        run_job();
        {
            std::lock_guard<std::mutex> lk(pool.mtx);
            if (--pool.n_busy == 0) {
                pool.cv_done.notify_all();
            }
        }
    }
}

void llama_d1_ctx::rescore_rows(float * logits, int64_t n_vocab, const std::vector<int64_t> & rows, int64_t need) {
    drain();
    if (rows.empty()) {
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
#ifndef _WIN32
    struct rusage ru0 = {};
    getrusage(RUSAGE_SELF, &ru0);
#endif
    arows = rows; // the pool may outlive the caller's list (async rows)
    if (fused) {
        // chunks per row: fill the pool, at least fused_min_chunk ids per chunk
        const int64_t nr = (int64_t) rows.size();
        n_chunk = (int) std::max<int64_t>(1, std::min<int64_t>((n_threads_total() + nr - 1) / nr, n_vocab / std::max<int64_t>(1, fused_min_chunk)));
        if (fch.size() < (size_t) (nr * n_chunk)) { fch.resize(nr * n_chunk); }
        if (frow_n < (size_t) nr) { frow.reset(new frow_t[nr]); frow_n = nr; }
        for (int64_t ri = 0; ri < nr; ++ri) {
            frow_t & fr = frow[ri];
            fr.left.store(n_chunk);
            fr.dleft.store(n_chunk);
            fr.cleft.store(n_chunk);
            fr.cready.store(0);
            fr.ready.store(0);
            fr.done.store(0);
        }
    }
    if (pool.threads.empty() || (rows.size() == 1 && (!fused || n_chunk == 1))) {
        pool.logits  = logits;
        pool.n_vocab = n_vocab;
        pool.rows    = &arows;
        pool.next    = 0;
        run_job();
        pool.rows = nullptr;
    } else {
        {
            std::lock_guard<std::mutex> lk(pool.mtx);
            pool.logits  = logits;
            pool.n_vocab = n_vocab;
            pool.rows    = &arows;
            pool.next    = 0;
            pool.n_busy  = (int) pool.threads.size();
            ++pool.gen;
        }
        pool.cv_work.notify_all();
        int64_t need_ri = -1;
        for (size_t ri = 0; fused && need >= 0 && ri < rows.size(); ++ri) {
            if (rows[ri] == need) { need_ri = (int64_t) ri; }
        }
        if (need_ri >= 0) {
            // [EXACTHEAD] async rows: return once the row being read is exact; the pool finishes the later rows
            // while the caller samples this one. Every other entry (and llama_context::synchronize) drains first.
            async_pending = true;
            const int64_t end = (need_ri + 1) * 3 * n_chunk;
            for (int64_t k; pool.next.load() < end && (k = pool.next.fetch_add(1)) < (int64_t) rows.size() * 3 * n_chunk; ) {
                fused_run(k);
            }
            while (!frow[need_ri].done.load(std::memory_order_acquire)) {
#ifdef LLAMA_D1_AVX2
                _mm_pause();
#else
                std::this_thread::yield();
#endif
            }
        } else {
            run_job();
            // block until every worker is done: the sampler must never see a half-written row
            const auto tw = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lk(pool.mtx);
            pool.cv_done.wait(lk, [&] { return pool.n_busy == 0; });
            pool.rows = nullptr;
            ns_wait += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tw).count();
        }
    }
#ifndef _WIN32
    struct rusage ru1 = {};
    getrusage(RUSAGE_SELF, &ru1);
    n_minflt += (uint64_t) (ru1.ru_minflt - ru0.ru_minflt);
    n_majflt += (uint64_t) (ru1.ru_majflt - ru0.ru_majflt);
    n_nvcsw  += (uint64_t) (ru1.ru_nvcsw  - ru0.ru_nvcsw);
    n_nivcsw += (uint64_t) (ru1.ru_nivcsw - ru0.ru_nivcsw);
#endif
    ++n_calls;
    for (const int64_t r : rows) {
        state[r] = ROW_EXACT;
    }
    n_rescored += rows.size();
    ns_rescore += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}

void llama_d1_ctx::on_read_row(float * logits, int64_t n_vocab, int64_t j, int64_t n_out) {
    if (async_pending) {
        // a row the pool already finished can be read; anything else waits for the pool
        bool ok = false;
        for (size_t ri = 0; ri < arows.size(); ++ri) {
            if (arows[ri] == j) { ok = frow[ri].done.load(std::memory_order_acquire) != 0; }
        }
        if (!ok) {
            drain();
        }
    }
    if (j < 0 || j >= n_rows) {
        return;
    }
    if (state[j] == ROW_PENDING) {
        std::vector<int64_t> rows;
        if (n_out <= eager_max) {
            // eager (3.5): a small call (decode / verify) is rescored in full on its first read
            for (int64_t r = 0; r < std::min(n_out, n_rows); ++r) {
                if (state[r] == ROW_PENDING) {
                    rows.push_back(r);
                }
            }
        } else {
            rows.push_back(j);
        }
        rescore_rows(logits, n_vocab, rows, j);
    }
    if (state[j] != ROW_NONE && !seen[j]) {
        seen[j] = 1;
        ++n_read_distinct;
    }
    if (state[j] == ROW_COARSE) {
        ++n_read_coarse;
    }
}

void llama_d1_ctx::on_read_all(float * logits, int64_t n_vocab, int64_t n_out) {
    drain();
    const int64_t n = std::min(n_out, n_rows);
    std::vector<int64_t> rows;
    for (int64_t r = 0; r < n; ++r) {
        if (state[r] == ROW_PENDING) {
            rows.push_back(r);
        }
    }
    rescore_rows(logits, n_vocab, rows);
    for (int64_t r = 0; r < n; ++r) {
        if (state[r] != ROW_NONE && !seen[r]) {
            seen[r] = 1;
            ++n_read_distinct;
        }
        if (state[r] == ROW_COARSE) {
            ++n_read_coarse;
        }
    }
}
