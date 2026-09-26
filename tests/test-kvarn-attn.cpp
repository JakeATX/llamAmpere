// KVarN attention oracle (PRD M3): the CPU kvarn flash-attention path (ops.cpp) against plain
// ggml_flash_attn_ext on the same data. Positions [0,S) and [B,N) are exact fp16 rows (sink + ring),
// [S,B) are records sealed with ggml_kvarn::seal_group; the plain reference sees those records decoded
// back to fp16, so the two graphs must agree up to fp16/fp32 rounding.
//
//   test-kvarn-attn            runs the built-in cases
//   test-kvarn-attn n_q N n_groups cap [nh hkv]

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-kvarn.h"
#include "ggml-quants.h"
#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-memory-hybrid.h"
#include "../src/llama-io.h"
#include "../src/llama-context.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <stdexcept>
#include <string>
#include <vector>

struct cfg { int n_q, N, n_groups, cap, nh, hkv; };

static int run_case(const cfg & c, uint32_t seed, ggml_type staging_type = GGML_TYPE_F16, bool f16_sink = false) {
    const int D = 256, S = 128, G = 128;
    const int nh = c.nh, hkv = c.hkv, n_q = c.n_q, N = c.N, cap = c.cap;
    const int B = S + c.n_groups*G;
    const int qpos0 = N - n_q;
    GGML_ASSERT(N >= B && N - B <= cap && qpos0 >= 0 && nh % hkv == 0 && cap % 128 == 0);
    const int n_kv_pad = (N + 255)/256*256;
    const int n_q_pad  = (n_q + 63)/64*64;
    const ggml_kvarn::layout l = ggml_kvarn::make_layout(D, G, 4, 4);

    std::mt19937 rng(seed);
    auto uni = [&](float lo, float hi) { return lo + (hi - lo) * (float) (rng() / 4294967296.0); };

    // per-position, per-KV-head rows: index (p*hkv + h)*D + d
    std::vector<ggml_fp16_t> Kd((size_t) N*hkv*D), Vd((size_t) N*hkv*D);
    for (size_t i = 0; i < Kd.size(); ++i) { Kd[i] = ggml_fp32_to_fp16(uni(-1.0f, 1.0f)); Vd[i] = ggml_fp32_to_fp16(uni(-1.0f, 1.0f)); }

    // seal [S,B) and replace those rows by their decoded values so both graphs see the same K/V
    std::vector<uint8_t> body((size_t) l.bytes*hkv*c.n_groups);
    std::vector<float> tmp(D);
    for (int g = 0; g < c.n_groups; ++g) {
        for (int h = 0; h < hkv; ++h) {
            uint8_t * rec = body.data() + ((size_t) g*hkv + h)*l.bytes;
            const size_t base = ((size_t) (S + g*G)*hkv + h)*D;
            ggml_kvarn::seal_group(Kd.data() + base, Vd.data() + base, (size_t) hkv*D, l, 16, rec);
            for (int t = 0; t < G; ++t) {
                const size_t row = ((size_t) (S + g*G + t)*hkv + h)*D;
                ggml_kvarn::decode_k_row(rec, l, t, tmp.data());
                for (int d = 0; d < D; ++d) Kd[row + d] = ggml_fp32_to_fp16(tmp[d]);
                ggml_kvarn::decode_v_row(rec, l, t, tmp.data());
                for (int d = 0; d < D; ++d) Vd[row + d] = ggml_fp32_to_fp16(tmp[d]);
            }
        }
    }

    // Keep decoded TQ6 values in F32 so the reference does not add F16 attention error.
    const ggml_type reference_type = staging_type == GGML_TYPE_TQ6_0 ? GGML_TYPE_F32 : GGML_TYPE_F16;
    const size_t mem = ggml_tensor_overhead()*32 + ggml_graph_overhead()
        + (size_t) D*n_q*nh*4 + 2*(size_t) D*n_kv_pad*hkv*ggml_type_size(reference_type) + 2*(size_t) D*(S+cap)*hkv*2
        + (size_t) n_kv_pad*n_q_pad*2 + body.size() + 2*(size_t) D*nh*n_q*4 + 64*1024*1024;
    ggml_init_params ip = { mem, NULL, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_q, nh, 1);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) ((float *) q->data)[i] = uni(-1.0f, 1.0f);

    // plain reference: k/v [D, n_kv_pad, hkv]; rows >= N are zero (masked anyway)
    ggml_tensor * kp = ggml_new_tensor_4d(ctx, reference_type, D, n_kv_pad, hkv, 1);
    ggml_tensor * vp = ggml_new_tensor_4d(ctx, reference_type, D, n_kv_pad, hkv, 1);
    memset(kp->data, 0, ggml_nbytes(kp)); memset(vp->data, 0, ggml_nbytes(vp));
    for (int p = 0; p < N; ++p) for (int h = 0; h < hkv; ++h) {
        if (reference_type == GGML_TYPE_F32) {
            float * pk = (float *) ((char *) kp->data + p*kp->nb[1] + h*kp->nb[2]);
            float * pv = (float *) ((char *) vp->data + p*vp->nb[1] + h*vp->nb[2]);
            if (p >= S && p < B) {
                const uint8_t * record = body.data() + ((size_t) ((p - S)/G)*hkv + h)*l.bytes;
                ggml_kvarn::decode_k_row(record, l, (p - S)%G, pk);
                ggml_kvarn::decode_v_row(record, l, (p - S)%G, pv);
            }
        } else {
            memcpy((char *) kp->data + p*kp->nb[1] + h*kp->nb[2], Kd.data() + ((size_t) p*hkv + h)*D, D*2);
            memcpy((char *) vp->data + p*vp->nb[1] + h*vp->nb[2], Vd.data() + ((size_t) p*hkv + h)*D, D*2);
        }
    }

    // kvarn: k/v ring [D, S+cap, hkv]; unused rows get a sentinel so any misread shows up
    const size_t packed_row = ggml_row_size(staging_type, D*hkv);
    const int alloc_rows = S + cap + (f16_sink ? (S*D*hkv*2 + packed_row - 1)/packed_row : 0);
    auto make_ring = [&]() {
        if (!f16_sink) { return ggml_new_tensor_4d(ctx, staging_type, D, S + cap, hkv, 1); }
        auto * storage = ggml_new_tensor_2d(ctx, staging_type, D*hkv, alloc_rows);
        return ggml_view_4d(ctx, storage, D, alloc_rows, hkv, 1, packed_row,
                ggml_row_size(staging_type, D), ggml_nbytes(storage), 0);
    };
    ggml_tensor * kr = make_ring();
    ggml_tensor * vr = make_ring();
    {
        std::vector<float> sentinel(ggml_nelements(kr), 7.0f);
        ggml_quantize_chunk(staging_type, sentinel.data(), kr->data, 0, alloc_rows*hkv, D, nullptr);
        memcpy(vr->data, kr->data, ggml_nbytes(kr));
    }
    for (int p = 0; p < N; ++p) {
        if (p >= S && p < B) continue;
        const int row = p < S ? p : S + (p - S) % cap;
        for (int h = 0; h < hkv; ++h) {
            auto store = [&](ggml_tensor * ring, ggml_tensor * plain, const ggml_fp16_t * values) {
                void * packed = (char *) ring->data + row*ring->nb[1] + h*ring->nb[2];
                if (f16_sink && p < S) {
                    packed = (char *) ring->data + (size_t) (S + cap)*ring->nb[1] + (size_t) (p*hkv + h)*D*2;
                    memcpy(packed, values, D*2);
                    if (reference_type == GGML_TYPE_F32) {
                        ggml_fp16_to_fp32_row(values, tmp.data(), D);
                        memcpy((char *) plain->data + p*plain->nb[1] + h*plain->nb[2], tmp.data(), D*sizeof(float));
                    }
                } else if (staging_type == GGML_TYPE_TQ6_0) {
                    ggml_fp16_to_fp32_row(values, tmp.data(), D);
                    quantize_row_tq6_0_rotated_ref(tmp.data(), (block_tq6_0 *) packed, D);
                    dequantize_row_tq6_0((const block_tq6_0 *) packed, tmp.data(), D);
                    memcpy((char *) plain->data + p*plain->nb[1] + h*plain->nb[2], tmp.data(), D*sizeof(float));
                } else {
                    memcpy(packed, values, D*2);
                }
            };
            store(kr, kp, Kd.data() + ((size_t) p*hkv + h)*D);
            store(vr, vp, Vd.data() + ((size_t) p*hkv + h)*D);
        }
    }

    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv_pad, n_q_pad, 1, 1);
    for (int j = 0; j < n_q_pad; ++j) for (int p = 0; p < n_kv_pad; ++p) {
        const bool vis = j < n_q ? (p <= qpos0 + j && p < N) : (p < N);
        ((ggml_fp16_t *) m->data)[(size_t) j*n_kv_pad + p] = ggml_fp32_to_fp16(vis ? 0.0f : -INFINITY);
    }

    ggml_tensor * bt = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) std::max<size_t>(body.size(), 1));
    if (!body.empty()) memcpy(bt->data, body.data(), body.size());
    ggml_tensor * desc = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, GGML_KVARN_DESC_N_ENTRIES);
    {
        int32_t * d = (int32_t *) desc->data;
        memset(d, 0, GGML_KVARN_DESC_N_ENTRIES*4);
        d[GGML_KVARN_DESC_S] = S; d[GGML_KVARN_DESC_CAP] = cap; d[GGML_KVARN_DESC_B] = B; d[GGML_KVARN_DESC_N] = N;
        d[GGML_KVARN_DESC_QPOS0] = qpos0; d[GGML_KVARN_DESC_G] = G; d[GGML_KVARN_DESC_D] = D;
        d[GGML_KVARN_DESC_RECBYTES] = (int32_t) l.bytes; d[GGML_KVARN_DESC_HKV] = hkv; d[GGML_KVARN_DESC_B_OLD] = B;
        d[GGML_KVARN_DESC_TYPE_K] = staging_type;
        d[GGML_KVARN_DESC_TYPE_V] = staging_type;
        d[GGML_KVARN_DESC_SINK_TYPE] = f16_sink ? GGML_TYPE_F16 : 0;
    }

    const float scale = 1.0f/sqrtf((float) D);
    ggml_tensor * ref = ggml_flash_attn_ext(ctx, q, kp, vp, m, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(ref, GGML_PREC_F32);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, kr, vr, m, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_kvarn(out, bt, desc, 4, 4, n_kv_pad);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    // compare: out/ref are [D, nh, n_q]
    const float * a = (const float *) out->data;
    const float * b = (const float *) ref->data;
    double se = 0.0, sb = 0.0, maxabs = 0.0; int64_t worst = -1;
    for (int64_t i = 0; i < ggml_nelements(ref); ++i) {
        const double d = (double) a[i] - (double) b[i];
        se += d*d; sb += (double) b[i]*b[i];
        if (fabs(d) > maxabs) { maxabs = fabs(d); worst = i; }
    }
    const double nmse = sb > 0 ? se/sb : se;
    const int64_t w_row = worst / D, w_head = w_row % nh, w_q = w_row / nh;
    const bool ok = nmse < 1e-4 && maxabs < 5e-2;
    printf("  staging=%s sink=%s n_q=%d N=%d B=%d cap=%d nh=%d hkv=%d qpos0=%d: nmse=%.3e maxabs=%.3e (q %ld head %ld pos %d) %s\n",
           ggml_type_name(staging_type), f16_sink ? "f16" : "staging", n_q, N, B, cap, nh, hkv, qpos0, nmse, maxabs, (long) w_q, (long) w_head, qpos0 + (int) w_q, ok ? "OK" : "FAIL");
    ggml_free(ctx);
    return ok ? 0 : 1;
}

// Optional model-backed maintenance gates; the normal CPU oracle needs no model.
struct maintenance_snapshot : llama_io_write_i {
    std::vector<uint8_t> bytes;
    void write(const void * data, size_t size) override {
        const auto * p = static_cast<const uint8_t *>(data);
        bytes.insert(bytes.end(), p, p + size);
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t begin = bytes.size();
        bytes.resize(begin + size);
        ggml_backend_tensor_get(tensor, bytes.data() + begin, offset, size);
    }
    size_t n_bytes() override { return bytes.size(); }
};

static void maintenance_require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static std::vector<int32_t> maintenance_descriptor(llama_kv_cache * cache, llama_pos next_pos) {
    ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * desc = cache->build_input_kvarn_desc(ctx);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    maintenance_require(buffer != nullptr, "allocate maintenance descriptor");
    llama_ubatch ubatch = {};
    ubatch.pos = &next_pos;
    cache->set_input_kvarn_desc(desc, &ubatch);
    std::vector<int32_t> result(GGML_KVARN_DESC_N_ENTRIES);
    ggml_backend_tensor_get(desc, result.data(), 0, result.size()*sizeof(int32_t));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

static int run_maintenance(const char * model_path, int n_gpu_layers, ggml_type staging_type = GGML_TYPE_Q8_0, int sink = 256, ggml_type body_type = GGML_TYPE_F32, bool independent_sink = false, bool fixed_tail = false) {
    const int tail = fixed_tail ? 8192 : staging_type == GGML_TYPE_TQ6_0 ? 2048 : 1024;
    const int tail_max = fixed_tail ? 0 : staging_type == GGML_TYPE_TQ6_0 ? 8192 : 4096;
    const int first_B = sink + 128;
    const int flushed_B = first_B + 24*128;
    const int pressure_start = flushed_B + tail_max;
    const int resumed_end = pressure_start + 2448;
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "maintenance: model load failed\n");
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 16384;
    cp.n_batch = 1024;
    cp.n_ubatch = 1024;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.n_seq_max = 1;
    cp.n_rs_seq = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.kvarn_bits_k = cp.kvarn_bits_v = 4;
    cp.kvarn_sink = sink;
    cp.kvarn_tail = tail;
    cp.kvarn_tail_max = tail_max;
    cp.kvarn_staging_type = staging_type;
    cp.kvarn_body_type = body_type;
    cp.kvarn_sink_type = independent_sink ? GGML_TYPE_F16 : GGML_TYPE_COUNT;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        fprintf(stderr, "maintenance: context creation failed\n");
        return 1;
    }
    int result = 0;
    try {
        auto * memory = llama_get_memory(ctx);
        auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory);
        auto * cache = hybrid ? hybrid->get_mem_attn() : dynamic_cast<llama_kv_cache *>(memory);
        maintenance_require(cache && cache->is_kvarn(), "KVarN attention cache required");
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const std::string text = "int sum(const int * values, int count) { int result = 0; for (int i = 0; i < count; ++i) result += values[i]; return result; }\n";
        std::vector<llama_token> tokens(text.size() + 32);
        const int n_tokens = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, false);
        maintenance_require(n_tokens > 0, "tokenize coding fixture");
        tokens.resize(n_tokens);
        llama_pos next = 0;
        auto append_to = [&](llama_pos end, int width) {
            while (next < end) {
                const int n = std::min<int>(width, end - next);
                llama_batch batch = llama_batch_init(n, 0, 1);
                batch.n_tokens = n;
                for (int j = 0; j < n; ++j) {
                    batch.token[j] = tokens[(next + j) % tokens.size()];
                    batch.pos[j] = next + j;
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = 0;
                    batch.logits[j] = j + 1 == n;
                }
                const int rc = llama_decode(ctx, batch);
                llama_batch_free(batch);
                maintenance_require(rc == 0, "decode fixture");
                next += n;
                const float * logits = llama_get_logits_ith(ctx, -1);
                maintenance_require(logits != nullptr, "decode logits present");
                for (int i = 0; i < n_vocab; ++i) {
                    maintenance_require(std::isfinite(logits[i]), "finite continuation logits");
                }
            }
        };
        auto seal_idle = [&](int expected, const char * name) {
            llama_synchronize(ctx);
            const float * logits = llama_get_logits_ith(ctx, -1);
            std::vector<float> saved(logits, logits + n_vocab);
            maintenance_snapshot before, after;
            if (hybrid) {
                hybrid->get_mem_recr()->state_write(before, 0);
            }
            const auto old_desc = maintenance_descriptor(cache, next);
            const auto old_events = cache->get_kvarn_maintenance_count();
            const auto old_groups = cache->get_kvarn_maintenance_groups();
            const llama_pos old_max = llama_memory_seq_pos_max(memory, 0);
            maintenance_require(llama_kvarn_compress_idle(ctx, 0, next) == expected, name);
            llama_synchronize(ctx);
            if (hybrid) {
                hybrid->get_mem_recr()->state_write(after, 0);
                maintenance_require(before.bytes == after.bytes, "maintenance leaves recurrent state byte-identical");
            }
            maintenance_require(old_max == llama_memory_seq_pos_max(memory, 0), "maintenance preserves sequence positions");
            maintenance_require(memcmp(saved.data(), llama_get_logits_ith(ctx, -1), saved.size()*sizeof(float)) == 0,
                                "maintenance does not evaluate model or change stored logits");
            const auto desc = maintenance_descriptor(cache, next);
            maintenance_require(desc[GGML_KVARN_DESC_N] == old_desc[GGML_KVARN_DESC_N], "maintenance preserves written frontier");
            maintenance_require(desc[GGML_KVARN_DESC_B] >= old_desc[GGML_KVARN_DESC_B], "sealed boundary monotonic");
            maintenance_require(cache->get_kvarn_maintenance_count() == old_events + expected, "one event per nonempty idle flush");
            maintenance_require(cache->get_kvarn_maintenance_groups() == old_groups + (desc[GGML_KVARN_DESC_B] - old_desc[GGML_KVARN_DESC_B])/128, "count all groups across maintenance chunks");
            maintenance_require(!cache->has_kvarn_maintenance(), "no pending boundary after completed maintenance");
            if (expected == 0) {
                maintenance_require(desc == old_desc, "idle noop preserves descriptor");
            }
            printf("maintenance: %s N=%d B=%d->%d recurrent_bytes=%zu OK\n", name, next,
                   old_desc[GGML_KVARN_DESC_B], desc[GGML_KVARN_DESC_B], before.bytes.size());
        };

        if (fixed_tail) {
            int attention = 0, stores = 0, rotations = 0;
            ggml_cgraph * graph = ctx->get_gf_res_reserve()->get_gf();
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                const ggml_tensor * node = ggml_graph_node(graph, i);
                if (ggml_flash_attn_ext_is_kvarn(node)) {
                    ++attention;
                    maintenance_require(node->src[1]->type == GGML_TYPE_TQ6_0 && node->src[2]->type == GGML_TYPE_TQ6_0,
                            "actual cache uses packed TQ6 staging");
                    maintenance_require(node->op_params[7] == body_type, "actual attention body type");
                }
                if (node->op == GGML_OP_SET_ROWS && node->type == GGML_TYPE_TQ6_0) {
                    ++stores;
                    maintenance_require(node->op_params[1] == 1 && node->op_params[2] == sink,
                            "actual writer uses stored-domain TQ6 and independent F16 sink");
                }
                if (node->op == GGML_OP_TURBO_WHT) {
                    ++rotations;
                    int group_size;
                    memcpy(&group_size, node->op_params + sizeof(int), sizeof(int));
                    maintenance_require(group_size == (body_type == GGML_TYPE_TURBO4_0 ? 128 : 256) && node->src[1] == nullptr,
                            "actual model graph uses the requested rotation basis exactly once");
                }
            }
            maintenance_require(attention > 0 && stores == 2*attention && rotations == 4*attention, "all layer cache graph operations checked");
            append_to(sink + tail + 127, 1024);
            maintenance_require(cache->get_kvarn_sealed_end() == (uint32_t) sink, "fixed tail has no premature seal");
            append_to(sink + tail + 128, 1);
            maintenance_require(cache->get_kvarn_sealed_end() == (uint32_t) sink, "first full group waits for the next query");
            append_to(sink + tail + 129, 1);
            maintenance_require(cache->get_kvarn_sealed_end() == (uint32_t) sink + 128, "fixed tail seals its first mature group");
            const auto first = maintenance_descriptor(cache, next);
            const size_t expected_record = body_type == GGML_TYPE_TURBO4_0 ?
                2*128*ggml_row_size(GGML_TYPE_TURBO4_0, first[GGML_KVARN_DESC_D]) :
                ggml_kvarn_rec_bytes(first[GGML_KVARN_DESC_D], 128, 4, 4);
            maintenance_require(first[GGML_KVARN_DESC_BODY_TYPE] == body_type &&
                    (size_t) first[GGML_KVARN_DESC_RECBYTES] == expected_record &&
                    first[GGML_KVARN_DESC_SINK_TYPE] == GGML_TYPE_F16, "actual compact body layout and F16 sink descriptor");
            const llama_pos wrap_end = sink + cache->get_kvarn_capacity() + 2*128;
            append_to(wrap_end, 1024);
            maintenance_require(cache->get_kvarn_maintenance_count() > 1 && cache->get_kvarn_maintenance_groups() > 1,
                    "multiple production seal events before physical ring wrap");
            maintenance_require(next > sink + (int) cache->get_kvarn_capacity(), "physical ring wrap exercised");
            for (int width : {6, 5, 4}) {
                const llama_pos start = next;
                append_to(start + width, width);
                maintenance_require(llama_memory_seq_rm(memory, 0, start + 2, -1), "remove speculative suffix after physical wrap");
                next = start + 2;
                maintenance_require(llama_memory_seq_pos_max(memory, 0) == next - 1, "accepted speculative frontier preserved");
                append_to(next + width, width);
                maintenance_require(cache->get_kvarn_sealed_end() <= (uint32_t) (next - tail), "rollback preserves tail retention");
            }
            maintenance_require(!cache->seq_rm(0, cache->get_kvarn_sealed_end() - 1, -1), "sealed body rollback rejected");
            const auto before_idle = maintenance_descriptor(cache, next);
            maintenance_require(llama_kvarn_compress_idle(ctx, 0, next) == -1, "fixed tail rejects adaptive idle compression");
            maintenance_require(maintenance_descriptor(cache, next) == before_idle, "rejected fixed-tail idle compression preserves descriptor");
            printf("maintenance: fixed tail8192 sink128 body=%s graph_basis=%d ringwrap and rollback6/5/4 OK\n",
                    ggml_type_name(body_type), body_type == GGML_TYPE_TURBO4_0 ? 128 : 256);
        } else {
        append_to(sink + tail, 1024);
        seal_idle(0, "sink plus minimum tail");
        append_to(first_B + tail - 1, 127);
        seal_idle(0, "one token before mature group");
        append_to(first_B + tail, 1);
        seal_idle(1, "first complete mature group");
        maintenance_require(maintenance_descriptor(cache, next)[GGML_KVARN_DESC_B] == first_B, "first idle group boundary");
        seal_idle(0, "repeat idle has no work");

        append_to(flushed_B + tail, 1024);
        maintenance_require(maintenance_descriptor(cache, next)[GGML_KVARN_DESC_B] == first_B, "active tail grows without eager sealing");
        seal_idle(1, "24-group idle flush");
        maintenance_require(maintenance_descriptor(cache, next)[GGML_KVARN_DESC_B] == flushed_B, "idle flush keeps the minimum tail");
        seal_idle(0, "repeat large flush has no work");

        // Verify pressure scheduling during active generation, then cross the physical ring wrap.
        append_to(pressure_start, 1024);
        maintenance_require(maintenance_descriptor(cache, next)[GGML_KVARN_DESC_B] == flushed_B, "no seal below active pressure boundary");
        const auto pressure_events = cache->get_kvarn_maintenance_count();
        append_to(pressure_start + 5, 5);
        maintenance_require(cache->get_kvarn_maintenance_count() == pressure_events + 1, "one pressure event for decode width5");
        const auto pressure = maintenance_descriptor(cache, next);
        maintenance_require(pressure[GGML_KVARN_DESC_B] >= pressure_start - tail, "pressure flush returns tail near minimum");
        append_to(resumed_end, 1024);
        const auto before_invalid = maintenance_descriptor(cache, next);
        maintenance_require(llama_kvarn_compress_idle(ctx, 0, next + 1) == -1, "unmaterialized accepted frontier rejected");
        maintenance_require(llama_kvarn_compress_idle(ctx, 1, next) == -1, "idle cannot compress another sequence");
        maintenance_require(maintenance_descriptor(cache, next) == before_invalid, "invalid maintenance leaves boundary unchanged");
        maintenance_require(!cache->seq_rm(0, before_invalid[GGML_KVARN_DESC_B] - 1, -1), "rollback into sealed body rejected");

        // Resolve a short speculative suffix before idle work; exercise real hybrid rollback.
        append_to(resumed_end + 5, 5);
        const auto unresolved = maintenance_descriptor(cache, next);
        maintenance_require(llama_kvarn_compress_idle(ctx, 0, resumed_end + 2) == -1, "reject idle before speculative suffix removed");
        maintenance_require(maintenance_descriptor(cache, next) == unresolved, "unresolved speculation cannot publish boundary");
        maintenance_require(llama_memory_seq_rm(memory, 0, resumed_end + 2, -1), "rollback unaccepted suffix");
        next = resumed_end + 2;
        seal_idle(1, "idle after speculative suffix rollback");
        const auto rolled = maintenance_descriptor(cache, next);
        maintenance_require(rolled[GGML_KVARN_DESC_B] <= next - tail, "idle uses accepted frontier after rollback");
        append_to(resumed_end + 17, 3);
        seal_idle(0, "resume short generation without unnecessary maintenance");
        }
        llama_memory_clear(memory, true);
        next = 0;
        maintenance_require(llama_kvarn_compress_idle(ctx, 0, 0) == (fixed_tail ? -1 : 0), "empty cache maintenance contract");
        append_to(1024, 1024);
        maintenance_require(maintenance_descriptor(cache, next)[GGML_KVARN_DESC_B] == sink, "clear cancels old boundary");
        printf("maintenance: all model-backed gates passed (%s recurrent state)\n", hybrid ? "checked" : "no");
    } catch (const std::exception & e) {
        fprintf(stderr, "maintenance: FAIL: %s\n", e.what());
        result = 1;
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return result;
}

static int run_tq6_rotated_codec() {
    const int D = 256, n_rows = 4;
    std::mt19937 rng(9831);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> values(D*n_rows), decoded(D*n_rows);
    for (int row = 0; row < n_rows; ++row) {
        const float scale = row == 0 ? 0.0f : row == 1 ? 0.01f : row == 2 ? 1.0f : 8.0f;
        for (int i = 0; i < D; ++i) {
            values[row*D + i] = scale*normal(rng);
        }
    }
    std::vector<block_tq6_0> packed(values.size()/QK_TQ6);
    quantize_row_tq6_0_rotated_ref(values.data(), packed.data(), values.size());
    dequantize_row_tq6_0(packed.data(), decoded.data(), decoded.size());
    double se = 0.0, energy = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(decoded[i])) { return 1; }
        const double error = decoded[i] - values[i];
        se += error*error;
        energy += (double) values[i]*values[i];
    }
    const double nmse = se/energy;
    if (!(nmse < 0.005)) {
        fprintf(stderr, "TQ6 stored-domain codec changed basis: nmse=%g\n", nmse);
        return 1;
    }

    ggml_init_params ip = { 4*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ6_0, D, n_rows + 2);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_rows);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_rows);
    memcpy(src->data, values.data(), values.size()*sizeof(float));
    std::vector<float> sentinel(D*(n_rows + 2), 0.375f);
    quantize_row_tq6_0_rotated_ref(sentinel.data(), (block_tq6_0 *) dst->data, sentinel.size());
    std::vector<uint8_t> untouched(ggml_nbytes(dst));
    memcpy(untouched.data(), dst->data, untouched.size());
    const int64_t indices[n_rows] = {4, 1, 5, 2};
    memcpy(idx->data, indices, sizeof(indices));
    ggml_tensor * written = ggml_set_rows_tq6_rotated(ctx, dst, src, idx);
    ggml_tensor * out = ggml_cpy(ctx, written, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_rows + 2));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_graph_compute_with_ctx(ctx, graph, 4);
    bool ok = true;
    for (int row = 0; row < n_rows; ++row) {
        const float * output = (const float *) ((const char *) out->data + indices[row]*out->nb[1]);
        const void * record = (const char *) dst->data + indices[row]*dst->nb[1];
        ok = ok && memcmp(record, packed.data() + row*(D/QK_TQ6), dst->nb[1]) == 0;
        for (int i = 0; i < D; ++i) {
            ok = ok && std::isfinite(output[i]) && output[i] == decoded[row*D + i];
        }
    }
    for (int row : {0, 3}) {
        ok = ok && memcmp((const char *) dst->data + row*dst->nb[1], untouched.data() + row*dst->nb[1], dst->nb[1]) == 0;
    }
    const int sink_rows = 3, ring_rows = 6;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_TQ6_0, D);
    auto * mixed = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ6_0, D,
            ring_rows + (sink_rows*D*2 + row_bytes - 1)/row_bytes);
    memset(mixed->data, 0x5a, ggml_nbytes(mixed));
    auto * mixed_write = ggml_set_rows_tq6_rotated(ctx, mixed, src, idx);
    mixed_write->op_params[2] = sink_rows;
    mixed_write->op_params[3] = ring_rows;
    auto * mixed_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(mixed_graph, mixed_write);
    ggml_graph_compute_with_ctx(ctx, mixed_graph, 4);
    const auto * sink_data = (const ggml_fp16_t *) ((const char *) mixed->data + ring_rows*row_bytes);
    for (int row = 0; row < n_rows; ++row) {
        if (indices[row] < sink_rows) {
            for (int i = 0; i < D; ++i) {
                ok = ok && sink_data[indices[row]*D + i] == ggml_fp32_to_fp16(values[row*D + i]);
            }
        } else {
            ok = ok && memcmp((const char *) mixed->data + indices[row]*row_bytes,
                    packed.data() + row*(D/QK_TQ6), row_bytes) == 0;
        }
    }
    for (int i = 0; i < D; ++i) { ok = ok && sink_data[i] == 0x5a5a; }
    ggml_free(ctx);
    printf("TQ6 stored-domain codec, SET_ROWS and independent F16 sink: nmse=%.6g, preserved untouched rows: %s\n", nmse, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

static int run_tq6_graph(const char * model_path) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.no_alloc = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 16384;
    cp.n_batch = cp.n_ubatch = 257;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.offload_kqv = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.kvarn_bits_k = cp.kvarn_bits_v = 4;
    cp.kvarn_staging_type = GGML_TYPE_TQ6_0;
    cp.kvarn_sink = 128;
    cp.kvarn_tail = 2048;
    cp.kvarn_tail_max = 8192;
    llama_context * ctx = llama_init_from_model(model, cp);
    bool ok = ctx != nullptr;
    int attention = 0, stores = 0, rotations = 0;
    if (ctx) {
        ggml_cgraph * graph = ctx->get_gf_res_reserve()->get_gf();
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (ggml_flash_attn_ext_is_kvarn(node)) {
                ++attention;
                ok = ok && node->src[1]->type == GGML_TYPE_TQ6_0 && node->src[2]->type == GGML_TYPE_TQ6_0;
            }
            if (node->op == GGML_OP_SET_ROWS && node->type == GGML_TYPE_TQ6_0) {
                ++stores;
                ok = ok && ggml_get_op_params_i32(node, 1) == 1;
            }
            if (node->op == GGML_OP_TURBO_WHT) {
                ++rotations;
                int group_size;
                memcpy(&group_size, node->op_params + sizeof(int), sizeof(int));
                ok = ok && group_size == 256 && node->src[1] == nullptr;
            }
        }
        ok = ok && attention > 0 && stores == 2*attention && rotations == 4*attention;
        llama_free(ctx);
    }
    llama_model_free(model);
    printf("TQ6 model graph (metadata only, CPU): attention=%d raw_stores=%d WHT256=%d, no extra rotation: %s\n", attention, stores, rotations, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

static int run_tq6_seal() {
    const int D = 256, G = 128, hkv = 3, S = 128, cap = 4096, pool = 50;
    const auto layout = ggml_kvarn::make_layout(D, G, 4, 4);
    int passed = 0;
    for (int mode = 0; mode < 4; ++mode) {
        const bool dynamic = mode >= 2;
        const int first = dynamic ? 24 : 0;
        const int count = mode == 3 ? 0 : dynamic ? 24 : 2;
        const int rows = dynamic ? S + cap : count*G;
        const int width = hkv*D + (mode == 1 ? QK_TQ6 : 0);
        ggml_init_params ip = {64*1024*1024, nullptr, false};
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * kbig = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ6_0, width, rows);
        ggml_tensor * vbig = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ6_0, width, rows);
        std::mt19937 rng(8831 + mode);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> values(width*rows), decoded(width*rows);
        std::vector<ggml_fp16_t> kref(width*rows), vref(width*rows);
        for (int side = 0; side < 2; ++side) {
            for (auto & value : values) { value = normal(rng); }
            auto * packed = (block_tq6_0 *) (side == 0 ? kbig : vbig)->data;
            quantize_row_tq6_0_rotated_ref(values.data(), packed, values.size());
            dequantize_row_tq6_0(packed, decoded.data(), decoded.size());
            ggml_fp32_to_fp16_row(decoded.data(), (side == 0 ? kref : vref).data(), decoded.size());
        }
        ggml_tensor * k = ggml_view_2d(ctx, kbig, hkv*D, rows, kbig->nb[1], 0);
        ggml_tensor * v = ggml_view_2d(ctx, vbig, hkv*D, rows, vbig->nb[1], 0);
        ggml_tensor * body = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, layout.bytes*hkv*(dynamic ? pool : count));
        memset(body->data, 0xAA, ggml_nbytes(body));
        std::vector<uint8_t> expected(ggml_nbytes(body), 0xAA);
        for (int g = first; g < first + count; ++g) {
            const int row = dynamic ? S + (g*G)%cap : g*G;
            for (int h = 0; h < hkv; ++h) {
                ggml_kvarn::seal_group(kref.data() + row*width + h*D, vref.data() + row*width + h*D,
                    width, layout, 16, expected.data() + (g*hkv + h)*layout.bytes);
            }
        }
        ggml_tensor * out;
        if (dynamic) {
            ggml_tensor * desc = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, GGML_KVARN_DESC_N_ENTRIES);
            int32_t * d = (int32_t *) desc->data;
            memset(d, 0, ggml_nbytes(desc));
            d[GGML_KVARN_DESC_S] = S; d[GGML_KVARN_DESC_CAP] = cap;
            d[GGML_KVARN_DESC_B_OLD] = S + first*G; d[GGML_KVARN_DESC_B] = S + (first + count)*G;
            d[GGML_KVARN_DESC_G] = G; d[GGML_KVARN_DESC_D] = D;
            d[GGML_KVARN_DESC_HKV] = hkv; d[GGML_KVARN_DESC_RECBYTES] = layout.bytes;
            d[GGML_KVARN_DESC_TYPE_K] = d[GGML_KVARN_DESC_TYPE_V] = GGML_TYPE_TQ6_0;
            out = ggml_kvarn_seal_dyn(ctx, body, k, v, desc, D, G, 4, 4, 16, 24);
        } else {
            out = ggml_kvarn_seal(ctx, body, k, v, D, G, 4, 4, 16);
        }
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_graph_compute_with_ctx(ctx, graph, 4);
        const bool ok = memcmp(body->data, expected.data(), expected.size()) == 0;
        printf("TQ6 seal: mode=%d groups=%d row_bytes=%zu byte parity and untouched records: %s\n", mode, count, k->nb[1], ok ? "OK" : "FAIL");
        passed += ok;
        ggml_free(ctx);
    }
    printf("TQ6 seal: %d/4 cases passed\n", passed);
    return passed == 4 ? 0 : 1;
}

extern "C" void turbo_cpu_fwht_inverse(float * x, int group_size);

static int run_tiered_tq_oracle() {
    constexpr int D = 256, S = 128, G = 128, cap = 256, N = 384, nq = 6;
    ggml_init_params ip = { 32*1024*1024, nullptr, false };
    auto * ctx = ggml_init(ip);
    std::mt19937 rng(20921);
    std::normal_distribution<float> normal(0.0f, 0.6f);
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nq, 1, 1);
    auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N);
    auto * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N);
    for (auto * tensor : {q, k, v}) {
        for (int64_t i = 0; i < ggml_nelements(tensor); ++i) ((float *) tensor->data)[i] = normal(rng);
    }
    const size_t rb = ggml_row_size(GGML_TYPE_TQ6_0, D);
    const int alloc_rows = S+cap + (S*D*2+rb-1)/rb;
    auto * kr = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ6_0, D, alloc_rows);
    auto * vr = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ6_0, D, alloc_rows);
    memset(kr->data, 0x5a, ggml_nbytes(kr));
    memset(vr->data, 0x5a, ggml_nbytes(vr));
    auto * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, N);
    for (int i = 0; i < N; ++i) ((int64_t *) idx->data)[i] = i;
    auto * ks = ggml_set_rows_tq6_rotated(ctx, kr, ggml_turbo_wht(ctx, k, 0, 128, nullptr), idx);
    auto * vs = ggml_set_rows_tq6_rotated(ctx, vr, ggml_turbo_wht(ctx, v, 0, 128, nullptr), idx);
    for (auto * tensor : {ks, vs}) { tensor->op_params[2] = S; tensor->op_params[3] = S+cap; }
    auto * body = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, 2*G*ggml_row_size(GGML_TYPE_TURBO4_0, D));
    body->op_params[7] = GGML_TYPE_TURBO4_0;
    auto * desc = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, GGML_KVARN_DESC_N_ENTRIES);
    auto * dd = (int32_t *) desc->data;
    memset(dd, 0, ggml_nbytes(desc));
    dd[GGML_KVARN_DESC_S]=S; dd[GGML_KVARN_DESC_CAP]=cap; dd[GGML_KVARN_DESC_B]=S+G;
    dd[GGML_KVARN_DESC_B_OLD]=S; dd[GGML_KVARN_DESC_N]=N; dd[GGML_KVARN_DESC_QPOS0]=N-nq;
    dd[GGML_KVARN_DESC_G]=G; dd[GGML_KVARN_DESC_D]=D; dd[GGML_KVARN_DESC_HKV]=1;
    dd[GGML_KVARN_DESC_RECBYTES]=ggml_nbytes(body);
    dd[GGML_KVARN_DESC_TYPE_K]=dd[GGML_KVARN_DESC_TYPE_V]=GGML_TYPE_TQ6_0;
    dd[GGML_KVARN_DESC_BODY_TYPE]=GGML_TYPE_TURBO4_0; dd[GGML_KVARN_DESC_SINK_TYPE]=GGML_TYPE_F16;
    auto * sealed = ggml_kvarn_seal_dyn(ctx, body, ks, vs, desc, D, G, 4, 4, 16, 1);
    auto * kv = ggml_view_4d(ctx, ks, D, S+cap, 1, 1, rb, rb*(S+cap), rb*(S+cap), 0);
    auto * vv = ggml_view_4d(ctx, vs, D, S+cap, 1, 1, rb, rb*(S+cap), rb*(S+cap), 0);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 512, nq);
    for (int j = 0; j < nq; ++j) for (int p = 0; p < 512; ++p) {
        ((ggml_fp16_t *) mask->data)[j*512+p] = ggml_fp32_to_fp16(p <= N-nq+j ? 0.0f : -INFINITY);
    }
    auto * attn = ggml_flash_attn_ext(ctx, ggml_turbo_wht(ctx, q, 0, 128, nullptr), kv, vv, mask, 1.0f/16, 0, 0);
    ggml_flash_attn_ext_set_kvarn(attn, sealed, desc, 4, 4, 512);
    auto * out = ggml_turbo_wht(ctx, attn, 1, 128, nullptr);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_graph_compute_with_ctx(ctx, graph, 4);

    // Independent raw-space reference: native codec entry point, then its scalar inverse transform.
    float inverse[128][128];
    for (int c = 0; c < 128; ++c) {
        float basis[128] = {}; basis[c]=1;
        turbo_cpu_fwht_inverse(basis, 128);
        for (int r = 0; r < 128; ++r) inverse[r][c]=basis[r];
    }
    std::vector<float> decoded_k(N*D), decoded_v(N*D);
    for (int is_v = 0; is_v < 2; ++is_v) for (int p = 0; p < N; ++p) {
        const float * raw = (const float *) (is_v ? v : k)->data + p*D;
        float * decoded = (is_v ? decoded_v : decoded_k).data()+p*D;
        if (p < S) {
            for (int block = 0; block < D/128; ++block) for (int c = 0; c < 128; ++c) {
                double sum=0; for (int r=0; r<128; ++r) sum+=inverse[r][c]*raw[block*128+r];
                decoded[block*128+c]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(sum));
            }
        } else {
            block_tq6_0 packed[D/QK_TQ6];
            quantize_row_tq6_0_ref(raw, packed, D);
            dequantize_row_tq6_0(packed, decoded, D);
            if (p < S+G) {
                for (int i=0; i<D; ++i) decoded[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(decoded[i]));
                block_turbo4_0 low[D/QK_TURBO4];
                quantize_row_turbo4_0_rotated_ref(decoded, low, D);
                dequantize_row_turbo4_0(low, decoded, D);
            }
        }
        for (int b=0; b<D; b+=128) turbo_cpu_fwht_inverse(decoded+b, 128);
    }
    double err=0, energy=0, maxabs=0;
    for (int j=0; j<nq; ++j) {
        double accum[D]={}, total=0;
        for (int p=0; p<=N-nq+j; ++p) {
            double score=0; for (int d=0; d<D; ++d) score+=((float *) q->data)[j*D+d]*decoded_k[p*D+d];
            const double w=exp(score/16); total+=w;
            for (int d=0; d<D; ++d) accum[d]+=w*decoded_v[p*D+d];
        }
        for (int d=0; d<D; ++d) {
            const double expected=accum[d]/total, delta=((float *) out->data)[j*D+d]-expected;
            err+=delta*delta; energy+=expected*expected; maxabs=std::max(maxabs,fabs(delta));
        }
    }
    const bool ok=err/energy<1e-7 && maxabs<1e-4;
    printf("Tiered TQ raw-space native128 oracle: all 3 regions, 6 queries, nmse=%.6g maxabs=%.6g %s\n",err/energy,maxabs,ok?"OK":"FAIL");
    ggml_free(ctx);
    return ok?0:1;
}

int main(int argc, char ** argv) {
    if (argc >= 3 && strcmp(argv[1], "--maintenance-tiered") == 0) {
        return run_maintenance(argv[2], argc > 3 ? atoi(argv[3]) : 99, GGML_TYPE_TQ6_0, 128, GGML_TYPE_TURBO4_0, true, true);
    }
    if (argc >= 3 && strcmp(argv[1], "--maintenance-f16sink-kvarn") == 0) {
        return run_maintenance(argv[2], argc > 3 ? atoi(argv[3]) : 99, GGML_TYPE_TQ6_0, 128, GGML_TYPE_F32, true, true);
    }
    if (argc == 2 && strcmp(argv[1], "--tiered-tq-oracle") == 0) return run_tiered_tq_oracle();
    if (argc == 3 && strcmp(argv[1], "--tq6-graph") == 0) {
        return run_tq6_graph(argv[2]);
    }
    if (argc == 2 && strcmp(argv[1], "--tq6-rotated") == 0) {
        return run_tq6_rotated_codec();
    }
    if (argc == 2 && strcmp(argv[1], "--tq6-seal") == 0) {
        return run_tq6_seal();
    }
    if (argc >= 3 && strcmp(argv[1], "--maintenance-tq6") == 0) {
        return run_maintenance(argv[2], argc > 3 ? atoi(argv[3]) : 99, GGML_TYPE_TQ6_0, 128);
    }
    if (argc >= 3 && strcmp(argv[1], "--maintenance") == 0) {
        return run_maintenance(argv[2], argc > 3 ? atoi(argv[3]) : 99);
    }
    std::vector<cfg> cases;
    const bool f16_sink = argc == 2 && strcmp(argv[1], "--tq6-f16-sink") == 0;
    const bool tq6_attn = f16_sink || (argc == 2 && strcmp(argv[1], "--tq6-attn") == 0);
    if (tq6_attn) {
        cases = {
            { 1, 1152, 0, 2048, 24, 4 },
            { 3, 1536, 3, 2048, 24, 4 },
            { 4, 1536, 3, 2048, 4, 4 },
            { 5, 8704, 35, 4096, 24, 4 },
            { 257, 8128, 60, 8192, 24, 3 },
        };
    } else if (argc >= 5) {
        cfg c = { atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), argc > 5 ? atoi(argv[5]) : 24, argc > 6 ? atoi(argv[6]) : 4 };
        cases.push_back(c);
    } else {
        cases = {
            {    1, 2000,  0, 2176, 24, 4 },  // decode, exact only (B = S)
            {    8, 1153,  8, 2176, 24, 4 },  // decode width 8, one ring row past the body
            {   64, 3500, 10, 2176, 24, 4 },  // body + ring wrap
            { 1024, 3000,  0, 3072, 24, 4 },  // prefill ubatch, exact only
            {  256, 3000,  8, 2048, 24, 4 },  // prefill ubatch, body + ring wrap
            {    1,  500,  2,  256, 16, 2 },  // small: N = B + 116, cap 256
        };
    }
    int fails = 0;
    for (size_t i = 0; i < cases.size(); ++i) fails += run_case(cases[i], 1234 + (uint32_t) i, tq6_attn ? GGML_TYPE_TQ6_0 : GGML_TYPE_F16, f16_sink);
    printf("%s: %d/%zu cases passed\n", fails ? "FAIL" : "OK", (int) (cases.size() - fails), cases.size());
    return fails ? 1 : 0;
}
