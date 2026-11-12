// SJ-KVaRN codec gate (PRD M0): the shared scalar codec and the ggml ops (TURBO_WHT 256, SJKVARN_SEAL)
// against the independent numpy oracle's record file.
//
//   test-sjkvarn-codec <tiles.bin> <records_oracle.bin> <n> [G D bits_k bits_v iters]
//
// tiles.bin: n groups of {K float32 [G][D], V float32 [G][D]} in the unrotated domain.
// records:   n records of ggml_sj_kvarn_rec_bytes(D,G,bits_k,bits_v) bytes.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-sjkvarn.h"
#include "ggml-cuda/sjkvarn-lowbits.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<uint8_t> read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(n);
    if (fread(buf.data(), 1, n, f) != (size_t) n) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    return buf;
}

// rotate every token row with H_D (float32 butterfly) and round to fp16, as the cache does
static void rotate_rows(const float * src, int G, int D, std::vector<ggml_fp16_t> & out) {
    std::vector<float> row(D);
    out.resize((size_t) G*D);
    for (int t = 0; t < G; ++t) {
        memcpy(row.data(), src + (size_t) t*D, D*sizeof(float));
        ggml_sj_kvarn::hadamard(row.data(), D);
        for (int d = 0; d < D; ++d) {
            out[(size_t) t*D + d] = ggml_fp32_to_fp16(row[d]);
        }
    }
}

struct diff_stats {
    size_t rec_mismatch = 0;
    size_t meta_bytes_diff = 0;
    size_t payload_vals = 0, payload_diff = 0, payload_diff_gt1 = 0;
    size_t noise_rows = 0, noise_diff = 0; // rows whose fp16 step is subnormal: codes carry no information
};

// A K channel / V token row whose quantization step (the stored fp16 scale) is below the fp16 normal range is a
// (near-)constant row: its 16 codes span less than 1e-3 of the row value, so float32 and float64 balancing land on
// arbitrary codes. Such rows are counted separately and excluded from the "differs by more than 1" gate.
static bool noise_row(const uint8_t * rec, size_t scale_off) {
    ggml_fp16_t h;
    memcpy(&h, rec + scale_off, 2);
    return fabsf(GGML_FP16_TO_FP32(h)) < 6.103515625e-05f; // 2^-14
}

// The oracle writes the PRD section-3 record: token-major bit stream per payload and channel metadata in natural
// order. The sealer's record is fragment-ordered (ggml_sj_kvarn::code_bit / k_ch_idx / v_ch_idx); re-pack the oracle
// record into that layout so the two can be compared code by code.
static std::vector<uint8_t> oracle_to_layout(const uint8_t * o, const ggml_sj_kvarn::layout & l) {
    std::vector<uint8_t> r(o, o + l.bytes);
    const bool fk = ggml_sj_kvarn::frag_order(l.bits_k, l.D, l.G);
    const bool fv = ggml_sj_kvarn::frag_order(l.bits_v, l.D, l.G);
    if (!fk && !fv) {
        return r;
    }
    // oracle slot of value d in a row: 4-bit rows interleave each 8-value word (value j at nibble j/2 for even j,
    // 4 + j/2 for odd j); other widths are the identity
    auto slot = [](int d, int bits) -> uint32_t {
        if (bits != 4) return (uint32_t) d;
        const int j = d & 7;
        return (uint32_t) ((d & ~7) + ((j & 1) ? 4 + (j >> 1) : (j >> 1)));
    };
    memset(r.data(), 0, l.k_scale);
    for (int t = 0; t < l.G; ++t) {
        for (int d = 0; d < l.D; ++d) {
            const uint32_t k = ggml_sj_kvarn::unpack_code(o + l.k_payload, (uint32_t) t*(l.D*l.bits_k) + slot(d, l.bits_k)*l.bits_k, l.bits_k);
            const uint32_t v = ggml_sj_kvarn::unpack_code(o + l.v_payload, (uint32_t) t*(l.D*l.bits_v) + slot(d, l.bits_v)*l.bits_v, l.bits_v);
            ggml_sj_kvarn::pack_code(r.data() + l.k_payload, ggml_sj_kvarn::code_bit(t, d, false, l.D, l.G, l.bits_k), l.bits_k, k);
            ggml_sj_kvarn::pack_code(r.data() + l.v_payload, ggml_sj_kvarn::code_bit(t, d, true,  l.D, l.G, l.bits_v), l.bits_v, v);
        }
    }
    for (int d = 0; d < l.D; ++d) {
        memcpy(r.data() + l.k_scale + 2*ggml_sj_kvarn::k_ch_idx(d, l.D, l.G, l.bits_k), o + l.k_scale + 2*d, 2);
        memcpy(r.data() + l.k_zero  + 2*ggml_sj_kvarn::k_ch_idx(d, l.D, l.G, l.bits_k), o + l.k_zero  + 2*d, 2);
        memcpy(r.data() + l.v_ch    + 2*ggml_sj_kvarn::v_ch_idx(d, l.D, l.G, l.bits_v), o + l.v_ch    + 2*d, 2);
    }
    return r;
}

static void compare_records(const uint8_t * a, const uint8_t * b, const ggml_sj_kvarn::layout & l, diff_stats & st) {
    bool any = false;
    for (size_t i = l.k_scale; i < l.bytes; ++i) {
        if (a[i] != b[i]) { st.meta_bytes_diff++; any = true; }
    }
    std::vector<bool> k_noise(l.D), v_noise(l.G);
    for (int d = 0; d < l.D; ++d) { k_noise[d] = noise_row(a, l.k_scale + 2*ggml_sj_kvarn::k_ch_idx(d, l.D, l.G, l.bits_k)); st.noise_rows += k_noise[d]; }
    for (int t = 0; t < l.G; ++t) { v_noise[t] = noise_row(a, l.v_scale + 2*t); st.noise_rows += v_noise[t]; }
    for (int t = 0; t < l.G; ++t) {
        for (int d = 0; d < l.D; ++d) {
            const int qa = ggml_sj_kvarn::k_code(a, l, t, d);
            const int qb = ggml_sj_kvarn::k_code(b, l, t, d);
            st.payload_vals++;
            if (qa != qb) {
                any = true;
                if (k_noise[d]) { st.noise_diff++; } else { st.payload_diff++; if (abs(qa - qb) > 1) st.payload_diff_gt1++; }
            }
            const int va = ggml_sj_kvarn::v_code(a, l, t, d);
            const int vb = ggml_sj_kvarn::v_code(b, l, t, d);
            st.payload_vals++;
            if (va != vb) {
                any = true;
                if (v_noise[t]) { st.noise_diff++; } else { st.payload_diff++; if (abs(va - vb) > 1) st.payload_diff_gt1++; }
            }
        }
    }
    if (any) st.rec_mismatch++;
}

static int test_stage_strips() {
    size_t checked = 0;
    for (int bk : {2, 3}) for (int bv : {2, 3}) {
        const auto l = ggml_sj_kvarn::make_layout(256, 128, bk, bv);
        std::vector<uint8_t> source(l.bytes), stage(l.bytes + 4096 + 16);
        for (int seed = 0; seed < 3; ++seed) {
            uint32_t rng = 12345 + seed;
            for (auto & b : source) { rng = rng*1664525u + 1013904223u; b = rng >> 24; }
            for (unsigned mask : {1u, 128u, 129u, 85u, 170u, 255u, 16u, 3u}) {
                std::fill(stage.begin(), stage.end(), 0xcd);
                memcpy(stage.data(), source.data() + l.k_scale, 512);
                memcpy(stage.data() + 512, source.data() + l.k_zero, 512);
                memcpy(stage.data() + 1024, source.data() + l.v_ch, 512);
                int rank = 0;
                for (int s = 0; s < 8; ++s) {
                    if (!(mask & (1u << s))) continue;
                    const int kb = 256*(2*bk + 1), vb = 256*(2*bv + 1);
                    uint8_t * out = stage.data() + 1536 + rank++*sj_kvarn_lowbits::stage_strip_bytes(bk, bv);
                    for (int b = 0; b < kb + vb; ++b) {
                        const bool v = b >= kb;
                        const int from = sj_kvarn_lowbits::stage_source_byte(v ? bv : bk, s, v ? b - kb : b);
                        if (from >= 4096*(v ? bv : bk)) return 1;
                        out[b] = from < 0 ? 0 : source[(v ? l.v_payload : 0) + from];
                    }
                    memcpy(out + kb + vb, source.data() + l.k_tok + s*32, 32);
                    memcpy(out + kb + vb + 32, source.data() + l.v_scale + s*32, 32);
                    memcpy(out + kb + vb + 64, source.data() + l.v_zero + s*32, 32);
                    for (int v = 0; v < 2; ++v) {
                        const uint8_t * packed = out + (v ? kb : 0);
                        const uint8_t * full = source.data() + (v ? l.v_payload : 0);
                        const int bits = v ? bv : bk;
                        for (int t = 0; t < 16; ++t) for (int d = 0; d < 256; ++d) {
                            const uint32_t got = bits == 3 ? sj_kvarn_lowbits::stage_window<3>(packed, t, d) : sj_kvarn_lowbits::stage_window<2>(packed, t, d);
                            const uint32_t want = bits == 3 ? ggml_sj_kvarn::trellis_lb_window_any<3>(full, s*16 + t, d, 256, 128, true)
                                                            : ggml_sj_kvarn::trellis_lb_window_any<2>(full, s*16 + t, d, 256, 128, true);
                            if (got != want) { fprintf(stderr, "stage window mismatch %d/%d mask=%u s=%d t=%d d=%d v=%d\n", bk, bv, mask, s, t, d, v); return 1; }
                            ++checked;
                        }
                    }
                    if (memcmp(out + kb + vb, source.data() + l.k_tok + s*32, 32) ||
                        memcmp(out + kb + vb + 32, source.data() + l.v_scale + s*32, 32) ||
                        memcmp(out + kb + vb + 64, source.data() + l.v_zero + s*32, 32)) return 1;
                }
                if (memcmp(stage.data(), source.data() + l.k_scale, 512) ||
                    memcmp(stage.data() + 512, source.data() + l.k_zero, 512) ||
                    memcmp(stage.data() + 1024, source.data() + l.v_ch, 512)) return 1;
                const size_t end = 1536 + rank*sj_kvarn_lowbits::stage_strip_bytes(bk, bv);
                for (size_t i = end; i < stage.size(); ++i) if (stage[i] != 0xcd) return 1;
            }
        }
    }
    printf("compact stage: %zu exact trellis indices, shared/token metadata, changing strips, bounds PASS\n", checked);
    return 0;
}

static int test_strip_master_seal() {
    const int32_t table[] = {0, 1, 3, 0, 2};
    const int rows[] = {384, 384, 128, 128, 256, 256};
    const int records[] = {6, 7, 0, 1, 4, 5};
    for (int bk : {2, 3}) for (int bv : {2, 3}) {
        sj_kvarn_lowbits::strip_master_layout m;
        if (!sj_kvarn_lowbits::make_strip_master_layout(bk, bv, m)) return 1;
        std::vector<uint8_t> scratch(8*size_t(m.canonical_bytes), 0xda);
        std::vector<uint8_t> master(8*size_t(m.bytes) + 32, 0xcd), expected = master;
        for (int i = 0; i < 8; ++i) {
            int row, record;
            const int active = sj_kvarn_lowbits::strip_master_seal_location(128, 384, 768, 384, 512, 4,
                table, 5, 8, 2, i, row, record);
            if (i >= 6) { if (active || row != -1 || record != -1) return 1; continue; }
            if (active != 1 || row != rows[i] || record != records[i]) return 1;
            for (int b = 0; b < m.canonical_bytes; ++b) scratch[size_t(i)*m.canonical_bytes + b] = uint8_t(i*37 + b*13 + b/251);
            const uint8_t * src = scratch.data() + size_t(i)*m.canonical_bytes;
            uint8_t * dst = master.data() + 16 + size_t(record)*m.bytes;
            for (int b = 0; b < m.bytes; ++b) {
                const int from = sj_kvarn_lowbits::strip_master_source_byte(m, b);
                dst[b] = from < 0 ? 0 : src[from];
            }
            if (!sj_kvarn_lowbits::strip_master_set_records(m, expected.data() + 16, 8, src,
                    size_t(records[i])*m.canonical_bytes, m.canonical_bytes)) return 1;
        }
        if (master != expected) return 1;
        // A seal can change nonadjacent physical pages; the card refresh must recover each head's canonical bytes.
        std::vector<uint8_t> refreshed(size_t(m.canonical_bytes) + 32, 0x6d);
        for (int i = 0; i < 6; ++i) {
            if (!sj_kvarn_lowbits::strip_master_get_records(m, master.data() + 16, 8, refreshed.data() + 16,
                    size_t(records[i])*m.canonical_bytes, m.canonical_bytes) ||
                memcmp(refreshed.data() + 16, scratch.data() + size_t(i)*m.canonical_bytes, m.canonical_bytes)) return 1;
            for (int b = 0; b < 16; ++b) if (refreshed[b] != 0x6d || refreshed[16 + m.canonical_bytes + b] != 0x6d) return 1;
        }
        for (int page : {2, 3}) for (int b = 0; b < m.bytes; ++b) {
            if (master[16 + size_t(page)*m.bytes + b] != 0xcd) return 1;
        }
    }
    int row, record;
    // No table, no new groups, short table, bad physical ID, ring bounds and a too-small scratch range.
    if (sj_kvarn_lowbits::strip_master_seal_location(128, 384, 768, 384, 512, 4,
            nullptr, 0, 10, 2, 4, row, record) != 1 || row != 256 || record != 8 ||
        sj_kvarn_lowbits::strip_master_seal_location(128, 384, 384, 384, 512, 4,
            table, 5, 8, 2, 0, row, record) != 0) return 1;
    const int32_t bad_table[] = {0, 1, -1, 4, 2};
    for (int i = 0; i < 10; ++i) {
        const int active = sj_kvarn_lowbits::strip_master_seal_location(i == 0 ? -1 : 128,
            i == 1 ? 385 : 384, i == 2 ? 769 : 768, i == 3 ? 127 : 384,
            i == 4 ? 511 : 512, i == 5 ? 2 : 4, i == 6 || i == 7 ? bad_table : table,
            i == 8 ? 4 : 5, i == 9 ? SIZE_MAX : 8, 2, i == 7 ? 2 : i == 8 ? 4 : 0, row, record);
        if (active != -1 || row != -1 || record != -1) return 1;
    }
    return 0;
}

static int test_strip_master_clock() {
    for (int bk = 2; bk <= 3; ++bk) for (int bv = 2; bv <= 3; ++bv) {
        sj_kvarn_lowbits::strip_master_layout l, checked;
        if (!sj_kvarn_lowbits::make_strip_master_layout(bk, bv, l)) return 1;
        const size_t stride = 4*size_t(l.canonical_bytes), stage_stride = 4*size_t(l.bytes), capacity = 3*stage_stride;
        if (!sj_kvarn_lowbits::strip_master_clock_layout(bk, bv, 3, 4, stride, stage_stride, capacity, checked)) return 1;
        if (checked.bytes != l.bytes || checked.canonical_bytes != l.canonical_bytes) return 1;
        for (int bad = 0; bad < 10; ++bad) {
            const size_t records = bad == 0 ? 0 : bad == 1 ? SIZE_MAX : bad == 2 ? size_t(INT32_MAX) : 3;
            const size_t heads = bad == 3 ? 0 : bad == 4 ? SIZE_MAX : 4;
            if (sj_kvarn_lowbits::strip_master_clock_layout(bad == 5 ? 4 : bk, bad == 9 ? 4 : bv, records, heads,
                    stride + (bad == 6), stage_stride + (bad == 7), capacity - (bad == 8), checked)) return 1;
        }
        std::vector<uint8_t> canonical(3*stride), master(capacity), output(2*stride + 32, 0xda), expected(output);
        for (size_t i = 0; i < canonical.size(); ++i) canonical[i] = uint8_t(i*37 + (i/stride)*53 + i/71);
        if (!sj_kvarn_lowbits::strip_master_set_records(l, master.data(), 12, canonical.data(), 0, canonical.size())) return 1;
        for (int r = 0; r < 12; ++r) for (int b = 0; b < l.bytes; ++b) {
            const int from = sj_kvarn_lowbits::strip_master_source_byte(l, b);
            if (from < 0 || sj_kvarn_lowbits::strip_master_byte(l, from) != b) master[size_t(r)*l.bytes + b] ^= 0xff;
        }
        const int records[] = {2, 0}, slots[] = {1, 0};
        for (int i = 0; i < 2; ++i) for (int head = 0; head < 4; ++head) {
            const uint8_t * src = master.data() + (size_t(records[i])*4 + head)*l.bytes;
            uint8_t * dst = output.data() + 16 + (size_t(slots[i])*4 + head)*l.canonical_bytes;
            std::vector<uint8_t> writes(l.canonical_bytes, 0);
            for (int lane = 0; lane < 128; ++lane) for (int q = lane; q < l.bytes/16; q += 128) {
                uint8_t value[16];
                memcpy(value, src + q*16, 16);
                for (int b = 0; b < 16; ++b) {
                    const int from = sj_kvarn_lowbits::strip_master_source_byte(l, q*16 + b);
                    if (from >= 0 && sj_kvarn_lowbits::strip_master_byte(l, from) == q*16 + b) {
                        dst[from] = value[b];
                        ++writes[from];
                    }
                }
            }
            for (uint8_t n : writes) if (n != 1) return 1;
            memcpy(expected.data() + 16 + (size_t(slots[i])*4 + head)*l.canonical_bytes,
                    canonical.data() + (size_t(records[i])*4 + head)*l.canonical_bytes, l.canonical_bytes);
        }
        if (output != expected) return 1;
    }
    return 0;
}

// Independent token-axis oracle: read each bit separately, without the production
// window helpers or strip offsets. This is a CPU check, not a CUDA arithmetic test.
static int test_strip_master_decode() {
    if (!ggml_sj_kvarn::trellis3::tokens()) {
        fprintf(stderr, "strip-master decode oracle requires token-axis trellis\n");
        return 1;
    }
    size_t checked = 0;
    for (int bk : {2, 3}) for (int bv : {2, 3}) {
        const auto l = ggml_sj_kvarn::make_layout(256, 128, bk, bv);
        sj_kvarn_lowbits::strip_master_layout m;
        if (!sj_kvarn_lowbits::make_strip_master_layout(bk, bv, m)) return 1;
        std::vector<uint8_t> source(l.bytes), master(m.bytes), restored(l.bytes);
        for (int seed = 0; seed < 3; ++seed) {
            uint32_t rng = 98765u + seed;
            for (auto & b : source) { rng = rng*1664525u + 1013904223u; b = rng >> 24; }
            for (int d = 0; d < 256; ++d) {
                ggml_sj_kvarn::put_half(source.data(), l.k_scale + 2*d, (1 + d%7)*0.0625f);
                ggml_sj_kvarn::put_half(source.data(), l.k_zero + 2*d, (d%5 - 2)*0.125f);
                ggml_sj_kvarn::put_half(source.data(), l.v_ch + 2*d, (1 + d%3)*0.25f);
            }
            for (int t = 0; t < 128; ++t) {
                ggml_sj_kvarn::put_half(source.data(), l.k_tok + 2*t, (1 + t%3)*0.25f);
                ggml_sj_kvarn::put_half(source.data(), l.v_scale + 2*t, (1 + t%7)*0.0625f);
                ggml_sj_kvarn::put_half(source.data(), l.v_zero + 2*t, (t%5 - 2)*0.125f);
            }
            if (!sj_kvarn_lowbits::strip_master_set(m, master.data(), source.data(), 0, m.canonical_bytes) ||
                !sj_kvarn_lowbits::strip_master_get(m, master.data(), restored.data(), 0, m.canonical_bytes) || restored != source) return 1;
            for (int v = 0; v < 2; ++v) {
                const int bits = v ? bv : bk;
                const uint8_t * payload = source.data() + (v ? l.v_payload : l.k_payload);
                const uint16_t * cb = bits == 2 ? ggml_sj_kvarn::trellis2::cb(v != 0) : ggml_sj_kvarn::trellis3::cb(v != 0);
                for (int t = 0; t < 128; ++t) {
                    float actual[256];
                    if (v) ggml_sj_kvarn::decode_v_row(restored.data(), l, t, actual, true);
                    else   ggml_sj_kvarn::decode_k_row(restored.data(), l, t, actual, true);
                    for (int d = 0; d < 256; ++d) {
                        const int history = std::min(6, t*bits), begin = (d*128 + t)*bits - history;
                        uint32_t win = 0;
                        for (int b = 0; b < bits + history; ++b) {
                            const int bit = begin + b;
                            win |= uint32_t((payload[bit/8] >> (bit%8)) & 1) << (6 - history + b);
                        }
                        const uint32_t full = bits == 2 ? ggml_sj_kvarn::trellis_lb_window_any<2>(payload, t, d, 256, 128, true)
                                                        : ggml_sj_kvarn::trellis_lb_window_any<3>(payload, t, d, 256, 128, true);
                        const uint8_t * strip = master.data() + 1536 + (t/16)*m.strip_bytes + (v ? m.k_strip : 0);
                        const uint32_t packed = bits == 2 ? sj_kvarn_lowbits::stage_window<2>(strip, t%16, d)
                                                          : sj_kvarn_lowbits::stage_window<3>(strip, t%16, d);
                        const float q = GGML_FP16_TO_FP32(cb[win]);
                        const float scale = v ? (1 + t%7)*0.0625f : (1 + d%7)*0.0625f;
                        const float zero = v ? (t%5 - 2)*0.125f : (d%5 - 2)*0.125f;
                        const float gain = v ? (1 + d%3)*0.25f : (1 + t%3)*0.25f;
                        const float want = (q*scale + zero)*gain;
                        if (win != full || win != packed || !std::isfinite(actual[d]) || actual[d] != want) {
                            fprintf(stderr, "strip-master CPU decode %d/%d seed=%d v=%d t=%d d=%d\n", bk, bv, seed, v, t, d);
                            return 1;
                        }
                        ++checked;
                    }
                }
            }
        }
    }
    printf("strip-master CPU decode: %zu bitwise windows and scalar values, all four width pairs PASS (CUDA untested)\n", checked);
    return 0;
}

static int test_strip_master_expand() {
    if (test_strip_master_decode()) return 1;
    for (bool master : {false, true}) for (int64_t queries : {int64_t(1), int64_t(8), int64_t(9), int64_t(64), int64_t(127), int64_t(128), int64_t(512), INT64_MAX}) {
        const bool safe = sj_kvarn_lowbits::strip_master_prefill_fallback_safe(master, queries);
        const bool want = !master || queries == 1 || queries == 8;
        if (safe != want) {
            fprintf(stderr, "strip-master prefill fallback: master=%d queries=%lld\n", int(master), (long long) queries);
            return 1;
        }
    }
    printf("strip-master prefill fallback: 16 OFF/decode/tile/prefill boundary cases PASS\n");
    for (int bk : {2, 3}) for (int bv : {2, 3}) {
        sj_kvarn_lowbits::strip_master_expand_plan p;
        const size_t record = size_t(4096*(bk + bv) + 2304), block = 4*(record + 65536);
        for (size_t budget : {block, 3*block - 1, 3*block, 64*block, SIZE_MAX}) {
            if (!sj_kvarn_lowbits::make_strip_master_expand_plan(bk, bv, 4096, 4, budget, p)) return 1;
            const size_t blocks = budget/block < 64 ? budget/block : 64;
            if (p.rows != int(64*blocks) || p.scratch_bytes != blocks*4*record || p.output_bytes != blocks*4*32768 ||
                p.total_bytes != p.scratch_bytes + 2*p.output_bytes || p.total_bytes > budget) return 1;
            if (p.rows < 4096 && budget - p.total_bytes >= block) return 1;
        }
        if (!sj_kvarn_lowbits::make_strip_master_expand_plan(bk, bv, 64, 1, record + 65536, p) ||
            p.rows != 64 || p.scratch_bytes != record || p.output_bytes != 32768) return 1;
        for (int rows : {64, 256, 4096}) for (int heads : {1, 4, 16}) {
            for (size_t existing : {size_t(0), size_t(1), size_t(127), size_t(128), size_t(65537)}) {
                const size_t aligned = (existing + 127)/128*128;
                const size_t want = aligned + size_t(rows/64)*heads*record;
                for (size_t budget : {want - 1, want, SIZE_MAX}) {
                    size_t offset = 7, total = 9;
                    const bool ok = sj_kvarn_lowbits::strip_master_prefill_workspace(bk, bv, rows, heads,
                        existing, budget, offset, total);
                    if (ok != (budget >= want) || (ok && (offset != aligned || total != want)) ||
                        (!ok && (offset || total))) return 1;
                }
            }
        }
        for (int bad = 0; bad < 8; ++bad) {
            size_t offset = 7, total = 9;
            if (sj_kvarn_lowbits::strip_master_prefill_workspace(bad == 0 ? 4 : bk, bv,
                    bad == 1 ? 0 : bad == 2 ? 65 : 64, bad == 3 ? 0 : bad == 4 ? 65536 : 1,
                    bad == 5 ? SIZE_MAX : bad == 6 ? SIZE_MAX - 127 : 128,
                    bad == 7 ? 128 : SIZE_MAX, offset, total) || offset || total) return 1;
        }
        if (SIZE_MAX > UINT32_MAX) {
            const uint64_t canonical = bk + bv == 4 ? 18688 : bk + bv == 5 ? 22784 : 26880;
            const uint64_t scratch = 33554431ull*65535ull*canonical;
            const uint64_t output = 2147483584ull*65535ull*512;
            if (!sj_kvarn_lowbits::make_strip_master_expand_plan(bk, bv, INT32_MAX - 63, 65535, SIZE_MAX, p) ||
                p.rows != INT32_MAX - 63 || p.scratch_bytes != scratch || p.output_bytes != output ||
                p.total_bytes != scratch + 2*output) return 1;
        }
        for (int bad = 0; bad < 11; ++bad) {
            p = {64, 1, 2, 5};
            if (sj_kvarn_lowbits::make_strip_master_expand_plan(bad == 0 ? 4 : bk, bad == 1 ? 1 : bv,
                    bad == 2 ? 0 : bad == 3 ? -64 : bad == 4 ? 65 : bad == 8 ? INT32_MAX : 4096,
                    bad == 5 ? 0 : bad == 6 ? 65536 : bad == 9 ? -1 : 4,
                    bad == 7 ? block - 1 : bad == 10 ? 0 : SIZE_MAX, p) ||
                p.rows || p.scratch_bytes || p.output_bytes || p.total_bytes) return 1;
        }
    }
    const int32_t table[] = {2, 0, 1};
    for (int group = 0; group < 3; ++group) for (int half = 0; half < 2; ++half) {
        int record;
        if (!sj_kvarn_lowbits::strip_master_expand_location(128, 512, 128 + group*128 + half*64, table, 3, 3, record) ||
            record != table[group] || !sj_kvarn_lowbits::strip_master_expand_location(128, 512, 128 + group*128 + half*64,
                nullptr, 0, 3, record) || record != group) return 1;
    }
    for (int bad = 0; bad < 13; ++bad) {
        int record = 99;
        const int32_t invalid[] = {2, bad == 9 ? -1 : 3, 1};
        if (sj_kvarn_lowbits::strip_master_expand_location(bad == 0 ? -128 : bad == 1 ? 64 : 128,
                bad == 2 ? 0 : bad == 3 ? 513 : 512,
                bad == 4 ? 64 : bad == 5 ? 512 : bad == 6 ? 257 : bad == 12 ? 128 : 256,
                bad == 9 || bad == 10 ? invalid : table, bad == 7 ? 1 : 3,
                bad == 8 ? 0 : bad == 11 ? SIZE_MAX : bad == 12 ? 2 : 3, record) || record != -1) return 1;
    }
    for (int bk : {2, 3}) for (int bv : {2, 3}) {
        sj_kvarn_lowbits::strip_master_layout l;
        if (!sj_kvarn_lowbits::make_strip_master_layout(bk, bv, l)) return 1;
        const size_t stride = 4*size_t(l.canonical_bytes);
        std::vector<uint8_t> source(3*stride), master(12*size_t(l.bytes)), card(stride);
        for (size_t i = 0; i < source.size(); ++i) source[i] = uint8_t(i*37 + i/67 + (i/stride)*13);
        if (!sj_kvarn_lowbits::strip_master_set_records(l, master.data(), 12, source.data(), 0, source.size())) return 1;
        memcpy(card.data(), source.data(), stride);
        for (int r = 0; r < 12; ++r) for (int b = 0; b < l.bytes; ++b) {
            const int from = sj_kvarn_lowbits::strip_master_source_byte(l, b);
            if (from < 0 || sj_kvarn_lowbits::strip_master_byte(l, from) != b) master[size_t(r)*l.bytes + b] ^= 0xff;
        }
        // Two nonzero heads, reordered pages, one resident page, and both halves of each record.
        std::vector<uint8_t> scratch(2*6*size_t(l.canonical_bytes) + 32, 0xcd), expected(scratch);
        for (int y = 0; y < 2; ++y) for (int x = 0; x < 6; ++x) {
            int record;
            if (!sj_kvarn_lowbits::strip_master_expand_location(128, 512, 128 + x*64, table, 3, 3, record)) return 1;
            const int head = y + 1;
            uint8_t * dst = scratch.data() + 16 + size_t(y*6 + x)*l.canonical_bytes;
            if (record == 0) {
                memcpy(dst, card.data() + size_t(head)*l.canonical_bytes, l.canonical_bytes);
            } else {
                const uint8_t * src = master.data() + (size_t(record)*4 + head)*l.bytes;
                for (int lane = 0; lane < 256; ++lane) for (int q = lane; q < l.bytes/16; q += 256) {
                    for (int b = 0; b < 16; ++b) {
                        const int from = sj_kvarn_lowbits::strip_master_source_byte(l, q*16 + b);
                        if (from >= 0 && sj_kvarn_lowbits::strip_master_byte(l, from) == q*16 + b) dst[from] = src[q*16 + b];
                    }
                }
            }
            memcpy(expected.data() + 16 + size_t(y*6 + x)*l.canonical_bytes,
                    source.data() + (size_t(record)*4 + head)*l.canonical_bytes, l.canonical_bytes);
        }
        if (scratch != expected) return 1;
    }
    return 0;
}

static int test_strip_master_arenas() {
    using namespace sj_kvarn_lowbits;
    for (int bk : {2, 3}) for (int bv : {2, 3}) {
        strip_master_layout layouts[2];
        if (!make_strip_master_layout(bk, bv, layouts[0]) || !make_strip_master_layout(bv, bk, layouts[1])) return 1;
        const size_t canonical = 2*size_t(layouts[0].canonical_bytes), physical = 2*size_t(layouts[0].bytes);
        const strip_master_arena arenas[] = {{bk, bv, 31, 2, physical + 13}, {bv, bk, 31 + physical + 60, 2, physical + 19}};
        const size_t bytes = arenas[1].base + arenas[1].capacity + 37;
        const size_t spans[][2] = {{0, bytes}, {0, 17}, {26, 19}, {31, canonical},
            {31 + canonical - 7, arenas[1].base + 19 - (31 + canonical - 7)},
            {31 + canonical + 5, 17}, {31 + physical + 10, 57},
            {arenas[1].base + canonical - 3, bytes - (arenas[1].base + canonical - 3)},
            {bytes - 11, 11}, {bytes, 0}};
        for (const auto & span : spans) {
            std::vector<uint8_t> storage(bytes + 32, 0xcd), expected(bytes, 0xcd);
            std::vector<uint8_t> source[2] = {std::vector<uint8_t>(canonical), std::vector<uint8_t>(canonical)};
            uint8_t * buffer = storage.data() + 16;
            for (size_t r = 0; r < 2; ++r) {
                for (size_t i = 0; i < canonical; ++i) source[r][i] = uint8_t(i*29 + r*71 + i/251);
                if (!strip_master_set_records(layouts[r], buffer + arenas[r].base, 2, source[r].data(), 0, canonical)) return 1;
            }
            memcpy(expected.data(), buffer, bytes);
            memset(expected.data() + span[0], 0x93, span[1]);
            for (size_t r = 0; r < 2; ++r) {
                const auto & a = arenas[r];
                memcpy(expected.data() + a.base, buffer + a.base, a.capacity);
                const size_t first = std::max(span[0], a.base), last = std::min(span[0] + span[1], a.base + canonical);
                if (first < last) memset(source[r].data() + first - a.base, 0x93, last - first);
                for (size_t record = 0; record < 2; ++record) for (int b = 0; b < layouts[r].bytes; ++b) {
                    const int from = strip_master_source_byte(layouts[r], b);
                    expected[a.base + record*size_t(layouts[r].bytes) + size_t(b)] = from < 0 ? 0 : source[r][record*size_t(layouts[r].canonical_bytes) + size_t(from)];
                }
            }
            if (!strip_master_fill_arenas(buffer, bytes, arenas, 2, 0x93, span[0], span[1]) || memcmp(buffer, expected.data(), bytes)) return 1;
            for (size_t i = 0; i < 16; ++i) if (storage[i] != 0xcd || storage[bytes + 16 + i] != 0xcd) return 1;
        }
        for (int bad = 0; bad < 12; ++bad) {
            strip_master_arena invalid[] = {arenas[0], arenas[1]};
            size_t offset = 0, size = bytes;
            if (bad == 0) std::swap(invalid[0], invalid[1]);
            if (bad == 1) invalid[1] = invalid[0];
            if (bad == 2) invalid[1].capacity = physical - 1;
            if (bad == 3) invalid[1].base = bytes + 1;
            if (bad == 4) invalid[1].records = SIZE_MAX;
            if (bad == 5) invalid[1].bits_k = 4;
            if (bad == 6) invalid[1].bits_v = 1;
            if (bad == 7) invalid[1].capacity = SIZE_MAX;
            if (bad == 8) invalid[1].base = invalid[0].base + invalid[0].capacity - 1;
            if (bad == 9) invalid[1].records = 0;
            if (bad == 10) offset = bytes + 1;
            if (bad == 11) size = SIZE_MAX;
            std::vector<uint8_t> buffer(bytes, 0xcd), before = buffer;
            if (strip_master_fill_arenas(buffer.data(), bytes, invalid, 2, 0, offset, size) || buffer != before) return 1;
        }
        if (strip_master_fill_arenas(nullptr, bytes, arenas, 2, 0, 0, 1) ||
            strip_master_fill_arenas(nullptr, bytes, nullptr, 2, 0, 0, 0) ||
            !strip_master_fill_arenas(nullptr, bytes, arenas, 2, 0, bytes, 0)) return 1;
    }
    return 0;
}

static int test_strip_master_records() {
    if (test_strip_master_arenas()) return 1;
    if (test_strip_master_expand()) return 1;
    if (test_strip_master_clock()) return 1;
    if (test_strip_master_seal()) return 1;
    for (int bk : {2, 3}) for (int bv : {2, 3}) {
        sj_kvarn_lowbits::strip_master_layout m;
        if (!sj_kvarn_lowbits::make_strip_master_layout(bk, bv, m)) return 1;
        size_t canonical, physical;
        if (!sj_kvarn_lowbits::strip_master_records_size(m, 3, canonical, physical) ||
            canonical != 3*size_t(m.canonical_bytes) || physical != 3*size_t(m.bytes)) return 1;
        std::vector<uint8_t> source(canonical), master(physical + 32, 0xcd), update(canonical);
        for (size_t i = 0; i < canonical; ++i) { source[i] = uint8_t(i*29 + i/251); update[i] = uint8_t(i*13 + 7); }
        uint8_t * arena = master.data() + 16;
        if (!sj_kvarn_lowbits::strip_master_set_records(m, arena, 3, source.data(), 0, canonical)) return 1;
        const auto exact = [&]() {
            for (size_t i = 0; i < 16; ++i) if (master[i] != 0xcd || master[physical + 16 + i] != 0xcd) return false;
            for (size_t r = 0; r < 3; ++r) for (int b = 0; b < m.bytes; ++b) {
                const int from = sj_kvarn_lowbits::strip_master_source_byte(m, b);
                if (arena[r*size_t(m.bytes) + size_t(b)] != (from < 0 ? 0 : source[r*size_t(m.canonical_bytes) + size_t(from)])) return false;
            }
            return true;
        };
        if (!exact()) return 1;
        // State I/O offsets are relative to a view, but the packed arena starts at the root.
        const size_t bases[] = {0, size_t(m.canonical_bytes) - 7, 2*size_t(m.canonical_bytes) - 3};
        for (size_t base : bases) {
            const size_t bytes = canonical - base;
            for (size_t offset : {size_t(0), size_t(5), bytes}) {
                const size_t n = std::min<size_t>(33, bytes - offset);
                size_t absolute = SIZE_MAX;
                if (!sj_kvarn_lowbits::strip_master_view_range(m, 3, base, bytes, offset, n, absolute) || absolute != base + offset) return 1;
                std::vector<uint8_t> restored(n + 16, 0xab);
                if (!sj_kvarn_lowbits::strip_master_set_records(m, arena, 3, update.data(), absolute, n)) return 1;
                memcpy(source.data() + absolute, update.data(), n);
                if (!exact() || !sj_kvarn_lowbits::strip_master_get_records(m, arena, 3, restored.data(), absolute, n) ||
                    memcmp(restored.data(), update.data(), n)) return 1;
                for (size_t i = n; i < restored.size(); ++i) if (restored[i] != 0xab) return 1;
                if (!sj_kvarn_lowbits::strip_master_fill_records(m, arena, 3, 0x93, absolute, n)) return 1;
                memset(source.data() + absolute, 0x93, n);
                if (!exact()) return 1;
            }
        }
        for (int bad = 0; bad < 6; ++bad) {
            size_t absolute = SIZE_MAX;
            const size_t base = bad == 0 ? canonical + 1 : bad == 4 ? SIZE_MAX : canonical - 8;
            const size_t bytes = bad == 1 ? 9 : bad == 5 ? SIZE_MAX : 8;
            const size_t offset = bad == 2 ? 9 : 7, n = bad == 3 ? 2 : 1;
            if (sj_kvarn_lowbits::strip_master_view_range(m, 3, base, bytes, offset, n, absolute) || absolute) return 1;
        }
        // Gather each mask from reordered master records and compare to the canonical stage oracle.
        const auto layout = ggml_sj_kvarn::make_layout(256, 128, bk, bv);
        std::vector<uint8_t> stage(physical + 32), expected(physical + 32);
        for (unsigned mask = 0; mask < 256; ++mask) {
            std::fill(stage.begin(), stage.end(), 0xcd);
            std::fill(expected.begin(), expected.end(), 0xcd);
            for (int i = 0; i < 3; ++i) {
                const int record = i == 2 && (mask & 1u) ? mask%3 : (mask + 2*i)%3;
                const unsigned wanted = (mask + 73*i) & 255u;
                const uint8_t * src = arena + size_t(record)*m.bytes;
                const uint8_t * full = source.data() + size_t(record)*m.canonical_bytes;
                uint8_t * dst = stage.data() + 16 + size_t(i)*m.bytes;
                uint8_t * oracle = expected.data() + 16 + size_t(i)*m.bytes;
                for (int q = 0; q < 1536/16; ++q) memcpy(dst + q*16, src + q*16, 16);
                memcpy(oracle, full + layout.k_scale, 512);
                memcpy(oracle + 512, full + layout.k_zero, 512);
                memcpy(oracle + 1024, full + layout.v_ch, 512);
                int rank = 0;
                for (int s = 0; s < 8; ++s) {
                    if (!(wanted & (1u << s))) continue;
                    uint8_t * out = dst + 1536 + rank*m.strip_bytes;
                    uint8_t * want = oracle + 1536 + rank++*m.strip_bytes;
                    for (int q = 0; q < m.strip_bytes/16; ++q)
                        memcpy(out + q*16, src + 1536 + s*m.strip_bytes + q*16, 16);
                    for (int b = 0; b < m.k_strip + m.v_strip; ++b) {
                        const bool v = b >= m.k_strip;
                        const int from = sj_kvarn_lowbits::stage_source_byte(v ? bv : bk, s, v ? b - m.k_strip : b);
                        want[b] = from < 0 ? 0 : full[(v ? layout.v_payload : 0) + from];
                    }
                    memcpy(want + m.k_strip + m.v_strip, full + layout.k_tok + s*32, 32);
                    memcpy(want + m.k_strip + m.v_strip + 32, full + layout.v_scale + s*32, 32);
                    memcpy(want + m.k_strip + m.v_strip + 64, full + layout.v_zero + s*32, 32);
                }
            }
            if (stage != expected) return 1;
        }
        // Reordered scratch records round-trip; poison all history copies to catch duplicate scatter writes.
        const int ids[] = {2, 0, 1};
        std::vector<uint8_t> packed(physical, 0xcd), unpacked(canonical + 32, 0xda);
        for (int i = 0; i < 3; ++i) for (int q = 0; q < m.bytes/16; ++q) {
            uint8_t block[16];
            for (int b = 0; b < 16; ++b) {
                const int from = sj_kvarn_lowbits::strip_master_source_byte(m, q*16 + b);
                block[b] = from < 0 ? 0 : source[size_t(i)*m.canonical_bytes + size_t(from)];
            }
            memcpy(packed.data() + size_t(ids[i])*m.bytes + q*16, block, 16);
        }
        for (int i = 0; i < 3; ++i) {
            std::vector<uint8_t> oracle(m.bytes);
            if (!sj_kvarn_lowbits::strip_master_set(m, oracle.data(), source.data() + size_t(i)*m.canonical_bytes, 0, m.canonical_bytes) ||
                memcmp(oracle.data(), packed.data() + size_t(ids[i])*m.bytes, m.bytes)) return 1;
            for (int b = 0; b < m.bytes; ++b) {
                const int from = sj_kvarn_lowbits::strip_master_source_byte(m, b);
                if (from < 0 || sj_kvarn_lowbits::strip_master_byte(m, from) != b) packed[size_t(ids[i])*m.bytes + b] ^= 0xff;
            }
            for (int q = 0; q < m.bytes/16; ++q) {
                uint8_t block[16];
                memcpy(block, packed.data() + size_t(ids[i])*m.bytes + q*16, 16);
                for (int b = 0; b < 16; ++b) {
                    const int from = sj_kvarn_lowbits::strip_master_source_byte(m, q*16 + b);
                    if (from >= 0 && sj_kvarn_lowbits::strip_master_byte(m, from) == q*16 + b)
                        unpacked[16 + size_t(i)*m.canonical_bytes + size_t(from)] = block[b];
                }
            }
        }
        if (memcmp(unpacked.data() + 16, source.data(), canonical)) return 1;
        for (size_t i = 0; i < 16; ++i) if (unpacked[i] != 0xda || unpacked[canonical + 16 + i] != 0xda) return 1;
        const size_t spans[][2] = {{size_t(m.canonical_bytes) - 17, 33},
                                   {size_t(m.canonical_bytes + m.k_payload) - 2, 33},
                                   {3, canonical - 7}, {0, canonical}, {canonical, 0}};
        for (const auto & span : spans) {
            const size_t offset = span[0], size = span[1];
            if (!sj_kvarn_lowbits::strip_master_set_records(m, arena, 3, update.data(), offset, size)) return 1;
            memcpy(source.data() + offset, update.data(), size);
            if (!exact()) return 1;
            std::vector<uint8_t> out(size + 32, 0xda);
            if (!sj_kvarn_lowbits::strip_master_get_records(m, arena, 3, out.data() + 16, offset, size) ||
                memcmp(out.data() + 16, source.data() + offset, size)) return 1;
            for (size_t i = 0; i < 16; ++i) if (out[i] != 0xda || out[size + 16 + i] != 0xda) return 1;
            if (!sj_kvarn_lowbits::strip_master_fill_records(m, arena, 3, 0x7b, offset, size)) return 1;
            memset(source.data() + offset, 0x7b, size);
            if (!exact()) return 1;
        }
        const auto before = master;
        const auto source_before = source;
        std::vector<uint8_t> rejected(canonical, 0x39);
        const size_t bad_spans[][2] = {{canonical, 1}, {canonical + 1, 0}, {SIZE_MAX, 1}, {1, SIZE_MAX}};
        for (const auto & span : bad_spans) {
            if (sj_kvarn_lowbits::strip_master_set_records(m, arena, 3, source.data(), span[0], span[1]) ||
                sj_kvarn_lowbits::strip_master_get_records(m, arena, 3, rejected.data(), span[0], span[1]) ||
                sj_kvarn_lowbits::strip_master_fill_records(m, arena, 3, 0, span[0], span[1])) return 1;
            for (uint8_t b : rejected) if (b != 0x39) return 1;
        }
        if (sj_kvarn_lowbits::strip_master_set_records(m, nullptr, 3, source.data(), 0, 1) ||
            sj_kvarn_lowbits::strip_master_set_records(m, arena, 3, nullptr, 0, 1) ||
            sj_kvarn_lowbits::strip_master_get_records(m, nullptr, 3, rejected.data(), 0, 1) ||
            sj_kvarn_lowbits::strip_master_get_records(m, arena, 3, nullptr, 0, 1) ||
            sj_kvarn_lowbits::strip_master_fill_records(m, nullptr, 3, 0, 0, 1)) return 1;
        for (uint8_t b : rejected) if (b != 0x39) return 1;
        for (size_t records : {size_t(0), size_t(3)}) {
            const size_t end = records*size_t(m.canonical_bytes);
            if (!sj_kvarn_lowbits::strip_master_set_records(m, nullptr, records, nullptr, end, 0) ||
                !sj_kvarn_lowbits::strip_master_get_records(m, nullptr, records, nullptr, end, 0) ||
                !sj_kvarn_lowbits::strip_master_fill_records(m, nullptr, records, 0, end, 0)) return 1;
        }
        const size_t max_records = SIZE_MAX/size_t(m.bytes);
        if (!sj_kvarn_lowbits::strip_master_records_size(m, max_records, canonical, physical) ||
            !sj_kvarn_lowbits::strip_master_records_range(m, max_records, canonical, 0)) return 1;
        for (size_t records : {max_records + 1, size_t(SIZE_MAX)}) {
            canonical = physical = 123;
            if (sj_kvarn_lowbits::strip_master_records_size(m, records, canonical, physical) || canonical || physical ||
                sj_kvarn_lowbits::strip_master_set_records(m, arena, records, source.data(), 0, 1) ||
                sj_kvarn_lowbits::strip_master_get_records(m, arena, records, source.data(), 0, 1) ||
                sj_kvarn_lowbits::strip_master_fill_records(m, arena, records, 0, 0, 1)) return 1;
        }
        if (master != before || source != source_before) return 1;
        const sj_kvarn_lowbits::strip_master_layout bad{};
        if (sj_kvarn_lowbits::strip_master_records_range(bad, 0, 0, 0)) return 1;
    }
    return 0;
}

static int test_strip_master() {
    if (test_strip_master_records()) return 1;
    size_t checked = 0;
    for (int bk : {2, 3}) for (int bv : {2, 3}) {
        sj_kvarn_lowbits::strip_master_layout m;
        if (!sj_kvarn_lowbits::make_strip_master_layout(bk, bv, m)) return 1;
        const auto l = ggml_sj_kvarn::make_layout(256, 128, bk, bv);
        if (m.canonical_bytes != int(l.bytes) || m.bytes != int(l.bytes) + 4096) return 1;
        if (m.bytes%32 || m.strip_bytes%32 || m.k_strip%32 || m.v_strip%32) return 1;
        std::vector<uint8_t> source(l.bytes), master(m.bytes + 32), restored(l.bytes);
        std::vector<bool> unique(m.bytes, false);
        for (int b = 0; b < m.canonical_bytes; ++b) {
            const int to = sj_kvarn_lowbits::strip_master_byte(m, b);
            if (to < 0 || to >= m.bytes || unique[to] || sj_kvarn_lowbits::strip_master_source_byte(m, to) != b) return 1;
            unique[to] = true;
        }
        for (int seed = 0; seed < 3; ++seed) {
            uint32_t rng = 12345 + seed;
            for (auto & b : source) { rng = rng*1664525u + 1013904223u; b = rng >> 24; }
            std::fill(master.begin(), master.end(), 0xcd);
            for (int b = 0; b < m.bytes; ++b) {
                const int from = sj_kvarn_lowbits::strip_master_source_byte(m, b);
                if (from < -1 || from >= m.canonical_bytes) return 1;
                master[b] = from < 0 ? 0 : source[from];
            }
            for (int b = 0; b < m.canonical_bytes; ++b) restored[b] = master[sj_kvarn_lowbits::strip_master_byte(m, b)];
            if (source != restored) return 1;
            std::vector<uint8_t> written(m.bytes + 32, 0xcd);
            if (!sj_kvarn_lowbits::strip_master_set(m, written.data(), source.data(), 0, m.canonical_bytes) || written != master) return 1;
            if (!sj_kvarn_lowbits::strip_master_get(m, written.data(), restored.data(), 0, m.canonical_bytes) || restored != source) return 1;
            for (int b = m.bytes; b < int(master.size()); ++b) if (master[b] != 0xcd) return 1;
            if (memcmp(master.data(), source.data() + l.k_scale, 1024) ||
                memcmp(master.data() + 1024, source.data() + l.v_ch, 512)) return 1;
            for (int s = 0; s < 8; ++s) {
                const uint8_t * strip = master.data() + 1536 + s*m.strip_bytes;
                const uint8_t * meta = strip + m.k_strip + m.v_strip;
                if (memcmp(meta, source.data() + l.k_tok + s*32, 32) ||
                    memcmp(meta + 32, source.data() + l.v_scale + s*32, 32) ||
                    memcmp(meta + 64, source.data() + l.v_zero + s*32, 32)) return 1;
                for (int v = 0; v < 2; ++v) {
                    const int bits = v ? bv : bk;
                    const uint8_t * payload = strip + (v ? m.k_strip : 0);
                    const uint8_t * full = source.data() + (v ? l.v_payload : 0);
                    for (int d = 0; d < 256; ++d) {
                        if (payload[d*(2*bits + 1)] != (s ? full[d*16*bits + s*2*bits - 1] : 0)) return 1;
                        for (int t = 0; t < 16; ++t) {
                            const uint32_t got = bits == 3 ? sj_kvarn_lowbits::stage_window<3>(payload, t, d) : sj_kvarn_lowbits::stage_window<2>(payload, t, d);
                            const uint32_t want = bits == 3 ? ggml_sj_kvarn::trellis_lb_window_any<3>(full, s*16 + t, d, 256, 128, true)
                                                            : ggml_sj_kvarn::trellis_lb_window_any<2>(full, s*16 + t, d, 256, 128, true);
                            if (got != want) return 1;
                            ++checked;
                        }
                    }
                }
            }
            const int offsets[] = {0, 3, 5, 16*bk - 1, m.k_payload - 1, m.k_payload + 2*bv - 1,
                                   m.k_payload + m.v_payload - 1, int(l.k_tok) - 1, int(l.v_ch) - 1,
                                   int(l.v_scale) - 1, int(l.v_zero) - 1, m.canonical_bytes - 17};
            const uint8_t update[33] = {0xe1, 0x93, 0x6a, 0x12};
            for (int offset : offsets) {
                const int size = std::min(33, m.canonical_bytes - offset);
                if (!sj_kvarn_lowbits::strip_master_set(m, written.data(), update, offset, size)) return 1;
                memcpy(source.data() + offset, update, size);
                for (int b = 0; b < m.bytes; ++b) {
                    const int from = sj_kvarn_lowbits::strip_master_source_byte(m, b);
                    if (written[b] != (from < 0 ? 0 : source[from])) return 1;
                }
                if (!sj_kvarn_lowbits::strip_master_get(m, written.data(), restored.data(), offset, size) ||
                    memcmp(restored.data(), source.data() + offset, size)) return 1;
                if (!sj_kvarn_lowbits::strip_master_fill(m, written.data(), 0x7b, offset, size)) return 1;
                memset(source.data() + offset, 0x7b, size);
                for (int b = 0; b < m.bytes; ++b) {
                    const int from = sj_kvarn_lowbits::strip_master_source_byte(m, b);
                    if (written[b] != (from < 0 ? 0 : source[from])) return 1;
                }
            }
            if (!sj_kvarn_lowbits::strip_master_fill(m, written.data(), 0xff, 0, m.canonical_bytes)) return 1;
            for (int b = 0; b < m.bytes; ++b) {
                const int from = sj_kvarn_lowbits::strip_master_source_byte(m, b);
                if (written[b] != (from < 0 ? 0 : 0xff)) return 1;
            }
            for (int b = m.bytes; b < int(written.size()); ++b) if (written[b] != 0xcd) return 1;
            const auto before = written;
            for (int offset : {-1, m.canonical_bytes + 1}) {
                if (sj_kvarn_lowbits::strip_master_set(m, written.data(), update, offset, 1) ||
                    sj_kvarn_lowbits::strip_master_fill(m, written.data(), 0, offset, 1) ||
                    sj_kvarn_lowbits::strip_master_get(m, written.data(), restored.data(), offset, 1)) return 1;
            }
            for (int size : {-1, 2, 2147483647}) {
                if (sj_kvarn_lowbits::strip_master_set(m, written.data(), update, m.canonical_bytes - 1, size) ||
                    sj_kvarn_lowbits::strip_master_fill(m, written.data(), 0, m.canonical_bytes - 1, size) ||
                    sj_kvarn_lowbits::strip_master_get(m, written.data(), restored.data(), m.canonical_bytes - 1, size)) return 1;
            }
            if (sj_kvarn_lowbits::strip_master_set(m, nullptr, update, 0, 1) ||
                sj_kvarn_lowbits::strip_master_set(m, written.data(), nullptr, 0, 1) ||
                sj_kvarn_lowbits::strip_master_fill(m, nullptr, 0, 0, 1) ||
                sj_kvarn_lowbits::strip_master_get(m, nullptr, restored.data(), 0, 1) ||
                sj_kvarn_lowbits::strip_master_get(m, written.data(), nullptr, 0, 1)) return 1;
            if (written != before || !sj_kvarn_lowbits::strip_master_set(m, nullptr, nullptr, m.canonical_bytes, 0) ||
                !sj_kvarn_lowbits::strip_master_fill(m, nullptr, 0, m.canonical_bytes, 0) ||
                !sj_kvarn_lowbits::strip_master_get(m, nullptr, nullptr, m.canonical_bytes, 0)) return 1;
        }
        if (sj_kvarn_lowbits::strip_master_byte(m, -1) != -1 || sj_kvarn_lowbits::strip_master_byte(m, m.canonical_bytes) != -1 ||
            sj_kvarn_lowbits::strip_master_source_byte(m, -1) != -1 || sj_kvarn_lowbits::strip_master_source_byte(m, m.bytes) != -1) return 1;
    }
    sj_kvarn_lowbits::strip_master_layout bad;
    for (int b : {-1, 0, 1, 4, 31}) {
        for (int v = 0; v < 2; ++v) {
            if (!sj_kvarn_lowbits::make_strip_master_layout(3, 2, bad) ||
                sj_kvarn_lowbits::make_strip_master_layout(v ? 3 : b, v ? b : 2, bad)) return 1;
            if (bad.bits_k || bad.bits_v || bad.k_payload || bad.v_payload || bad.canonical_bytes ||
                bad.k_strip || bad.v_strip || bad.strip_bytes || bad.bytes) return 1;
            if (sj_kvarn_lowbits::strip_master_range(bad, 0, 0)) return 1;
        }
    }
    printf("strip master: %zu exact trellis indices, canonical intervals, history, metadata and bounds PASS\n", checked);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 2 && strcmp(argv[1], "--strip-master") == 0) return test_strip_master();
    if (argc == 2 && strcmp(argv[1], "--stage-strips") == 0) return test_stage_strips();
    if (argc < 4) {
        fprintf(stderr, "usage: %s tiles.bin records.bin n [G D bits_k bits_v iters]\n", argv[0]);
        return 2;
    }
    const int n      = atoi(argv[3]);
    const int G      = argc > 4 ? atoi(argv[4]) : 128;
    const int D      = argc > 5 ? atoi(argv[5]) : 256;
    const int bits_k = argc > 6 ? atoi(argv[6]) : 4;
    const int bits_v = argc > 7 ? atoi(argv[7]) : 4;
    const int iters  = argc > 8 ? atoi(argv[8]) : 16;

    const ggml_sj_kvarn::layout l = ggml_sj_kvarn::make_layout(D, G, bits_k, bits_v);
    if (l.bytes != ggml_sj_kvarn_rec_bytes(D, G, bits_k, bits_v)) {
        fprintf(stderr, "layout/rec_bytes disagree: %zu vs %zu\n", l.bytes, ggml_sj_kvarn_rec_bytes(D, G, bits_k, bits_v));
        return 1;
    }

    const std::vector<uint8_t> tiles = read_file(argv[1]);
    std::vector<uint8_t> recs        = read_file(argv[2]);
    const size_t tile_bytes = 2 * (size_t) G * D * sizeof(float);
    if (tiles.size() < (size_t) n * tile_bytes || recs.size() < (size_t) n * l.bytes) {
        fprintf(stderr, "files too small for n=%d (tiles %zu need %zu, records %zu need %zu)\n",
                n, tiles.size(), (size_t) n*tile_bytes, recs.size(), (size_t) n*l.bytes);
        return 2;
    }

    for (int g = 0; g < n; ++g) {
        const std::vector<uint8_t> r = oracle_to_layout(recs.data() + (size_t) g*l.bytes, l);
        memcpy(recs.data() + (size_t) g*l.bytes, r.data(), l.bytes);
    }

    int rc = 0;

    // ---- 1. scalar codec vs oracle, all n groups
    {
        diff_stats st;
        int n_skipped = 0;
        std::vector<ggml_fp16_t> kr, vr;
        std::vector<uint8_t> rec(l.bytes);
        for (int g = 0; g < n; ++g) {
            const float * K = (const float *) (tiles.data() + (size_t) g*tile_bytes);
            const float * V = K + (size_t) G*D;
            {
                // a sealer only ever sees finite fp16 rows; skip synthetic tiles carrying inf/nan
                bool finite = true;
                for (size_t i = 0; i < (size_t) 2*G*D && finite; ++i) finite = std::isfinite(K[i]);
                if (!finite) { printf("  group %d: non-finite input, skipped\n", g); ++n_skipped; continue; }
            }
            rotate_rows(K, G, D, kr);
            rotate_rows(V, G, D, vr);
            ggml_sj_kvarn::seal_group(kr.data(), vr.data(), D, l, iters, rec.data());
            diff_stats one;
            compare_records(rec.data(), recs.data() + (size_t) g*l.bytes, l, one);
            if (one.payload_diff_gt1 > 0 || one.meta_bytes_diff > 4) {
                printf("  group %d: %zu metadata bytes differ, %zu payload values differ, %zu by more than 1\n",
                       g, one.meta_bytes_diff, one.payload_diff, one.payload_diff_gt1);
            }
            st.rec_mismatch += one.rec_mismatch; st.meta_bytes_diff += one.meta_bytes_diff;
            st.payload_vals += one.payload_vals; st.payload_diff += one.payload_diff; st.payload_diff_gt1 += one.payload_diff_gt1;
            st.noise_rows += one.noise_rows; st.noise_diff += one.noise_diff;
        }
        const double frac = st.payload_vals ? (double) st.payload_diff / st.payload_vals : 0.0;
        printf("codec vs oracle: %d groups, %zu mismatching records, %zu metadata bytes differ, "
               "%zu/%zu payload values differ (%.2e), %zu differ by more than 1\n",
               n, st.rec_mismatch, st.meta_bytes_diff, st.payload_diff, st.payload_vals, frac, st.payload_diff_gt1);
        // Gate: the C codec (float32 libm) and the numpy oracle (float64 std chain) may land on different sides
        // of an fp16 rounding boundary for a handful of metadata halves (observed 17 of ~87K over 195 tiles) and
        // flip a code by one at an RTN tie; anything beyond that is a real codec divergence.
        const size_t meta_bytes = (size_t) (n - n_skipped) * (l.bytes - l.k_scale);
        const double meta_frac  = meta_bytes ? (double) st.meta_bytes_diff / meta_bytes : 0.0;
        printf("  metadata bytes differing: %.2e of %zu, groups skipped: %d, subnormal-step rows: %zu (%zu code diffs ignored)\n",
               meta_frac, meta_bytes, n_skipped, st.noise_rows, st.noise_diff);
        const bool ok = meta_frac < 1e-3 && st.payload_diff_gt1 == 0 && frac < 1e-4;
        printf("  %s\n", ok ? "OK" : "FAIL");
        if (!ok) rc = 1;
    }

    // ---- 2. ggml ops on the CPU backend: TURBO_WHT 256 (rotation) then SJKVARN_SEAL, first n_op groups
    {
        const int n_op = n < 8 ? n : 8;
        const int hkv  = 2;                      // fold pairs of groups into two heads
        const int n_groups = (n_op + hkv - 1) / hkv;
        const int n_tok = n_groups * G;

        ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead() + 4*(size_t) n_tok*hkv*D*sizeof(float) + (size_t) n_groups*hkv*l.bytes + 1024*1024, NULL, false };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * kf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) hkv*D, n_tok);
        ggml_tensor * vf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) hkv*D, n_tok);
        float * kfp = (float *) kf->data;
        float * vfp = (float *) vf->data;
        memset(kfp, 0, ggml_nbytes(kf));
        memset(vfp, 0, ggml_nbytes(vf));
        for (int r = 0; r < n_op; ++r) {
            const int g = r / hkv, h = r % hkv;
            const float * K = (const float *) (tiles.data() + (size_t) r*tile_bytes);
            const float * V = K + (size_t) G*D;
            for (int t = 0; t < G; ++t) {
                memcpy(kfp + ((size_t) (g*G + t)*hkv + h)*D, K + (size_t) t*D, D*sizeof(float));
                memcpy(vfp + ((size_t) (g*G + t)*hkv + h)*D, V + (size_t) t*D, D*sizeof(float));
            }
        }

        ggml_tensor * kw = ggml_turbo_wht(ctx, kf, 0, 256, NULL);
        ggml_tensor * vw = ggml_turbo_wht(ctx, vf, 0, 256, NULL);
        ggml_tensor * kh = ggml_cast(ctx, kw, GGML_TYPE_F16);
        ggml_tensor * vh = ggml_cast(ctx, vw, GGML_TYPE_F16);
        ggml_tensor * body = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) l.bytes*hkv*n_groups);
        memset(body->data, 0xAB, ggml_nbytes(body));
        ggml_tensor * seal = ggml_sj_kvarn_seal(ctx, body, kh, vh, D, G, bits_k, bits_v, iters);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, seal);
        ggml_graph_compute_with_ctx(ctx, gf, 8);

        // rotation check: op output vs scalar hadamard, tolerance a few ulp
        double max_rot_err = 0.0;
        {
            std::vector<float> row(D);
            const float * kwp = (const float *) kw->data;
            for (int t = 0; t < n_tok; ++t) {
                for (int h = 0; h < hkv; ++h) {
                    memcpy(row.data(), kfp + ((size_t) t*hkv + h)*D, D*sizeof(float));
                    ggml_sj_kvarn::hadamard(row.data(), D);
                    for (int d = 0; d < D; ++d) {
                        const double e = fabs((double) row[d] - kwp[((size_t) t*hkv + h)*D + d]);
                        if (e > max_rot_err) max_rot_err = e;
                    }
                }
            }
        }
        diff_stats st;
        for (int r = 0; r < n_op; ++r) {
            compare_records((const uint8_t *) body->data + (size_t) r*l.bytes, recs.data() + (size_t) r*l.bytes, l, st);
        }
        const double frac = st.payload_vals ? (double) st.payload_diff / st.payload_vals : 0.0;
        printf("ggml ops (WHT256 + SJKVARN_SEAL, CPU): %d records, max rotation abs err %.3e, %zu mismatching records, "
               "%zu metadata bytes differ, %zu/%zu payload values differ (%.2e), %zu by more than 1\n",
               n_op, max_rot_err, st.rec_mismatch, st.meta_bytes_diff, st.payload_diff, st.payload_vals, frac, st.payload_diff_gt1);
        // the op rotates in float32 with the same butterfly, then rounds to fp16: a rare 1-ulp fp16 flip is possible
        const bool ok = max_rot_err < 1e-5 && st.payload_diff_gt1 == 0 && frac < 1e-3 && st.meta_bytes_diff < 16;
        printf("  %s\n", ok ? "OK" : "FAIL");
        if (!ok) rc = 1;
        ggml_free(ctx);
    }

    return rc;
}
