#pragma once

// [#81] Compact resident draft head for the MTP draft vocabulary shortlist.
//
// The shortlist normally scores the selected rows of the output head as one-row experts (mul_mat_id). An EXL3
// head cannot do that: its bytes are trellis-coded 16x16 tiles, not rows, and the forward is
//   y = svh * H128( W_trellis . H128( suh * x ) )        (llama build_lora_mm, ggml-cuda exl3.cu)
// so one logit mixes the 128 trellis rows of its 128-row output block. This header builds, on the CPU and once,
// the n_sel shortlisted rows of the effective head in the ROTATED INPUT domain:
//   row_eff[j] = svh[j] * sum_r had[j % 128][r] * W_trellis[128 * (j / 128) + r]        (r = 0..127)
// so that  y[j] = row_eff[j] . x_rot  with  x_rot = H128(suh * x), the activation the EXL3 kernel consumes.
// The rows are then quantized to a compact type; the graph scores them with one plain mul_mat on x_rot.
// An AI+ head (ggml-ai4.h: AITk_U / AITk) has no one-row path and no to_float either: its rows come in 16-row
// trellis slabs. Its rows are decoded slab by slab in the ORIGINAL basis (ggml_ai4_dequantize_slab undoes the offline
// rotation), so x_rot = x as for a plain head.
// For any other head type (LLAMA_DRAFT_VOCAB_COMPACT=1 A/B) the rows are the head rows (to_float) and x_rot = x.
// Header-only so tests/test-draft-vocab-compact.cpp can check it without a model.

#include "ggml.h"
#include "ggml-ai4-decode.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

struct llama_draft_compact_src {
    const uint8_t * data = nullptr;   // host copy of the whole head tensor (ggml_nbytes bytes)
    ggml_type       type = GGML_TYPE_COUNT;
    int64_t         K    = 0;         // row length (n_embd)
    int64_t         N    = 0;         // rows (n_vocab)
    const float   * svh  = nullptr;   // EXL3 only: [N] output scales
    const float   * had  = nullptr;   // EXL3 only: [128*128] exl3_had128.weight as stored (ne0 fastest)
};

// [KAIFUSE R2] fractional AI+ head types (AIF25_U / AIF25_U2 / AIF175_U2 / AIF25_U2_512: the 8 GB file's head)
inline bool llama_draft_vocab_aif_head(ggml_type t) {
    return t == GGML_TYPE_AIF25_U || t == GGML_TYPE_AIF25_U2 || t == GGML_TYPE_AIF175_U2 || t == GGML_TYPE_AIF25_U2_512;
}

// [KAIFUSE R2] LLAMA_RM24H_AIHEAD: 1 = a fractional AI+ head takes the shortlist as one-row experts (the CUDA
// row-group kernel, ggml_cuda_mul_mat_id_aif_rows); 2 = reference: score the full head, then gather (bit-identical
// shortlist logits, for the equivalence test); 0 (default) = the head is not eligible, as before (head_layout), so the
// draft scores the full head. Opt-in until its speed gate is settled (the first 8dense gate is under audit).
inline int llama_draft_vocab_aihead_mode() {
    static const int v = [] {
        const char * env = std::getenv("LLAMA_RM24H_AIHEAD");
        return env ? std::atoi(env) : 0;
    }();
    return v;
}

// Score the existing AI+ head, then gather shortlist logits. This trades extra
// projection work for avoiding a second resident weight matrix; opt-in only.
inline bool llama_draft_vocab_full_head(const ggml_tensor * head) {
    if (head && llama_draft_vocab_aif_head(head->type)) {
        return llama_draft_vocab_aihead_mode() == 2;
    }
    const char * env = std::getenv("LLAMA_DRAFT_VOCAB_FULL_HEAD");
    return env && std::atoi(env) == 1 && head &&
        (ggml_ai4_rate(head->type) != 0 || ggml_ait_rate(head->type) != 0);
}

inline ggml_tensor * llama_draft_vocab_gather_logits(
        ggml_context * ctx, ggml_tensor * logits, ggml_tensor * ids) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32 && logits->ne[2] == 1 && logits->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_vector(ids));
    // GET_ROWS selects ne[1], so transpose vocabulary into that dimension.
    ggml_tensor * rows = ggml_cont(ctx, ggml_transpose(ctx, logits));
    return ggml_cont(ctx, ggml_transpose(ctx, ggml_get_rows(ctx, rows, ids)));
}

// row types the compact head may use: no imatrix needed, a CUDA mul_mat path at widths 1-8
inline bool llama_draft_compact_type_ok(ggml_type t, int64_t K) {
    switch (t) {
        case GGML_TYPE_F32: case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q6_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ4_NL:
            return K % ggml_blck_size(t) == 0;
        default:
            return false;
    }
}

// Writes the n_sel compact rows (row i = effective row of token ids[i]) into dst, ggml_row_size(qtype, K) bytes
// each, or as f32 when qtype == GGML_TYPE_F32. Returns false (dst untouched or partial) on an unsupported input.
inline bool llama_draft_compact_build(
        const llama_draft_compact_src & src,
        const int32_t * ids, int64_t n_sel,
        ggml_type qtype, void * dst, int n_threads) {
    const int64_t K = src.K, N = src.N;
    const int bits = ggml_exl3_bits(src.type);
    if (src.data == nullptr || ids == nullptr || n_sel <= 0 || K <= 0 || N <= 0 || !llama_draft_compact_type_ok(qtype, K)) {
        return false;
    }
    if (bits != 0 && (src.svh == nullptr || src.had == nullptr || K % 16 != 0 || N % 128 != 0)) {
        return false;
    }
    const bool is_ai = ggml_ai4_rate(src.type) != 0 || ggml_ait_rate(src.type) != 0; // AIT: same slab decoder
    if (is_ai && (N % 16 != 0 || ggml_ai4_slab_nbytes(src.type, K) == 0)) {
        return false;
    }
    const ggml_type_traits * tt = ggml_get_type_traits(src.type);
    if (bits == 0 && !is_ai && (tt == nullptr || tt->to_float == nullptr)) {
        return false;
    }
    for (int64_t i = 0; i < n_sel; ++i) {
        if (ids[i] < 0 || ids[i] >= N) {
            return false;
        }
    }
    if (qtype != GGML_TYPE_F32) {
        ggml_quantize_init(qtype); // before the workers: ggml_quantize_chunk calls it again as a no-op
    }
    const size_t out_row = ggml_row_size(qtype, K);

    auto emit = [&](int64_t pos, const float * row) {
        uint8_t * d = (uint8_t *) dst + (size_t) pos * out_row;
        if (qtype == GGML_TYPE_F32) {
            memcpy(d, row, (size_t) K * sizeof(float));
        } else {
            ggml_quantize_chunk(qtype, row, d, 0, 1, K, nullptr);
        }
    };

    n_threads = std::max(1, n_threads);

    if (is_ai) {
        // AI+: group the selected positions by 16-row slab, decode each slab that holds one once
        const int64_t n_slab = N / 16;
        std::vector<int64_t> slab_start(n_slab + 1, 0);
        for (int64_t i = 0; i < n_sel; ++i) {
            slab_start[ids[i] / 16 + 1]++;
        }
        for (int64_t b = 0; b < n_slab; ++b) {
            slab_start[b + 1] += slab_start[b];
        }
        std::vector<int64_t> order(n_sel);
        {
            std::vector<int64_t> fill(slab_start.begin(), slab_start.end() - 1);
            for (int64_t i = 0; i < n_sel; ++i) {
                order[fill[ids[i] / 16]++] = i;
            }
        }
        std::atomic<int64_t> next{0};
        std::atomic<bool>    ok{true};
        auto worker = [&]() {
            std::vector<float> slab((size_t) 16 * K);
            for (int64_t b; (b = next.fetch_add(1)) < n_slab; ) {
                const int64_t s0 = slab_start[b], s1 = slab_start[b + 1];
                if (s0 == s1) {
                    continue;
                }
                if (!ggml_ai4_dequantize_slab(src.type, src.data, K, b, slab.data())) {
                    ok = false;
                    return;
                }
                for (int64_t q = s0; q < s1; ++q) {
                    const int64_t pos = order[q];
                    emit(pos, slab.data() + (size_t) (ids[pos] % 16) * K);
                }
            }
        };
        std::vector<std::thread> th;
        for (int t = 1; t < n_threads; ++t) th.emplace_back(worker);
        worker();
        for (auto & t : th) t.join();
        return ok;
    }

    if (bits == 0) {
        // plain row-addressable head: rows are the head rows
        const size_t in_row = ggml_row_size(src.type, K);
        std::atomic<int64_t> next{0};
        auto worker = [&]() {
            std::vector<float> row(K);
            for (int64_t pos; (pos = next.fetch_add(64)) < n_sel; ) {
                for (int64_t p = pos; p < std::min(n_sel, pos + 64); ++p) {
                    tt->to_float(src.data + (size_t) ids[p] * in_row, row.data(), K);
                    emit(p, row.data());
                }
            }
        };
        std::vector<std::thread> th;
        for (int t = 1; t < n_threads; ++t) th.emplace_back(worker);
        worker();
        for (auto & t : th) t.join();
        return true;
    }

    // EXL3: group the selected positions by 128-row output block
    const int64_t n_blk = N / 128;
    std::vector<int64_t> blk_start(n_blk + 1, 0);
    for (int64_t i = 0; i < n_sel; ++i) {
        blk_start[ids[i] / 128 + 1]++;
    }
    for (int64_t b = 0; b < n_blk; ++b) {
        blk_start[b + 1] += blk_start[b];
    }
    std::vector<int64_t> order(n_sel);
    {
        std::vector<int64_t> fill(blk_start.begin(), blk_start.end() - 1);
        for (int64_t i = 0; i < n_sel; ++i) {
            order[fill[ids[i] / 128]++] = i;
        }
    }

    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        std::vector<float> blk((size_t) 128 * K);   // the 128 trellis rows of one output block, row-major
        std::vector<float> acc;
        for (int64_t b; (b = next.fetch_add(1)) < n_blk; ) {
            const int64_t s0 = blk_start[b], s1 = blk_start[b + 1];
            if (s0 == s1) {
                continue;
            }
            for (int64_t g = 0; g < 8; ++g) {
                // group 8b+g = trellis rows 128b + 16g .. +15, written as y[col*K + k]
                ggml_exl3_dequantize_row_group(src.data, K, N, bits, 8*b + g, blk.data() + (size_t) 16*g*K);
            }
            const int64_t m = s1 - s0;
            acc.assign((size_t) m * K, 0.0f);
            // acc[i] = sum_r had[i128][r] * blk[r], k-chunked so the block chunk stays in L2
            const int64_t KC = 256;
            for (int64_t k0 = 0; k0 < K; k0 += KC) {
                const int64_t kn = std::min(KC, K - k0);
                for (int64_t q = 0; q < m; ++q) {
                    const int64_t j = ids[order[s0 + q]];
                    const float * h = src.had + (size_t) (j % 128) * 128;
                    float * a = acc.data() + (size_t) q * K + k0;
                    for (int r = 0; r < 128; ++r) {
                        const float hr = h[r];
                        const float * w = blk.data() + (size_t) r * K + k0;
                        for (int64_t k = 0; k < kn; ++k) {
                            a[k] += hr * w[k];
                        }
                    }
                }
            }
            for (int64_t q = 0; q < m; ++q) {
                const int64_t pos = order[s0 + q];
                const float   sv  = src.svh[ids[pos]];
                float * a = acc.data() + (size_t) q * K;
                for (int64_t k = 0; k < K; ++k) {
                    a[k] *= sv;
                }
                emit(pos, a);
            }
        }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < n_threads; ++t) th.emplace_back(worker);
    worker();
    for (auto & t : th) t.join();
    return true;
}
