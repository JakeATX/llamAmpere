// Always-resident expert bank: offload pattern anchoring, regular-call skip of bank rows, resident census.
#include "common.h"
#include "fit.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <set>
#include <utility>
#include <vector>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

static void test_patterns() {
    const std::string per_layer = "blk\\.7\\.ffn_(up|down|gate_up|gate)_(ch|)exps\\."; // fit.cpp LAYER_FRACTION_MOE form
    const std::regex pats[] = { std::regex(LLM_FFN_EXPS_REGEX), std::regex(common_moe_cache_tensor_override_pattern()),
                                std::regex(llm_ffn_block_regex(7, LLM_FFN_EXPS_REGEX)), std::regex(per_layer) };
    for (const auto & re : pats) {
        for (const char * name : { "blk.7.ffn_up_exps.weight", "blk.7.ffn_gate_exps.weight", "blk.7.ffn_down_exps.weight",
                                   "blk.7.ffn_gate_up_exps.weight", "blk.7.ffn_down_chexps.weight", "blk.7.ffn_up_exps.scale" }) {
            CHECK(std::regex_search(name, re));
        }
        for (const char * name : { "blk.7.ffn_up_exps_hot.weight", "blk.7.ffn_gate_exps_hot.weight",
                                   "blk.7.ffn_down_exps_hot.weight", "blk.7.ffn_exps_hot_map.weight", "blk.7.ffn_up.weight" }) {
            CHECK(!std::regex_search(name, re));
        }
    }
}

// one CPU MUL_MAT_ID over F32 experts; returns dst [n_out, n_used, n_tok]
static std::vector<float> run_mmid(bool register_bank, const std::vector<int32_t> & ids_v, const std::vector<uint8_t> & bank) {
    const int n_in = 32, n_out = 16, n_exp = 8, n_used = 2, n_tok = 3;
    ggml_init_params ip = { 16u << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, n_out, n_exp);
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, n_used, n_tok);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    for (int64_t i = 0; i < ggml_nelements(w); ++i) { ((float *) w->data)[i] = 0.01f * (float) ((i * 37) % 101) - 0.5f; }
    for (int64_t i = 0; i < ggml_nelements(x); ++i) { ((float *) x->data)[i] = 0.02f * (float) ((i * 13) % 53) - 0.5f; }
    memcpy(ids->data, ids_v.data(), ids_v.size() * sizeof(int32_t));
    ggml_tensor * out = ggml_mul_mat_id(ctx, w, x, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (register_bank) {
        ggml_moe_bank_register(w->data, 0, n_exp, bank.data());
    }
    ggml_graph_compute_with_ctx(ctx, gf, 2);
    std::vector<float> r((const float *) out->data, (const float *) out->data + ggml_nelements(out));
    if (register_bank) {
        ggml_moe_bank_unregister(w->data);
    }
    ggml_free(ctx);
    return r;
}

static void test_skip(bool skip_on) {
    const std::vector<int32_t> ids = { 1, 2,  4, 0,  3, 1 };  // tokens x n_used
    std::vector<uint8_t> bank(8, 0);
    bank[1] = bank[4] = 1;
    const auto ref = run_mmid(false, ids, bank);
    const auto got = run_mmid(true, ids, bank);
    CHECK(ref.size() == got.size());
    const int n_out = 16;
    for (size_t row = 0; row < ids.size(); ++row) {
        const bool zero = skip_on && bank[ids[row]];
        for (int i = 0; i < n_out; ++i) {
            const float g = got[row * n_out + i], r = ref[row * n_out + i];
            CHECK(zero ? g == 0.0f : memcmp(&g, &r, sizeof(float)) == 0);
        }
        bool nonzero = false;
        for (int i = 0; i < n_out; ++i) { nonzero = nonzero || ref[row * n_out + i] != 0.0f; }
        CHECK(nonzero);
    }
    CHECK((ggml_moe_bank_lookup(ref.data()) == nullptr));
}

static std::set<std::pair<const void *, int32_t>> g_resident;
static bool resident_fn(void *, const void * data, int32_t e) { return g_resident.count({ data, e }) > 0; }

static void test_census() {
    static char t[2][3][8];  // 2 layers x up/gate/down, fake data pointers
    uint8_t bank1[8] = { 0, 1, 1, 0, 0, 0, 0, 0 };
    for (int p = 0; p < 3; ++p) {
        ggml_moe_bank_register(t[0][p], 0, 8, nullptr); // layer 0: no bank
        ggml_moe_bank_register(t[1][p], 1, 8, bank1);   // layer 1: experts 1, 2 in bank
    }
    CHECK(ggml_moe_bank_lookup(t[0][0]) == nullptr);
    CHECK(ggml_moe_bank_lookup(t[1][2]) != nullptr);
    for (int p = 0; p < 3; ++p) {
        g_resident.insert({ t[0][p], 0 });              // layer 0 expert 0 complete
        g_resident.insert({ t[1][p], 2 });              // layer 1 expert 2 complete, but it is a bank expert
        g_resident.insert({ t[1][p], 5 });              // layer 1 expert 5 complete
    }
    g_resident.insert({ t[0][0], 3 });                  // layer 0 expert 3: up only, not complete
    g_resident.insert({ t[0][1], 3 });
    int32_t lo = 0, hi = 0;
    size_t cache = 0, bank = 0, both = 0;
    CHECK(ggml_moe_bank_census(resident_fn, nullptr, &lo, &hi, &cache, &bank, &both));
    CHECK(lo == 0 && hi == 1);
    CHECK(cache == 2);  // L0 e0, L1 e5
    CHECK(bank  == 2);  // L1 e1, e2
    CHECK(both  == 1);  // L1 e2 also in the cache
    for (int l = 0; l < 2; ++l) { for (int p = 0; p < 3; ++p) { ggml_moe_bank_unregister(t[l][p]); } }
    CHECK(!ggml_moe_bank_census(resident_fn, nullptr, &lo, &hi, &cache, &bank, &both));
}

// The card-side cache points bank expert entries at an all-zero block: every quant type that has a row decoder
// must turn all-zero bytes into exactly zero weights (finite, so weight 0 times the result stays 0).
static void test_zero_block() {
    int checked = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const ggml_type type = (ggml_type) t;
        const auto * traits = ggml_get_type_traits(type);
        if (!traits || !traits->to_float || ggml_blck_size(type) <= 0 || traits->type_size == 0) { continue; }
        const int64_t k = 256 * (int64_t) ggml_blck_size(type);
        std::vector<uint8_t> zero(ggml_row_size(type, k), 0);
        std::vector<float> out(k, 1.0f);
        traits->to_float(zero.data(), out.data(), k);
        bool ok = true;
        for (float v : out) { ok = ok && v == 0.0f; }
        if (!ok) { fprintf(stderr, "FAIL zero block decodes nonzero: %s\n", ggml_type_name(type)); failures++; }
        ++checked;
    }
    CHECK(checked > 10);
}

static void test_small_expert_reduce() {
    for (int nt : { 1, 2, 3, 4, 5, 8 }) {
        for (int used : { 2, 7, 10 }) {
            ggml_init_params ip = { 16u << 20, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            const int rows = 2560;
            ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rows, used, nt);
            float * data = (float *) x->data;
            for (int t = 0; t < nt; ++t) {
                for (int e = 0; e < used; ++e) {
                    for (int r = 0; r < rows; ++r) {
                        // Distinct tokens/rows expose a wrong permute; masked columns model bank holes.
                        data[(t*used + e)*rows + r] = (t + e) % 3 == 0 ? 0.0f :
                            ((r*13 + e*17 + t*29) % 101 - 50)*0.013f;
                    }
                }
            }
            ggml_tensor * out = ggml_reshape_2d(ctx,
                ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3))), rows, nt);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            CHECK(ggml_graph_compute_with_ctx(ctx, gf, 2) == GGML_STATUS_SUCCESS);
            for (int t = 0; t < nt; ++t) {
                for (int r = 0; r < rows; ++r) {
                    double expected = 0.0, scale = 0.0;
                    for (int e = 0; e < used; ++e) {
                        const float v = data[(t*used + e)*rows + r];
                        expected += v;
                        scale += std::abs(v);
                    }
                    const float got = ((float *) out->data)[t*rows + r];
                    CHECK(std::isfinite(got) && std::abs(got - expected) <= 2e-6*(1.0 + scale));
                }
            }
            ggml_free(ctx);
        }
    }
}

static void test_mixed_ffn_shape(int n_in, int n_hidden, bool iq4_down = false) {
    for (ggml_type bank_type : { GGML_TYPE_AIT2_U, GGML_TYPE_AIT3_U }) {
        for (int width : { 1, 4, 5, 8 }) {
            for (int mode : { 0, 1, 2 }) {
                ggml_context * ctx = ggml_init({ 64u << 20, nullptr, false });
                CHECK(ctx != nullptr);
                if (!ctx) { return; }
                ggml_tensor * hw[3], * cw[3];
                for (int p = 0; p < 3; ++p) {
                    const int cols = p == 2 ? n_hidden : n_in;
                    const int rows = p == 2 ? n_in : n_hidden;
                    const ggml_type hot_type = iq4_down && p == 2 ? GGML_TYPE_IQ4_NL : bank_type;
                    const ggml_type cold_type = p == 2 ?
                            (iq4_down ? GGML_TYPE_IQ4_NL : n_hidden == 640 ? GGML_TYPE_Q2_0 : GGML_TYPE_Q4_0) : GGML_TYPE_Q4_0;
                    hw[p] = ggml_new_tensor_3d(ctx, hot_type, cols, rows, 4);
                    cw[p] = ggml_new_tensor_3d(ctx, cold_type, cols, rows, 4);
                    std::vector<float> values(ggml_nelements(cw[p]));
                    if (hot_type == GGML_TYPE_IQ4_NL) {
                        for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.001f*((int)((i*29 + p*19)%103) - 51); }
                        ggml_quantize_chunk(hot_type, values.data(), hw[p]->data, 0, rows*4, cols, nullptr);
                    } else {
                        const int unit = bank_type == GGML_TYPE_AIT2_U ? 544 : 800;
                        auto * bytes = (uint8_t *) hw[p]->data;
                        for (size_t i = 0; i < ggml_nbytes(hw[p]); ++i) { bytes[i] = (uint8_t) ((i*73 + i/7 + p*31)%256); }
                        for (size_t i = 0; i < ggml_nbytes(hw[p]); i += unit) {
                            auto * scales = (ggml_fp16_t *) (bytes + i);
                            for (int r = 0; r < 16; ++r) { scales[r] = ggml_fp32_to_fp16(0.0005f*(1 + (r + p)%5)); }
                        }
                    }
                    for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.001f*((int)((i*37 + p*13)%101) - 50); }
                    ggml_quantize_chunk(cold_type, values.data(), cw[p]->data, 0, rows*4, cols, nullptr);
                }
                ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, 1, width);
                ggml_tensor * hi = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, width);
                ggml_tensor * ci = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, width);
                ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 4, width);
                for (int i = 0; i < width*n_in; ++i) { ((float *) x->data)[i] = 0.01f*((i*17)%61 - 30); }
                for (int j = 0; j < width*4; ++j) {
                    const bool bank = mode == 1 || (mode == 2 && j%3 != 0);
                    ((int32_t *) hi->data)[j] = bank ? (j*3 + j/4)%4 : -1;
                    ((int32_t *) ci->data)[j] = (j + j/4)%4;
                    ((float *) weights->data)[j] = 0.125f + 0.03125f*(j%7);
                }
                ggml_tensor * cold = ggml_reshape_3d(ctx,
                        ggml_step(ctx, ggml_scale_bias(ctx, ggml_cast(ctx, hi, GGML_TYPE_F32), -1.0f, -0.5f)), 1, 4, width);
                auto reduce = [&](ggml_tensor * e, ggml_tensor * w) {
                    return ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, ggml_mul(ctx, e, w), 1, 0, 2, 3)));
                };
                auto branch = [&](ggml_tensor ** w, ggml_tensor * ids) {
                    ggml_tensor * up = ggml_mul_mat_id(ctx, w[0], x, ids);
                    ggml_tensor * gate = ggml_mul_mat_id(ctx, w[1], x, ids);
                    return ggml_mul_mat_id(ctx, w[2], ggml_swiglu_split(ctx, gate, up), ids);
                };
                ggml_tensor * wh = ggml_sub(ctx, weights, ggml_mul(ctx, weights, cold));
                ggml_tensor * old = ggml_add(ctx, reduce(branch(hw, hi), wh), reduce(branch(cw, ci), ggml_mul(ctx, weights, cold)));
                auto mixed_mm = [&](int p, ggml_tensor * in) {
                    return ggml_add(ctx, ggml_mul_mat_id(ctx, hw[p], in, hi),
                            ggml_mul(ctx, ggml_mul_mat_id(ctx, cw[p], in, ci), cold));
                };
                ggml_tensor * up = mixed_mm(0, x), * gate = mixed_mm(1, x);
                ggml_tensor * mixed = reduce(mixed_mm(2, ggml_swiglu_split(ctx, gate, up)), weights);
                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, old);
                ggml_build_forward_expand(gf, mixed);
                CHECK(ggml_graph_compute_with_ctx(ctx, gf, 2) == GGML_STATUS_SUCCESS);
                for (int i = 0; i < width*n_in; ++i) {
                    const float a = ((const float *) old->data)[i], b = ((const float *) mixed->data)[i];
                    CHECK(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= 1e-5f*(1 + std::abs(a)));
                }
                ggml_free(ctx);
            }
        }
    }
}

static void test_mixed_ffn() {
    test_mixed_ffn_shape(256, 256);
    test_mixed_ffn_shape(2560, 640);
    test_mixed_ffn_shape(2560, 640, true);
    printf("mixed FFN: 72 two-call reference comparisons\n");
}

int main(int argc, char ** argv) {
    const bool child_off = argc > 1 && strcmp(argv[1], "--skip-off") == 0;
    if (child_off) {
        test_skip(false);
        return failures ? 1 : 0;
    }
    test_patterns();
    test_skip(true);
    test_census();
    test_zero_block();
    test_small_expert_reduce();
    const int64_t margin = 300 * 1024 * 1024;
    CHECK(common_fit_verify_compact_safe(true, { margin }, { margin }));
    CHECK(common_fit_verify_compact_safe(true, { margin, 2 * margin }, { margin, margin }));
    CHECK(!common_fit_verify_compact_safe(true, { margin - 1 }, { margin }));
    CHECK(!common_fit_verify_compact_safe(true, { 2 * margin, margin - 1 }, { margin, margin }));
    CHECK(!common_fit_verify_compact_safe(true, { -1 }, { margin }));
    CHECK(!common_fit_verify_compact_safe(true, { 2 * margin }, { margin - 1 }));
    CHECK(!common_fit_verify_compact_safe(true, {}, {}));
    CHECK(!common_fit_verify_compact_safe(true, { margin }, { margin, margin }));
    CHECK(!common_fit_verify_compact_safe(false, { 2 * margin }, { margin }));
    test_mixed_ffn();
    // GGML_MOE_BANK_SKIP=0 is read once per process: check it in a child
    char cmd[4200];
    snprintf(cmd, sizeof(cmd), "GGML_MOE_BANK_SKIP=0 '%s' --skip-off", argv[0]);
    CHECK(system(cmd) == 0);
    printf("%s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
