// [D1] CPU unit test for the opt-in exact head rescore (src/llama-d1.*), design 8.1. No model, no GPU.
// usage: test-d1-rescore [work_dir]   (writes two small sidecar GGUFs there; default: current directory)

#include "../src/llama-d1.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; std::fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } } while (0)

static constexpr int64_t E = 64;    // n_embd
static constexpr int64_t V = 1003;  // n_vocab (not a multiple of 8)

// reference top-n: partial_sort by (value desc, id asc)
static std::vector<int32_t> ref_top(const float * row, int64_t n_vocab, int n) {
    std::vector<int32_t> id(n_vocab);
    for (int64_t i = 0; i < n_vocab; ++i) { id[i] = (int32_t) i; }
    n = (int) std::min<int64_t>(n, n_vocab);
    std::partial_sort(id.begin(), id.begin() + n, id.end(), [&](int32_t a, int32_t b) {
        return row[a] > row[b] || (row[a] == row[b] && a < b);
    });
    id.resize(n);
    std::sort(id.begin(), id.end());
    return id;
}

static void test_select() {
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    for (int64_t nv : { (int64_t) 1003, (int64_t) 248320 }) {
        for (int n : { 1, 7, 256, 1000 }) {
            for (int rep = 0; rep < 3; ++rep) {
                std::vector<float> row(nv);
                for (auto & x : row) {
                    x = nd(rng);
                    if (rep == 2) { x = std::round(x * 4.0f) / 4.0f; } // many ties
                }
                std::vector<int32_t> got;
                llama_d1_select_top(row.data(), nv, n, got);
                std::sort(got.begin(), got.end());
                const auto want = ref_top(row.data(), nv, n);
                CHECK(got == want, "select nv=%lld n=%d rep=%d", (long long) nv, n, rep);
            }
        }
    }
    std::vector<int32_t> ids;
    std::vector<float> small = { 1.0f, 2.0f, 3.0f };
    llama_d1_select_top(small.data(), 3, 5, ids);
    CHECK(ids.size() == 3, "n > n_vocab returns every id");
    llama_d1_select_top(small.data(), 3, 0, ids);
    CHECK(ids.empty(), "n = 0 returns nothing");
}

struct sidecar {
    std::string path;
    ggml_type   type;
    std::vector<float> w;   // [V][E] the exact rows as stored (after f16 / q8_0 round trip)
    llama_d1_src src;
};

// write a sidecar; tamper: 0 none, 1 wrong rows_type KV, 2 wrong self fnv, 3 wrong n_vocab
static sidecar make_sidecar(const std::string & dir, const char * name, ggml_type type, uint64_t src_fnv, int tamper) {
    sidecar sc;
    sc.path = dir + "/" + name;
    sc.type = type;
    const int64_t nv = tamper == 3 ? V + 1 : V;
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> wf(nv * E);
    for (auto & x : wf) { x = nd(rng); }

    ggml_init_params ip = { /*.mem_size =*/ ggml_tensor_overhead() * 4 + (size_t) nv * E * 4 + 4096, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, E, nv);
    ggml_set_name(t, "output_exact.weight");
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(wf.data(), (ggml_fp16_t *) t->data, nv * E);
    } else {
        ggml_quantize_chunk(type, wf.data(), t->data, 0, nv, E, nullptr);
    }
    sc.w.resize(nv * E);
    ggml_get_type_traits(type)->to_float(t->data, sc.w.data(), nv * E);

    const size_t nbytes = ggml_nbytes(t);
    uint64_t self = llama_d1_fnv64_ends((const uint8_t *) t->data, nbytes);
    if (tamper == 2) { self ^= 1; }

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "d1.rows_type", tamper == 1 ? (type == GGML_TYPE_F16 ? "q8_0" : "f16") : (type == GGML_TYPE_F16 ? "f16" : "q8_0"));
    gguf_set_val_u64(g, "d1.source_bytes", 4096);
    gguf_set_val_u64(g, "d1.source_fnv64", src_fnv);
    gguf_set_val_u64(g, "d1.self_bytes", nbytes);
    gguf_set_val_u64(g, "d1.self_fnv64", self);
    gguf_add_tensor(g, t);
    CHECK(gguf_write_to_file(g, sc.path.c_str(), false), "write %s", sc.path.c_str());
    gguf_free(g);
    ggml_free(ctx);

    sc.src.valid  = true;
    sc.src.type   = GGML_TYPE_AIT3;
    sc.src.nbytes = 4096;
    sc.src.fnv    = src_fnv;
    return sc;
}

static double ref_dot(const sidecar & sc, const float * h, int64_t id, double * abs_sum) {
    double s = 0, a = 0;
    for (int64_t k = 0; k < E; ++k) {
        s += (double) sc.w[id * E + k] * h[k];
        a += std::fabs((double) sc.w[id * E + k] * h[k]);
    }
    *abs_sum = a;
    return s;
}

static void test_head_and_scope(const std::string & dir) {
    const uint64_t SRC = 0x1234567890abcdefULL;
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        const char * tn = type == GGML_TYPE_F16 ? "f16" : "q8_0";
        sidecar sc = make_sidecar(dir, type == GGML_TYPE_F16 ? "d1-test-f16.gguf" : "d1-test-q8.gguf", type, SRC, 0);
        std::string err;
        auto head = llama_d1_head_open(sc.path.c_str(), sc.src, E, V, err);
        CHECK(head != nullptr, "%s open: %s", tn, err.c_str());
        if (!head) { continue; }
        CHECK(head->type == type, "%s type", tn);

        // dot vs f64 reference
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> h(E);
        for (auto & x : h) { x = nd(rng); }
        double worst = 0;
        for (int64_t id = 0; id < V; ++id) {
            double as = 0;
            const double want = ref_dot(sc, h.data(), id, &as);
            worst = std::max(worst, std::fabs(head->dot(h.data(), id) - want) / std::max(as, 1e-12));
        }
        CHECK(worst <= 1e-4, "%s dot rel err %g", tn, worst);

        // refusals
        { llama_d1_src bad = sc.src; bad.fnv ^= 1;
          CHECK(!llama_d1_head_open(sc.path.c_str(), bad, E, V, err), "%s src fnv mismatch refused", tn); }
        { llama_d1_src bad = sc.src; bad.nbytes += 1;
          CHECK(!llama_d1_head_open(sc.path.c_str(), bad, E, V, err), "%s src bytes mismatch refused", tn); }
        { llama_d1_src bad = {};
          CHECK(!llama_d1_head_open(sc.path.c_str(), bad, E, V, err), "%s no src refused", tn); }
        CHECK(!llama_d1_head_open(sc.path.c_str(), sc.src, E + 32, V, err), "%s n_embd mismatch refused", tn);
        for (int tamper = 1; tamper <= 3; ++tamper) {
            std::string nm = std::string("d1-test-bad") + std::to_string(tamper) + "-" + tn + ".gguf";
            sidecar bad = make_sidecar(dir, nm.c_str(), type, SRC, tamper);
            CHECK(!llama_d1_head_open(bad.path.c_str(), bad.src, E, V, err), "%s tamper %d refused", tn, tamper);
            std::remove(bad.path.c_str());
        }

        // scope (design 3.1)
        llama_d1_scope_in in;
        in.head_path = sc.path.c_str();
        in.out_type = GGML_TYPE_AIT2_U2_512; in.out_ne0 = E; in.out_ne1 = V; in.n_embd = E; in.n_vocab = V;
        std::shared_ptr<llama_d1_head> got; std::string msg;
        {
            llama_d1_model_slot slot; slot.src = sc.src;
            llama_d1_scope_in x = in; x.head_path = nullptr;
            CHECK(llama_d1_scope(x, slot, got, msg) == LLAMA_D1_SCOPE_UNSET, "unset");
            x = in; x.ctx_default = false;
            CHECK(llama_d1_scope(x, slot, got, msg) == LLAMA_D1_SCOPE_OFF && !got, "MTP context off");
            x = in; x.embeddings = true;
            CHECK(llama_d1_scope(x, slot, got, msg) == LLAMA_D1_SCOPE_OFF, "embeddings off");
            x = in; x.mtp_chain = true;
            CHECK(llama_d1_scope(x, slot, got, msg) == LLAMA_D1_SCOPE_OFF, "mtp_chain off");
            // fit probe then target
            x = in; x.no_alloc = true;
            CHECK(llama_d1_scope(x, slot, got, msg) == LLAMA_D1_SCOPE_OFF && !slot.tried, "fit probe off, sidecar not opened");
            CHECK(llama_d1_scope(in, slot, got, msg) == LLAMA_D1_SCOPE_ON && got, "target after fit probe on: %s", msg.c_str());
            // two contexts on one model share one handle
            std::shared_ptr<llama_d1_head> got2;
            CHECK(llama_d1_scope(in, slot, got2, msg) == LLAMA_D1_SCOPE_ON && got2 == got, "second context on, same handle");
            // the sidecar's own base head (overlay missing): WARN
            x = in; x.out_type = GGML_TYPE_AIT3;
            CHECK(llama_d1_scope(x, slot, got, msg) == LLAMA_D1_SCOPE_WARN, "base head without overlay warns");
        }
        {
            // a different model (classic -md draft): off, no error, no warning
            llama_d1_model_slot slot; slot.src = sc.src; slot.src.fnv ^= 0xff;
            llama_d1_scope_in x = in; x.out_type = GGML_TYPE_AIT3;
            CHECK(llama_d1_scope(x, slot, got, msg) == LLAMA_D1_SCOPE_OFF, "other model head off");
            x = in; x.out_type = GGML_TYPE_Q6_K;
            CHECK(llama_d1_scope(x, slot, got, msg) == LLAMA_D1_SCOPE_OFF, "q6_k head off");
            // type 128 with a wrong source: context creation fails, and stays failed
            CHECK(llama_d1_scope(in, slot, got, msg) == LLAMA_D1_SCOPE_ERROR && !got, "type 128 + bad src = error");
            CHECK(llama_d1_scope(in, slot, got, msg) == LLAMA_D1_SCOPE_ERROR, "error is sticky");
        }
    }
}

// rows of logits + h; rescore expectations vs the reference
static void test_ctx(const std::string & dir) {
    sidecar sc = make_sidecar(dir, "d1-test-f16.gguf", GGML_TYPE_F16, 42, 0);
    std::string err;
    auto head = llama_d1_head_open(sc.path.c_str(), sc.src, E, V, err);
    CHECK(head != nullptr, "open: %s", err.c_str());
    if (!head) { return; }
    const int NTOP = 8;
    const int R = 6;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> L0(R * V), H0(R * E);
    for (auto & x : L0) { x = nd(rng); }
    for (auto & x : H0) { x = nd(rng); }

    auto expect_row = [&](const std::vector<float> & L, const float * Lorig, const float * h, int64_t r, bool rescored, const char * what) {
        const auto top = ref_top(Lorig, V, NTOP);
        int bad = 0;
        for (int64_t i = 0; i < V; ++i) {
            const bool in_top = std::binary_search(top.begin(), top.end(), (int32_t) i);
            if (rescored && in_top) {
                double as = 0;
                const double want = ref_dot(sc, h, i, &as);
                if (std::fabs(L[r * V + i] - want) > 1e-4 * std::max(as, 1e-12)) { ++bad; }
            } else if (std::memcmp(&L[r * V + i], &Lorig[i], sizeof(float)) != 0) {
                ++bad;
            }
        }
        CHECK(bad == 0, "%s row %lld: %d bad entries", what, (long long) r, bad);
    };

    // lazy (n_out > eager_max), one thread
    {
        llama_d1_ctx c(head, NTOP, /*eager*/ 4, /*threads*/ 1);
        c.reset(R);
        std::memcpy(c.h_row(0), H0.data(), sizeof(float) * R * E);
        c.mark_copied(0, 3, true);
        c.mark_copied(3, 2, true);
        c.mark_copied(5, 1, false);
        std::vector<float> L = L0;
        c.on_read_row(L.data(), V, 1, R);
        expect_row(L, &L0[1 * V], &H0[1 * E], 1, true, "lazy read");
        expect_row(L, &L0[0 * V], &H0[0 * E], 0, false, "lazy untouched");
        CHECK(c.n_rescored == 1, "lazy rescored %llu", (unsigned long long) c.n_rescored.load());
        c.on_read_row(L.data(), V, 1, R);
        CHECK(c.n_rescored == 1 && c.n_read_distinct == 1, "second read: no rescore, distinct 1");
        c.on_read_row(L.data(), V, 5, R);
        CHECK(c.n_read_coarse == 1 && c.n_h_mismatch == 1, "row without h is coarse");
        c.on_read_all(L.data(), V, R);
        for (int r = 0; r < 5; ++r) { expect_row(L, &L0[r * V], &H0[r * E], r, true, "read all"); }
        expect_row(L, &L0[5 * V], &H0[5 * E], 5, false, "coarse row untouched");
        CHECK(c.n_rescored == 5 && c.n_read_distinct == 6 && c.n_read_coarse == 2,
              "counters rescored %llu distinct %llu coarse %llu", (unsigned long long) c.n_rescored.load(),
              (unsigned long long) c.n_read_distinct.load(), (unsigned long long) c.n_read_coarse.load());
        // a new call invalidates every row
        c.reset(R);
        c.on_read_row(L.data(), V, 0, R);
        CHECK(c.n_rescored == 5 && c.n_read_distinct == 6, "rows of a reset call are NONE (not counted)");
    }
    // eager (n_out <= eager_max) and threaded == serial, bit for bit
    {
        std::vector<float> Ls = L0, Lt = L0;
        llama_d1_ctx cs(head, NTOP, 64, 1), ct(head, NTOP, 64, 4);
        for (auto * c : { &cs, &ct }) {
            c->reset(R);
            std::memcpy(c->h_row(0), H0.data(), sizeof(float) * R * E);
            c->mark_copied(0, R, true);
        }
        cs.on_read_row(Ls.data(), V, 2, R);
        ct.on_read_row(Lt.data(), V, 2, R);
        // async rows: the row being read is exact on return; the rest after drain (llama_get_logits* synchronize)
        expect_row(Lt, &L0[2 * V], &H0[2 * E], 2, true, "async read row");
        CHECK(ct.async_pending, "threaded fused read returns before the pool is done");
        ct.drain();
        CHECK(cs.n_rescored == R && ct.n_rescored == R, "eager rescored all %d rows on the first read", R);
        CHECK(cs.n_read_distinct == 1, "eager: distinct counts reads, not rescored rows");
        for (int r = 0; r < R; ++r) { expect_row(Lt, &L0[r * V], &H0[r * E], r, true, "threaded"); }
        CHECK(std::memcmp(Ls.data(), Lt.data(), sizeof(float) * R * V) == 0, "threaded == serial bit for bit");
        CHECK(cs.n_read_coarse == 0 && ct.n_read_coarse == 0, "no coarse reads");
    }
    // output_reorder: swaps keep h, state and logits aligned
    {
        llama_d1_ctx c(head, NTOP, 0, 1);
        c.reset(R);
        std::memcpy(c.h_row(0), H0.data(), sizeof(float) * R * E);
        c.mark_copied(0, 4, true);
        c.mark_copied(4, 2, false);
        std::vector<float> L = L0;
        // swap rows 0 <-> 4 the way output_reorder does
        for (int64_t k = 0; k < V; ++k) { std::swap(L[0 * V + k], L[4 * V + k]); }
        c.swap_rows(0, 4);
        c.on_read_row(L.data(), V, 4, R);  // row 4 now holds old row 0 (PENDING)
        expect_row(L, &L0[0 * V], &H0[0 * E], 4, true, "swapped pending");
        c.on_read_row(L.data(), V, 0, R);  // row 0 now holds old row 4 (COARSE)
        expect_row(L, &L0[4 * V], &H0[4 * E], 0, false, "swapped coarse");
        CHECK(c.n_rescored == 1 && c.n_read_coarse == 1, "swap counters");
    }
    // N = 0 negative control: every row coarse
    {
        llama_d1_ctx c(head, 0, 64, 2);
        c.reset(R);
        c.mark_copied(0, R, true);
        std::vector<float> L = L0;
        c.on_read_all(L.data(), V, R);
        CHECK(c.n_rescored == 0 && c.n_read_coarse == (uint64_t) R, "N=0: rescored 0, coarse %d", R);
        CHECK(std::memcmp(L.data(), L0.data(), sizeof(float) * R * V) == 0, "N=0 leaves logits untouched");
    }
    std::printf("%s\n", llama_d1_ctx(head, 1, 1, 1).summary().c_str());
}

// [EXACTHEAD] bf16 rows + output_exact.gl2 + the certificate. With sigma = sqrt(32 * G) the bound is a hard
// (Cauchy-Schwarz) bound, so after certify the row's top k must equal the exact top k over the whole vocab.
struct cert_sidecar {
    std::string        path;
    std::vector<float> w;      // [V][E] exact rows (bf16 values)
    std::vector<float> coarse; // [V][E] coarse rows
    llama_d1_src       src;
};

static cert_sidecar make_cert_sidecar(const std::string & dir, const char * name, int tamper) {
    cert_sidecar sc;
    sc.path = dir + "/" + name;
    const int64_t G = E / 32;
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> wf(V * E);
    for (auto & x : wf) { x = nd(rng); }
    std::vector<ggml_bf16_t> wb(V * E);
    ggml_fp32_to_bf16_row(wf.data(), wb.data(), V * E);
    sc.w.resize(V * E);
    ggml_bf16_to_fp32_row(wb.data(), sc.w.data(), V * E);
    // coarse = exact + per-row noise; every 50th row gets a large error (the rows the certificate has to catch)
    sc.coarse = sc.w;
    std::vector<float> gl2(V * G);
    for (int64_t i = 0; i < V; ++i) {
        const float scale = i % 50 == 7 ? 0.08f : 0.004f;
        std::normal_distribution<float> ne(0.0f, scale);
        for (int64_t g = 0; g < G; ++g) {
            double s2 = 0;
            for (int q = 0; q < 32; ++q) {
                const float e = ne(rng);
                sc.coarse[i * E + g * 32 + q] += e;
                s2 += (double) e * e;
            }
            gl2[i * G + g] = (float) std::sqrt(s2) * 1.0001f; // rounded up
        }
    }

    ggml_init_params ip = { ggml_tensor_overhead() * 4 + (size_t) V * E * 2 + (size_t) V * G * 4 + 8192, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, E, V);
    ggml_set_name(t, "output_exact.weight");
    std::memcpy(t->data, wb.data(), ggml_nbytes(t));
    ggml_tensor * tg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, G, V);
    ggml_set_name(tg, "output_exact.gl2");
    std::memcpy(tg->data, gl2.data(), ggml_nbytes(tg));

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "d1.rows_type", tamper == 1 ? "f16" : "bf16");
    gguf_set_val_u64(g, "d1.source_bytes", 4096);
    gguf_set_val_u64(g, "d1.source_fnv64", 7);
    gguf_set_val_u64(g, "d1.self_bytes", ggml_nbytes(t));
    gguf_set_val_u64(g, "d1.self_fnv64", llama_d1_fnv64_ends((const uint8_t *) t->data, ggml_nbytes(t)));
    gguf_set_val_u64(g, "d1.gl2_fnv64", llama_d1_fnv64_ends((const uint8_t *) tg->data, ggml_nbytes(tg)) ^ (tamper == 2 ? 1 : 0));
    gguf_add_tensor(g, t);
    gguf_add_tensor(g, tg);
    CHECK(gguf_write_to_file(g, sc.path.c_str(), false), "write %s", sc.path.c_str());
    gguf_free(g);
    ggml_free(ctx);

    sc.src.valid  = true;
    sc.src.type   = GGML_TYPE_AIF25_U2;
    sc.src.nbytes = 4096;
    sc.src.fnv    = 7;
    return sc;
}

static void test_cert(const std::string & dir) {
    std::string err;
    {
        cert_sidecar bad = make_cert_sidecar(dir, "d1-test-bf16-bad1.gguf", 1);
        CHECK(!llama_d1_head_open(bad.path.c_str(), bad.src, E, V, err), "bf16 with rows_type f16 refused");
        cert_sidecar bad2 = make_cert_sidecar(dir, "d1-test-bf16-bad2.gguf", 2);
        CHECK(!llama_d1_head_open(bad2.path.c_str(), bad2.src, E, V, err), "gl2 fingerprint mismatch refused");
        std::remove(bad.path.c_str());
        std::remove(bad2.path.c_str());
    }
    cert_sidecar sc = make_cert_sidecar(dir, "d1-test-bf16.gguf", 0);
    auto head = llama_d1_head_open(sc.path.c_str(), sc.src, E, V, err);
    CHECK(head != nullptr, "bf16 open: %s", err.c_str());
    if (!head) { return; }
    CHECK(head->type == GGML_TYPE_BF16 && head->gl2 && head->n_group == E / 32, "bf16 head with gl2");

    // scope: the 8 GB file's own AIF25_U2 head is a coarse head
    {
        llama_d1_model_slot slot; slot.src = sc.src;
        llama_d1_scope_in in;
        in.head_path = sc.path.c_str(); in.out_type = GGML_TYPE_AIF25_U2; in.out_ne0 = E; in.out_ne1 = V;
        in.n_embd = E; in.n_vocab = V;
        std::shared_ptr<llama_d1_head> got; std::string msg;
        CHECK(llama_d1_scope(in, slot, got, msg) == LLAMA_D1_SCOPE_ON && got, "AIF25_U2 head on: %s", msg.c_str());
    }

    const int R = 40, NTOP = 8, K = 4;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> H(R * E), L0(R * V), X(R * V);
    for (auto & x : H) { x = nd(rng); }
    for (int r = 0; r < R; ++r) {
        for (int64_t i = 0; i < V; ++i) {
            double c = 0, e = 0;
            for (int64_t k = 0; k < E; ++k) {
                c += (double) sc.coarse[i * E + k] * H[r * E + k];
                e += (double) sc.w[i * E + k] * H[r * E + k];
            }
            L0[r * V + i] = (float) c;
            X[r * V + i]  = (float) e;
        }
    }
    const auto top_ids = [&](const float * row, int n) { return ref_top(row, V, n); };

    llama_d1_cert cert;
    cert.on = true; cert.k = K; cert.sigma = std::sqrt(32.0f * (E / 32)); cert.margin = 1e-3f;
    int miss_plain = 0, miss_cert = 0;
    for (const bool on : { false, true }) {
        llama_d1_cert c0 = cert; c0.on = on;
        llama_d1_ctx c(head, NTOP, 64, 1, c0);
        c.reset(R);
        std::memcpy(c.h_row(0), H.data(), sizeof(float) * R * E);
        c.mark_copied(0, R, true);
        std::vector<float> L = L0;
        c.on_read_all(L.data(), V, R);
        for (int r = 0; r < R; ++r) {
            const bool same = top_ids(&L[r * V], K) == top_ids(&X[r * V], K);
            (on ? miss_cert : miss_plain) += same ? 0 : 1;
        }
        if (on) {
            CHECK(c.n_cert_rows == (uint64_t) R, "cert ran on every row");
            CHECK(c.n_cert_forced > 0, "cert forced some rows (%llu)", (unsigned long long) c.n_cert_forced.load());
            std::printf("%s\n", c.summary().c_str());
        }
    }
    CHECK(miss_cert == 0, "certified top %d == exact top %d on every row (%d misses)", K, K, miss_cert);
    // audit (LLAMA_D1_AUDIT): with a hard certificate at k = 20 the served top 20 and the sampled distribution are
    // the full exact head's, bit for bit; plain N = 20 without the certificate is not
    for (const bool on : { false, true }) {
        llama_d1_cert c0 = cert; c0.on = on; c0.k = 20;
        llama_d1_ctx c(head, 20, 64, 2, c0);
        c.audit_every = 1;
        c.reset(R);
        std::memcpy(c.h_row(0), H.data(), sizeof(float) * R * E);
        c.mark_copied(0, R, true);
        std::vector<float> L = L0;
        c.on_read_all(L.data(), V, R);
        std::printf("%s %s\n", on ? "cert" : "plain", c.audit_summary().c_str());
        CHECK(c.audit.rows == (uint64_t) R, "audit saw every row");
        if (on) {
            CHECK(c.audit.top_same == (uint64_t) R && c.audit.dist_same == (uint64_t) R && c.audit.dlogit_max == 0.0,
                  "certified rows match the full exact head");
        } else {
            CHECK(c.audit.top_same < (uint64_t) R, "uncertified N = 20 misses some rows (negative control)");
        }
        CHECK(c.audit.coarse_top_same < (uint64_t) R, "coarse rows differ from exact (negative control)");
        CHECK(c.audit.z_n == (uint64_t) R * 20 && c.audit.z_max < 6.0, "aligned h: z_max %.3g < 6", c.audit.z_max);
    }
    // misaligned h (row r gets h of row r+1): the z check must fire
    {
        llama_d1_ctx c(head, 20, 64, 1, cert);
        c.audit_every = 1;
        c.reset(R);
        for (int r = 0; r < R; ++r) { std::memcpy(c.h_row(r), &H[((r + 1) % R) * E], sizeof(float) * E); }
        c.mark_copied(0, R, true);
        std::vector<float> L = L0;
        c.on_read_all(L.data(), V, R);
        std::printf("misaligned %s\n", c.audit_summary().c_str());
        CHECK(c.audit.z_max > 20.0 && c.audit.z_gt6 > c.audit.z_n / 2, "misaligned h: z check fires (z_max %.3g, %llu/%llu > 6)",
              c.audit.z_max, (unsigned long long) c.audit.z_gt6, (unsigned long long) c.audit.z_n);
    }
    // [EXACTHEAD] fused pick == separate select + full certificate scan, bit for bit (cert on and off, 1 and 4 threads,
    // many chunks per row), including rows where the certificate falls back to the full scan (coarse shifted by +10
    // puts T below every chunk's floor)
    for (const float shift : { 0.0f, 10.0f }) {
        std::vector<float> Ls = L0;
        for (auto & x : Ls) { x += shift; }
        for (const bool on : { false, true }) {
            for (const int thr : { 1, 4 }) {
              for (const int eager : { 64, 0 }) { // 0: lazy, one row per call, so 4 threads split each row in 4 chunks
                llama_d1_cert c0 = cert; c0.on = on; c0.sigma = 6.0f; c0.margin = 0.05f;
                std::vector<float> Lf = Ls, Lo = Ls;
                llama_d1_ctx cf(head, NTOP, eager, thr, c0), co(head, NTOP, eager, thr, c0);
                cf.fused_min_chunk = 37;
                co.fused = false;
                for (auto * c : { &cf, &co }) {
                    c->reset(R);
                    std::memcpy(c->h_row(0), H.data(), sizeof(float) * R * E);
                    c->mark_copied(0, R, true);
                }
                for (int r = 0; r < R; ++r) {
                    cf.on_read_row(Lf.data(), V, r, R);
                    co.on_read_row(Lo.data(), V, r, R);
                }
                CHECK(cf.n_chunk == (thr == 4 && eager == 0 ? 4 : 1), "fused: %d chunks per row", cf.n_chunk);
                cf.drain();
                co.drain();
                CHECK(std::memcmp(Lf.data(), Lo.data(), sizeof(float) * R * V) == 0,
                      "fused == separate bit for bit (shift %g, cert %d, threads %d)", shift, (int) on, thr);
                CHECK(cf.n_cert_forced == co.n_cert_forced, "fused forced %llu == separate %llu",
                      (unsigned long long) cf.n_cert_forced.load(), (unsigned long long) co.n_cert_forced.load());
                if (on) {
                    // shift 0: a few noisy rows may fall back too (N = 8 here); shift 10: every row must
                    CHECK(shift == 0.0f ? cf.n_fused_fallback < (uint64_t) R / 2 : cf.n_fused_fallback == (uint64_t) R,
                          "fused fallback %llu (shift %g)", (unsigned long long) cf.n_fused_fallback.load(), shift);
                    std::printf("fused shift %g threads %d chunks %d: %s\n", shift, thr, cf.n_chunk, cf.summary().c_str());
                }
              }
            }
        }
    }
    std::printf("cert test: top-%d misses plain N=%d: %d/%d, certified: %d/%d\n", K, NTOP, miss_plain, R, miss_cert, R);
    std::remove(sc.path.c_str());
}

// [EXACTHEAD] llama_d1_scan_chunk: chunk heaps merge to the select_top set; buf holds every id >= the floor
static void test_scan_chunk() {
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (const int64_t nv : { 5, 64, 1000, 40007 }) {
        std::vector<float> row(nv);
        for (auto & x : row) { x = std::round(nd(rng) * 8.0f) / 8.0f; } // ties
        for (const int n : { 1, 8, 64 }) {
            std::vector<int32_t> want;
            llama_d1_select_top(row.data(), nv, n, want);
            std::sort(want.begin(), want.end());
            for (const int nc : { 1, 3, 8 }) {
                for (const float D : { 0.0f, 0.5f, 3.0f }) {
                    std::vector<llama_d1_cand> all, heap;
                    std::vector<int32_t> buf;
                    const int64_t step = ((nv + nc - 1) / nc + 7) & ~(int64_t) 7;
                    int bad = 0;
                    for (int c = 0; c < nc; ++c) {
                        const int64_t i0 = std::min(nv, c * step), i1 = std::min(nv, i0 + step);
                        const float floor = llama_d1_scan_chunk(row.data(), i0, i1, n, D, true, heap, buf);
                        all.insert(all.end(), heap.begin(), heap.end());
                        std::vector<uint8_t> in(nv, 0);
                        for (const int32_t id : buf) { in[id] = 1; }
                        for (int64_t i = i0; i < i1; ++i) { bad += row[i] >= floor && !in[i]; }
                    }
                    std::sort(all.begin(), all.end(), [](const llama_d1_cand & a, const llama_d1_cand & b) {
                        return a.v > b.v || (a.v == b.v && a.id < b.id); });
                    if ((int64_t) all.size() > n) { all.resize(n); }
                    std::vector<int32_t> got;
                    for (const auto & x : all) { got.push_back(x.id); }
                    std::sort(got.begin(), got.end());
                    CHECK(got == want && bad == 0, "scan_chunk nv %lld n %d chunks %d D %g: set %d, missing buf %d",
                          (long long) nv, n, nc, D, (int) (got == want), bad);
                }
            }
        }
    }
}

int main(int argc, char ** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    test_select();
    test_scan_chunk();
    test_head_and_scope(dir);
    test_ctx(dir);
    test_cert(dir);
    std::remove((dir + "/d1-test-f16.gguf").c_str());
    std::remove((dir + "/d1-test-q8.gguf").c_str());
    if (g_fail) {
        std::printf("test-d1-rescore: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("test-d1-rescore: OK\n");
    return 0;
}
