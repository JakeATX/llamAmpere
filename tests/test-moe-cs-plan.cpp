// fork AIPFIX: CPU test of the CPU-split planner's greedy step (moe-cs-greedy.h, shared with the plan kernel) and of
// the per-type cost table (GGML_MOE_CPU_SPLIT_PROBE_PER_TYPE=1).
//   1. moe_cs_greedy == the planner loop as it was written inline before (random cases, exact).
//   2. Two layers with the same misses and different host costs: the cheaper one sends more experts to the host.
//   3. moe_cs_type_costs: one probe per distinct type triple, per-layer costs, failed probe -> global, reuse.
#include "../ggml/src/ggml-cuda/moe-cs-greedy.h"

#include <cstdio>
#include <cstring>
#include <random>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// The inline loop of the plan kernel before the refactor (moe-cache.cu, fork MOEGAP), verbatim arithmetic.
static int old_loop(const moe_cs_greedy_cfg & c, float gpu, int ncand, const short * order, const short * cnt,
        const unsigned char flag[3][64], int * take, int * first, int * nr_out, unsigned long long * parts_out,
        unsigned long long * partial_out) {
    float cpu = 0.0f;
    int ne = 0, nr = 0;
    unsigned long long parts = 0, partial = 0;
    const float route_ns = c.route_ns;
    first[0] = 0;
    for (int r = 0; r < ncand && ne < c.max_exp; ++r) {
        const int i = order[r];
        const int cc = cnt[i] / 4;
        if (nr + cc > c.max_route) { break; }
        float saved = 0.0f;
        int nm = 0;
        for (int k = 0; k < 3; ++k) {
            if ((flag[k][i] & 3) == 3) { saved += c.bytes[k]; ++nm; }
        }
        const float cpu2 = c.fixed_ns + (float) (nr + cc) * route_ns;
        const float gpu2 = gpu - saved * c.w;
        const float now = fmaxf(cpu, fmaxf(c.floor_ns, gpu));
        const float next = fmaxf(cpu2, fmaxf(c.floor_ns, gpu2));
        if (c.floor_ns > 0.0f ? next > now : next >= now) { break; }
        cpu = cpu2;
        gpu = gpu2;
        take[ne] = i;
        ++ne;
        nr += cc;
        parts += (unsigned long long) nm;
        partial += nm < 3;
        first[ne] = nr;
    }
    *nr_out = nr; *parts_out = parts; *partial_out = partial;
    return ne;
}

int main() {
    std::mt19937 rng(1234);
    // 1. identity with the old inline loop
    for (int it = 0; it < 20000; ++it) {
        const int ncand = (int) (rng() % 40);
        short order[64], cnt[64];
        unsigned char flag[3][64];
        for (int i = 0; i < 64; ++i) {
            order[i] = (short) (rng() % 64);
            cnt[i] = (short) (4 * (1 + rng() % 4));
            for (int k = 0; k < 3; ++k) { flag[k][i] = (unsigned char) (rng() % 8); }
        }
        moe_cs_greedy_cfg c = { 2000.0f + (float) (rng() % 120000), 5000.0f + (float) (rng() % 30000),
            (float) (rng() % 100) / 1000.0f + 0.01f, (rng() & 1) ? 0.0f : (float) (rng() % 40000),
            { 1.4e6f, 1.4e6f, 1.1e6f }, 1 + (int) (rng() % 16), 1 + (int) (rng() % 32) };
        const float gpu = (float) (rng() % 2000000);
        int ta[16], tb[16], fa[17], fb[17], na, nb;
        unsigned long long pa, pb, qa, qb;
        const unsigned char * const fl[3] = { flag[0], flag[1], flag[2] };
        const int ea = moe_cs_greedy(c, gpu, ncand, order, cnt, fl, ta, fa, &na, &pa, &qa);
        const int eb = old_loop(c, gpu, ncand, order, cnt, flag, tb, fb, &nb, &pb, &qb);
        bool same = ea == eb && na == nb && pa == pb && qa == qb;
        for (int j = 0; same && j < ea; ++j) { same = ta[j] == tb[j] && fa[j + 1] == fb[j + 1]; }
        CHECK(same, "case %d: greedy differs from the old loop (ne %d vs %d)", it, ea, eb);
    }
    // 2. two layers, same misses (8 experts x 1 route, all three projections missing), different host costs.
    //    Link 25 GB/s (0.04 ns/byte), 3.9 MB per expert -> ~156 us of copy per expert; 8 copies = ~1.25 ms on the card.
    {
        short order[64], cnt[64];
        unsigned char flag[3][64];
        for (int i = 0; i < 8; ++i) { order[i] = (short) i; cnt[i] = 4; for (int k = 0; k < 3; ++k) { flag[k][i] = 3; } }
        const float bytes[3] = { 1.4e6f, 1.4e6f, 1.1e6f };
        float gpu = 0.0f;
        for (int k = 0; k < 3; ++k) { gpu += 8.0f * bytes[k]; }
        gpu *= 0.04f;
        const unsigned char * const fl[3] = { flag[0], flag[1], flag[2] };
        int take[16], first[17], nr;
        unsigned long long parts, partial;
        const moe_cs_greedy_cfg fast = { 62.2e3f, 18e3f, 0.04f, 0.0f, { bytes[0], bytes[1], bytes[2] }, 8, 24 };   // 3XS-like
        const moe_cs_greedy_cfg slow = { 94.0e3f, 18e3f, 0.04f, 0.0f, { bytes[0], bytes[1], bytes[2] }, 8, 24 };   // AI+ float host
        const int ef = moe_cs_greedy(fast, gpu, 8, order, cnt, fl, take, first, &nr, &parts, &partial);
        const int es = moe_cs_greedy(slow, gpu, 8, order, cnt, fl, take, first, &nr, &parts, &partial);
        printf("two costs: card %.0f us; host 62.2 us/route -> %d experts on the host, 94.0 us/route -> %d\n", gpu / 1e3, ef, es);
        CHECK(ef > es, "cheaper host cost must take more experts (%d vs %d)", ef, es);
        // closed form: the greedy stops before the host side passes the card side
        auto expect = [&](float route) { int n = 0; float g = gpu; float cpu = 0;
            while (n < 8) { const float c2 = 18e3f + (n + 1) * route, g2 = g - 3.9e6f * 0.04f; if (fmaxf(c2, g2) >= fmaxf(cpu, g)) break; cpu = c2; g = g2; ++n; }
            return n; };
        CHECK(ef == expect(62.2e3f) && es == expect(94.0e3f), "closed form %d/%d vs %d/%d", expect(62.2e3f), expect(94.0e3f), ef, es);
    }
    // Forced shares ignore the cost probe but keep the expert and route bounds.
    {
        short order[8], cnt[8];
        unsigned char flag[3][8];
        for (int i = 0; i < 8; ++i) {
            order[i] = (short) i;
            cnt[i] = 4;
            for (int k = 0; k < 3; ++k) { flag[k][i] = 3; }
        }
        const unsigned char * const fl[3] = {flag[0], flag[1], flag[2]};
        moe_cs_greedy_cfg c = {1e9f, 1e9f, 0.04f, 0.0f, {1e6f, 1e6f, 1e6f}, 8, 24};
        int take[8], first[9], nr;
        unsigned long long parts, partial;
        for (float share : {0.0f, 0.6f, 0.8f, 1.0f}) {
            c.target_parts = (int) ceilf(share * 24);
            const int ne = moe_cs_greedy(c, 24e6f * c.w, 8, order, cnt, fl, take, first, &nr, &parts, &partial);
            const int expected = (c.target_parts + 2) / 3;
            CHECK(ne == expected && nr == expected && parts == (unsigned) (3 * expected) && partial == 0,
                  "share %.1f: got %d experts / %llu parts", share, ne, parts);
            for (int i = 0; i < ne; ++i) { CHECK(take[i] == i && first[i + 1] == i + 1, "forced route offsets"); }
        }
        c.target_parts = 24;
        c.max_exp = 2;
        CHECK(moe_cs_greedy(c, 0, 8, order, cnt, fl, take, first, &nr, &parts, &partial) == 2 && parts == 6,
              "forced share must obey the expert cap");
        c.max_exp = 8;
        c.max_route = 3;
        cnt[0] = 8;
        CHECK(moe_cs_greedy(c, 0, 8, order, cnt, fl, take, first, &nr, &parts, &partial) == 2 && nr == 3,
              "forced share must obey the route cap");
        c.max_route = 24;
        cnt[0] = 4;
        flag[1][0] = flag[2][0] = 0;
        flag[2][1] = 0;
        c.target_parts = 3;
        CHECK(moe_cs_greedy(c, 0, 8, order, cnt, fl, take, first, &nr, &parts, &partial) == 2 && parts == 3 && partial == 2,
              "partial experts count only their missed projections");
        c.target_parts = -1;
        CHECK(moe_cs_greedy(c, 0, 8, order, cnt, fl, take, first, &nr, &parts, &partial) == 0,
              "auto mode still uses the cost probe");
    }
    // 3. per-type cost table
    {
        const std::array<int, 3> A = {23, 23, 20}, B = {150, 150, 150}, none = {-1, -1, -1};
        std::vector<std::array<int, 3>> key = { A, none, B, A, B, B };
        int calls = 0, last = -1;
        auto probe = [&](int l, double * r, double * f) { ++calls; last = l; *r = l == 0 ? 60e3 : 90e3; *f = 15e3; return true; };
        std::map<std::array<int, 3>, std::array<double, 2>> known;
        std::vector<double> rt, fx;
        int n = moe_cs_type_costs(key, probe, 60e3, 15e3, known, rt, fx);
        CHECK(n == 2 && calls == 2, "two triples -> two probes (got %d / %d)", n, calls);
        CHECK(rt[0] == 60e3 && rt[3] == 60e3 && rt[2] == 90e3 && rt[4] == 90e3 && rt[5] == 90e3 && rt[1] == 0.0,
              "per-layer costs follow their triple");
        CHECK(last == 2, "the first layer of a triple is the one probed (got %d)", last);
        // second session: no new probes
        n = moe_cs_type_costs(key, probe, 60e3, 15e3, known, rt, fx);
        CHECK(n == 0 && calls == 2, "known triples are reused");
        // failed probe -> global cost
        std::map<std::array<int, 3>, std::array<double, 2>> k2;
        auto bad = [&](int, double *, double *) { return false; };
        moe_cs_type_costs(key, bad, 61e3, 14e3, k2, rt, fx);
        CHECK(rt[0] == 61e3 && fx[2] == 14e3, "a failed probe falls back to the global cost");
    }
    printf("test-moe-cs-plan: %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
