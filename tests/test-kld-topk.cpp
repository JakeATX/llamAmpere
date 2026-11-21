// KL@10 / EAR@10 / full-vocabulary EAR of the KL-divergence scorer (tools/perplexity/kld-topk.h) on hand-made
// distributions with known answers. Exit 0 = pass.
#include "kld-topk.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_fail = 0;

static void expect(const char * what, double got, double want, double tol = 1e-6) {
    const bool ok = std::fabs(got - want) <= tol;
    std::printf("%-44s got %.9f want %.9f %s\n", what, got, want, ok ? "ok" : "FAIL");
    g_fail += ok ? 0 : 1;
}

struct metrics { double kl10, ear10, ear; std::vector<int> top; };

// p, q: probabilities over the same vocabulary (need not sum to 1 for q: the scorer works from log-probs).
static metrics score(const std::vector<double> & p, const std::vector<double> & q) {
    kld_topk::top_k top;
    double ear = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        top.push((int) i, (float) std::log(p[i]));
        ear += kld_topk::accept((float) std::log(p[i]), (float) std::log(q[i]));
    }
    float lq[kld_topk::K];
    for (int k = 0; k < top.n; ++k) {
        lq[k] = (float) std::log(q[top.id[k]]);
    }
    const kld_topk::result r = kld_topk::over(top.lp, lq, top.n);
    metrics m{r.kl, r.ear, ear, {}};
    for (int k = 0; k < top.n; ++k) {
        m.top.push_back(top.id[k]);
    }
    return m;
}

int main() {
    // 1. identical distributions: KL@10 = 0, EAR@10 = EAR = 1
    {
        std::vector<double> p = {0.4, 0.3, 0.2, 0.1};
        const metrics m = score(p, p);
        expect("identical: KL@10", m.kl10, 0.0);
        expect("identical: EAR@10", m.ear10, 1.0);
        expect("identical: EAR", m.ear, 1.0);
    }
    // 2. two tokens, p = (0.5, 0.5), q = (0.9, 0.1): KL = 0.5 ln(25/9), EAR = 0.5 + 0.1 = 0.6 (fewer than 10 tokens)
    {
        const metrics m = score({0.5, 0.5}, {0.9, 0.1});
        expect("two-token: KL@10", m.kl10, 0.5*std::log(25.0/9.0));
        expect("two-token: EAR@10", m.ear10, 0.6);
        expect("two-token: EAR", m.ear, 0.6);
    }
    // 3. renormalization: p = 10 x 0.09 + 2 x 0.05, q = 10 x 0.045 + 2 x 0.275. Inside p's top 10 both are uniform after
    //    renormalizing, so KL@10 = 0 and EAR@10 = 1, while the full EAR = 10 x 0.045 + 2 x 0.05 = 0.55.
    {
        std::vector<double> p(12, 0.09), q(12, 0.045);
        p[3] = p[7] = 0.05;
        q[3] = q[7] = 0.275;
        const metrics m = score(p, q);
        expect("renorm: KL@10", m.kl10, 0.0);
        expect("renorm: EAR@10", m.ear10, 1.0);
        expect("renorm: EAR", m.ear, 0.55);
        bool skip = true;
        for (int id : m.top) skip = skip && id != 3 && id != 7;
        expect("renorm: top-10 excludes ids 3 and 7", skip ? 1 : 0, 1);
    }
    // 4. a reordered top 10: p over 11 tokens = (0.2, 0.1 x 7, 0.04, 0.03, 0.03); q swaps tokens 0 and 1. Top 10 of p =
    //    ids 0..9 (the 0.03 tie keeps id 9), mass 0.97 on both sides. KL@10 = (0.2 ln 2 + 0.1 ln 0.5) / 0.97 = 0.1 ln 2 / 0.97;
    //    EAR@10 = (0.1 + 0.1 + 0.6 + 0.04 + 0.03) / 0.97 = 0.87 / 0.97; full EAR = 0.87 + 0.03 = 0.90 (TV = 0.1).
    {
        std::vector<double> p = {0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.04, 0.03, 0.03};
        std::vector<double> q = p;
        q[0] = 0.1; q[1] = 0.2;
        const metrics m = score(p, q);
        expect("swap: KL@10", m.kl10, 0.1*std::log(2.0)/0.97);
        expect("swap: EAR@10", m.ear10, 0.87/0.97);
        expect("swap: EAR", m.ear, 0.90);
        expect("swap: top-10 = ids 0..9", m.top.size() == 10 && m.top.back() == 9 ? 1 : 0, 1);
    }
    // 5. ties: 12 equal tokens keep ids 0..9 in order; a later larger value goes first
    {
        std::vector<double> p(12, 1.0/13.0);
        p[11] = 2.0/13.0;
        const metrics m = score(p, p);
        const std::vector<int> want = {11, 0, 1, 2, 3, 4, 5, 6, 7, 8};
        expect("ties: top-10 order", m.top == want ? 1 : 0, 1);
    }
    // 6. disjoint support inside the top 10 cannot happen with q > 0, but a far-off q gives EAR close to 0 and a large KL
    {
        std::vector<double> p(20, 0.0), q(20, 0.0);
        for (int i = 0; i < 20; ++i) {
            p[i] = i < 10 ? 0.0999 : 0.0001;
            q[i] = i < 10 ? 1e-9 : 0.1 - 1e-9;
        }
        const metrics m = score(p, q);
        expect("far: EAR (= 10 x 1e-9 + 10 x 1e-4)", m.ear, 10*1e-9 + 10*1e-4, 1e-8);
        expect("far: EAR@10 (q uniform after renorm)", m.ear10, 1.0, 1e-6);
    }
    std::printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
