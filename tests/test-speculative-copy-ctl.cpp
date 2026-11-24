// asserts must stay live in Release builds: undefine NDEBUG before <cassert>
#undef NDEBUG

#include "speculative-copy-ctl.h"
#include "speculative-cost-depth.h"
#include "sampling-hash.h"

#include <cmath>

#include <cassert>
#include <cstdio>
#include <random>

typedef std::vector<llama_token> toks;

static rtb::stream_view view(const toks & h, llama_token last) {
    rtb::stream_view s;
    s.h = h.data();
    s.n = (int32_t) h.size();
    s.last = last;
    return s;
}

// reference for anchor_index::find by direct comparison over all earlier positions
static rtb::anchor_index::hit ref_find(const rtb::stream_view & s, int32_t F, int32_t after, int32_t p_end, int32_t max_back) {
    rtb::anchor_index::hit best, near;
    for (int32_t p = F - 1; p >= 3; --p) { // newest first
        if (p_end >= 0 && p >= p_end) {
            continue;
        }
        int32_t l = 0;
        while (l < max_back && p - l >= 0 && s[p - l] == s[F - l]) {
            ++l;
        }
        if (l < 4) {
            continue;
        }
        if (l > best.len) {
            best.p = p;
            best.len = l;
        }
        if (after >= 0 && p >= after) {
            near.p = p;
            near.len = l;
        }
    }
    if (near.p >= 0 && near.len >= std::min(best.len, (int32_t) 8)) {
        return near;
    }
    return best;
}

static void test_index_matches_reference() {
    std::mt19937 rng(42);
    for (int it = 0; it < 3000; ++it) {
        const int n = 5 + (int) (rng() % 200);
        const int alpha = 2 + (int) (rng() % 4);
        toks h(n);
        for (auto & t : h) {
            t = (llama_token) (rng() % alpha);
        }
        const llama_token last = (llama_token) (rng() % alpha);
        const auto s = view(h, last);
        const int32_t F = s.size() - 1;
        rtb::anchor_index idx;
        // index in two steps, as rounds do
        idx.extend(s, F / 2);
        idx.extend(s, F);
        const int32_t after = (rng() % 2) ? (int32_t) (rng() % (F + 1)) : -1;
        const int32_t p_end = (rng() % 3 == 0) ? (int32_t) (rng() % (F + 1)) : -1;
        const auto a = idx.find(s, F, after, p_end, 1 << 20, 64);
        const auto b = ref_find(s, F, after, p_end, 64);
        if (a.p != b.p || a.len != b.len) {
            fprintf(stderr, "index mismatch it=%d n=%d: got (%d,%d) want (%d,%d)\n", it, n, a.p, a.len, b.p, b.len);
            assert(false);
        }
    }
    printf("  index == reference: ok\n");
}

// a record-like source: R records of `len` tokens that share a template, so look-alike matches exist
static toks records(int R, int len, int base) {
    toks t;
    for (int r = 0; r < R; ++r) {
        for (int j = 0; j < len; ++j) {
            // template tokens 1000+j except one field that carries the record number
            t.push_back(j == len / 2 ? (llama_token) (base + r) : (llama_token) (1000 + j));
        }
    }
    return t;
}

// the cursor keeps copying the right record; without it the lookup lands in a look-alike one
static void test_cursor_follows_source() {
    const toks src = records(20, 12, 5000);
    toks h = src;
    h.push_back(7); // separator: the reply starts here
    const int32_t n_src = (int32_t) h.size();
    // the reply copies the source from record 3 on
    toks reply(src.begin() + 3 * 12, src.end());
    rtb::copy_flags fl;
    fl.cursor = true;
    rtb::copy_seq cs;
    cs.reset_request();
    // emit 5 tokens of the reply first (no proposal yet), then run rounds that commit whole proposals
    size_t pos = 0;
    for (; pos < 5; ++pos) {
        h.push_back(reply[pos]);
    }
    int n_rounds = 0, n_full = 0;
    while (pos + 1 < reply.size() && n_rounds < 200) {
        const auto s = view(h, reply[pos]);
        const auto span = rtb::copy_round(fl, cs, s, 8, n_src, false, 16, 8);
        n_rounds++;
        // the "target": commits the true reply; count how much of the span was right
        size_t k = 0;
        while (k < span.size() && pos + 1 + k < reply.size() && span[k] == reply[pos + 1 + k]) {
            ++k;
        }
        if (!span.empty() && k == span.size()) {
            n_full++;
        }
        // commit id_last + accepted + one bonus token
        const size_t adv = std::min(reply.size() - 1 - pos, k + 1);
        for (size_t j = 0; j < adv; ++j) {
            h.push_back(reply[pos + j]);
        }
        pos += adv;
    }
    printf("  cursor: %d rounds, %d full spans, %llu cursor hits\n", n_rounds, n_full, (unsigned long long) cs.n_cursor);
    assert(cs.n_cursor > 10);
    assert(n_full >= n_rounds - 3);
}

static void test_ctl_choice() {
    rtb::copy_ctl c;
    // nothing measured: no copy
    assert(c.choose(32, 3, false, false) == 0);
    // drafter rounds: 5 rows, 20 ms, 4 tokens per round
    for (int i = 0; i < 20; ++i) {
        c.time_alt(20.0, 4, 5);
    }
    // a copy source that is always followed: learn it
    for (int i = 0; i < 200; ++i) {
        c.learn(32, 32, 3, false, false);
    }
    const int w = c.choose(32, 3, false, false);
    printf("  ctl: high acceptance -> W = %d\n", w);
    assert(w >= 15);
    // a source that is never followed
    rtb::copy_ctl d = c;
    for (int i = 0; i < 200; ++i) {
        d.learn(0, 1, 0, false, false);
    }
    assert(d.choose(32, 0, false, false) == 0);
    // greedy and sampled rates are separate
    assert(d.choose(32, 3, false, true) == 0 || d.r[0].first[3] == rtb::copy_ctl::rates().first[3]);
    // measured copy rounds dominate the prior: make 32-row passes very slow
    rtb::copy_ctl e = c;
    for (int i = 0; i < 10; ++i) {
        e.time_copy(32, 1000.0);
        e.time_copy(16, 1000.0);
        e.time_copy(24, 1000.0);
        e.time_copy(48, 1000.0);
        e.time_copy(64, 1000.0);
        e.time_copy(96, 1000.0);
        e.time_copy(128, 1000.0);
        e.time_copy(8, 1000.0);
        e.time_copy(4, 1000.0);
        e.time_copy(2, 1000.0);
    }
    assert(e.choose(32, 3, false, false) == 0);
    printf("  ctl choice: ok\n");
}

// rename edit: every record's field token 1000+3 becomes 9999 in the reply
static void test_edit_rules() {
    const toks src = records(30, 12, 5000);
    toks h = src;
    h.push_back(7);
    const int32_t n_src = (int32_t) h.size();
    toks reply;
    for (llama_token t : src) {
        reply.push_back(t == 1003 ? (llama_token) 9999 : t);
    }
    rtb::copy_flags fl;
    fl.cursor = true;
    fl.edit = true;
    fl.ctl = true;
    rtb::copy_seq cs;
    cs.reset_request();
    for (int i = 0; i < 20; ++i) {
        cs.ctl.time_alt(20.0, 3, 5);
    }
    size_t pos = 0;
    for (; pos < 5; ++pos) {
        h.push_back(reply[pos]);
    }
    int n_rounds = 0, n_copy_tok = 0;
    while (pos + 1 < reply.size() && n_rounds < 2000) {
        const auto s = view(h, reply[pos]);
        const auto span = rtb::copy_round(fl, cs, s, 32, n_src, false, 16, 8);
        n_rounds++;
        size_t k = 0;
        while (k < span.size() && pos + 1 + k < reply.size() && span[k] == reply[pos + 1 + k]) {
            ++k;
        }
        cs.on_verified((int32_t) k);
        n_copy_tok += (int) k;
        // the copy rounds cost 10 ms + 0.2 ms per row, the drafter's 20 ms for 3 tokens
        if (cs.kind == rtb::copy_seq::ROUND_COPY || cs.kind == rtb::copy_seq::ROUND_EDIT) {
            cs.ctl.time_copy(cs.rows, 10.0 + 0.2 * cs.rows);
        }
        const size_t adv = std::min(reply.size() - 1 - pos, span.empty() ? (size_t) 3 : k + 1);
        for (size_t j = 0; j < adv; ++j) {
            h.push_back(reply[pos + j]);
        }
        pos += adv;
    }
    int usable = 0;
    for (int i = 0; i < cs.edits.n_rules; ++i) {
        usable += cs.edits.rules[i].usable() ? 1 : 0;
    }
    printf("  edit: %d rounds for %zu tokens, %llu edit spans (%llu took the edit), %d usable rules\n", n_rounds,
            reply.size(), (unsigned long long) cs.n_edit, (unsigned long long) cs.n_edit_hit, usable);
    for (int i = 0; i < cs.edits.n_rules; ++i) {
        const auto & r = cs.edits.rules[i];
        printf("    rule %d: nf=%d nt=%d from0=%d to0=%d places=%d edit=%d kept=%d miss=%d\n", i, r.nf, r.nt, r.from[0], r.to[0], r.places, r.n_edit, r.n_kept, r.miss);
    }
    assert(usable >= 1);
    assert(cs.n_edit_hit >= 10);
    assert(n_rounds < (int) reply.size() / 6);
}

// random text: the controller learns that copies are not followed and stays with the drafter
static void test_ctl_random_text_stays_off() {
    std::mt19937 rng(7);
    toks h;
    for (int i = 0; i < 4000; ++i) {
        h.push_back((llama_token) (rng() % 50));
    }
    const int32_t n_src = (int32_t) h.size();
    rtb::copy_flags fl;
    fl.ctl = true;
    fl.cursor = true;
    rtb::copy_seq cs;
    cs.reset_request();
    for (int i = 0; i < 20; ++i) {
        cs.ctl.time_alt(20.0, 3, 5);
    }
    int n_copy = 0;
    for (int r = 0; r < 3000; ++r) {
        const llama_token last = (llama_token) (rng() % 50);
        const auto s = view(h, last);
        const auto span = rtb::copy_round(fl, cs, s, 32, n_src, false, 16, 8);
        n_copy += span.empty() ? 0 : 1;
        h.push_back(last);
        for (int j = 0; j < 2; ++j) {
            h.push_back((llama_token) (rng() % 50));
        }
    }
    printf("  random text: %d copy rounds of 3000, %llu shadow scores\n", n_copy, (unsigned long long) cs.n_shadow);
    assert(n_copy < 60);
}

static void test_flags_from_env() {
    unsetenv("LLAMA_RTB_COPYCTL");
    unsetenv("LLAMA_RTB_COPYCURSOR");
    unsetenv("LLAMA_RTB_EDITCOPY");
    assert(!rtb::copy_flags::from_env().any());
    setenv("LLAMA_RTB_COPYCTL", "1", 1);
    assert(rtb::copy_flags::from_env().ctl);
    setenv("LLAMA_RTB_COPYCTL", "0", 1);
    assert(!rtb::copy_flags::from_env().any());
    unsetenv("LLAMA_RTB_COPYCTL");
}


// cost-model depth: converges to the true optimum and moves down at deep context
static int true_best(double p, double a, double b, double c, double ctx, int n_max) {
    int best = 1; double br = 0;
    for (int d = 1; d <= n_max; ++d) {
        double e = 1, r = 1;
        for (int k = 0; k < d; ++k) { r *= p; e += r; }
        const double t = a + b * (d + 1) + c * (d + 1) * ctx / 1024.0;
        if (e / t > br) { br = e / t; best = d; }
    }
    return best;
}

static void test_cost_depth() {
    std::mt19937 rng(3);
    const double p = 0.8, a = 10000, b = 900, c = 25;
    for (double ctx : { 3000.0, 100000.0 }) {
        rtb::cost_depth cd;
        cd.reset(8);
        int last = 0;
        std::vector<int> picks;
        for (int r = 0; r < 3000; ++r) {
            const int d = cd.pick(ctx);
            int acc = 0;
            while (acc < d && (rng() % 1000) < p * 1000) {
                ++acc;
            }
            const double t = a + b * (d + 1) + c * (d + 1) * ctx / 1024.0;
            const double noise = 1.0 + 0.05 * ((double) (rng() % 1000) / 1000.0 - 0.5);
            cd.observe(d, acc, (int64_t) (t * noise), ctx);
            last = d;
            if (r >= 2000) picks.push_back(d);
        }
        std::sort(picks.begin(), picks.end());
        const int med = picks[picks.size() / 2];
        const int want = true_best(p, a, b, c, ctx, 8);
        printf("  cost depth ctx %.0f: median pick %d, true best %d (%s)\n", ctx, med, want, cd.summary().c_str());
        assert(std::abs(med - want) <= 1);
        (void) last;
    }
}


// the keyed Gumbel-max samples softmax(logits) and is a pure function of (seed, n, id)
static void test_hash_gumbel() {
    const double logit[5] = { 1.0, 0.2, -0.5, 2.0, 0.0 };
    double z = 0;
    for (double l : logit) z += std::exp(l);
    int cnt[5] = {};
    const int N = 400000;
    for (int n = 0; n < N; ++n) {
        int best = 0; double bv = -1e300;
        for (int i = 0; i < 5; ++i) {
            const double v = logit[i] + rtb::hash_gumbel(1234, (uint64_t) n, 100 + i);
            if (v > bv) { bv = v; best = i; }
        }
        cnt[best]++;
    }
    for (int i = 0; i < 5; ++i) {
        const double p = std::exp(logit[i]) / z, f = (double) cnt[i] / N;
        const double se = std::sqrt(p * (1 - p) / N);
        printf("  hash gumbel: token %d p=%.4f freq=%.4f\n", i, p, f);
        assert(std::fabs(f - p) < 5 * se);
    }
    assert(rtb::hash_gumbel(1, 2, 3) == rtb::hash_gumbel(1, 2, 3));
    assert(rtb::hash_gumbel(1, 2, 3) != rtb::hash_gumbel(1, 3, 3));
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("test-speculative-copy-ctl\n");
    test_flags_from_env();
    test_index_matches_reference();
    test_cursor_follows_source();
    test_ctl_choice();
    test_edit_rules();
    test_ctl_random_text_stays_off();
    test_cost_depth();
    test_hash_gumbel();
    printf("OK\n");
    return 0;
}
