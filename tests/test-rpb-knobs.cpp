// Host checks for the small-width launch knobs. No CUDA, no GPU.
#include "rpb-knobs.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

static int g_fails = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        g_fails++;
    }
}

static void test_pick() {
    bool rejected = false;
    check(rpb_pick_env(nullptr, rpb_exl3_nwarps_allowed, rpb_exl3_nwarps_n, 4, &rejected) == 4 && !rejected,
          "unset exl3");
    rejected = false;
    check(rpb_pick_env("", rpb_exl3_nwarps_allowed, rpb_exl3_nwarps_n, 4, &rejected) == 4 && !rejected,
          "empty exl3");
    rejected = true;
    check(rpb_pick_env("4", rpb_exl3_nwarps_allowed, rpb_exl3_nwarps_n, 4, &rejected) == 4 && !rejected,
          "explicit 4 clears rejected");
    rejected = false;
    check(rpb_pick_env("2", rpb_exl3_nwarps_allowed, rpb_exl3_nwarps_n, 4, &rejected) == 2 && !rejected, "nwarps 2");
    check(rpb_pick_env("  8", rpb_exl3_nwarps_allowed, rpb_exl3_nwarps_n, 4, &rejected) == 8 && !rejected, "nwarps 8 ws");
    check(rpb_pick_env("+8", rpb_exl3_nwarps_allowed, rpb_exl3_nwarps_n, 4, &rejected) == 8 && !rejected, "nwarps +8");
    rejected = false;
    check(rpb_pick_env("3", rpb_exl3_nwarps_allowed, rpb_exl3_nwarps_n, 4, &rejected) == 4 && rejected, "reject 3");
    rejected = false;
    check(rpb_pick_env("0", rpb_exl3_nwarps_allowed, rpb_exl3_nwarps_n, 4, &rejected) == 4 && rejected, "reject 0");
    rejected = false;
    check(rpb_pick_env("1", rpb_exl3_nwarps_allowed, rpb_exl3_nwarps_n, 4, &rejected) == 4 && rejected, "reject 1");
    rejected = false;
    check(rpb_pick_env("16", rpb_exl3_nwarps_allowed, rpb_exl3_nwarps_n, 4, &rejected) == 4 && rejected, "reject 16");
    rejected = false;
    check(rpb_pick_env("-4", rpb_exl3_nwarps_allowed, rpb_exl3_nwarps_n, 4, &rejected) == 4 && rejected, "reject neg");
    rejected = false;
    check(rpb_pick_env("nope", rpb_exl3_nwarps_allowed, rpb_exl3_nwarps_n, 4, &rejected) == 4 && rejected, "reject text");
    rejected = false;
    check(rpb_pick_env("999999999999999999999", rpb_exl3_nwarps_allowed, rpb_exl3_nwarps_n, 4, &rejected) == 4 && rejected,
          "reject overflow");

    const rpb_ai4_mmv def = rpb_ai4_resolve(nullptr, nullptr);
    check(def.warps == 4 && def.minb == 6 && !def.rejected_warps && !def.rejected_minb, "ai4 unset");
    const rpb_ai4_mmv exp = rpb_ai4_resolve("4", "6");
    check(exp.warps == 4 && exp.minb == 6 && !exp.rejected_warps && !exp.rejected_minb, "ai4 explicit default");
    const rpb_ai4_mmv alt = rpb_ai4_resolve("8", "2");
    check(alt.warps == 8 && alt.minb == 2 && !alt.rejected_warps && !alt.rejected_minb, "ai4 8/2");
    const rpb_ai4_mmv bad = rpb_ai4_resolve("5", "7");
    check(bad.warps == 4 && bad.minb == 6 && bad.rejected_warps && bad.rejected_minb, "ai4 reject both");
    const rpb_ai4_mmv one = rpb_ai4_resolve("2", "3");
    check(one.warps == 2 && one.minb == 6 && !one.rejected_warps && one.rejected_minb, "ai4 minb 3 not in set");
    const rpb_ai4_mmv zero = rpb_ai4_resolve("0", "0");
    check(zero.warps == 4 && zero.minb == 6 && zero.rejected_warps && zero.rejected_minb, "ai4 zero");
}

static void test_ksplit() {
    // K = N = 5120, T = 1: ktiles cap does not bind. nwarps 4 matches the base formula.
    check(rpb_exl3_ksplit(320, 320, 128, 4) == 5, "ksplit wide nwarps 4");
    check(rpb_exl3_ksplit(320, 320, 128, 2) == 5, "ksplit wide nwarps 2");
    check(rpb_exl3_ksplit(320, 320, 128, 8) == 5, "ksplit wide nwarps 8");
    // nt = 1: the ">= 2 k-tiles per warp" cap binds and differs by nwarps.
    check(rpb_exl3_ksplit(320, 1, 128, 4) == 40, "ksplit thin nwarps 4");
    check(rpb_exl3_ksplit(320, 1, 128, 2) == 80, "ksplit thin nwarps 2");
    check(rpb_exl3_ksplit(320, 1, 128, 8) == 20, "ksplit thin nwarps 8");
    // k-tiles-per-item cap pushes ksplit up to 128 for every nwarps in the set.
    check(rpb_exl3_ksplit(4096, 512, 32, 4) == 128, "ksplit ktiles cap 4");
    check(rpb_exl3_ksplit(4096, 512, 32, 2) == 128, "ksplit ktiles cap 2");
    check(rpb_exl3_ksplit(4096, 512, 32, 8) == 128, "ksplit ktiles cap 8");
    check(rpb_exl3_xrows(1) == 1 && rpb_exl3_ktiles_max(1) == 128, "cfg T1");
    check(rpb_exl3_xrows(8) == 8 && rpb_exl3_ktiles_max(8) == 64, "cfg T8");
    check(rpb_exl3_xrows(16) == 16 && rpb_exl3_ktiles_max(16) == 32, "cfg T16");
}

static void test_ai4_ampere_range() {
    const auto def = rpb_ai4_ampere_resolve(nullptr, nullptr);
    check(!def.rejected && !def.contains(8) && def.contains(9) && def.contains(32) && !def.contains(33), "ai4 measured boundaries");
    const auto tuned = rpb_ai4_ampere_resolve("17", "48");
    check(!tuned.rejected && !tuned.contains(16) && tuned.contains(17) && tuned.contains(48) && !tuned.contains(49), "ai4 tuned boundaries");
    const auto full = rpb_ai4_ampere_resolve("1", "64");
    check(!full.rejected && full.contains(1) && full.contains(64) && !full.contains(65), "ai4 supported width limit");
    for (const char * text : {"0", "-1", "65", "abc", "9junk", "9999999999999999999999999999999999"}) {
        const auto bad_min = rpb_ai4_ampere_resolve(text, "64");
        const auto bad_max = rpb_ai4_ampere_resolve("1", text);
        check(bad_min.rejected && bad_min.min_n == 9 && bad_min.max_n == 32, "ai4 invalid minimum resets pair");
        check(bad_max.rejected && bad_max.min_n == 9 && bad_max.max_n == 32, "ai4 invalid maximum resets pair");
    }
    const auto reversed = rpb_ai4_ampere_resolve("33", "32");
    check(reversed.rejected && reversed.min_n == 9 && reversed.max_n == 32, "ai4 reversed interval");
}

static void test_exl3_resolve() {
    // Defaults of the real launcher: dec on, tmin 2, weight-major on, fragment staging on, pf2 off, sm_86.
    const auto def = rpb_exl3_resolve(4, 1, false, 4, true, 2, true, true, false, 860, true);
    check(def.nwarps == 4 && !def.use_dec && !def.ignored, "resolve default request");
    const auto t1 = rpb_exl3_resolve(8, 1, false, 4, true, 2, true, true, false, 860, true);
    check(t1.nwarps == 8 && !t1.use_dec && !t1.ignored, "T1 uses old kernel");
    const auto t1dec = rpb_exl3_resolve(8, 1, false, 4, true, 1, true, true, false, 860, true);
    check(t1dec.nwarps == 4 && t1dec.ignored, "T1 dec-on has no alt");
    const auto t4 = rpb_exl3_resolve(2, 4, false, 4, true, 2, true, true, false, 860, true);
    check(t4.nwarps == 2 && t4.use_dec && !t4.ignored, "T4 dec alt");
    const auto t4b2 = rpb_exl3_resolve(8, 4, false, 2, true, 2, true, true, false, 860, true);
    check(t4b2.nwarps == 8 && t4b2.use_dec, "2-bit sm86 weight-major");
    const auto t4cpu = rpb_exl3_resolve(8, 4, false, 2, true, 2, true, true, false, 800, true);
    check(t4cpu.nwarps == 4 && t4cpu.ignored, "2-bit non-86 stays at 4");
    const auto t4b3 = rpb_exl3_resolve(8, 4, false, 3, true, 2, true, true, false, 800, true);
    check(t4b3.nwarps == 8 && t4b3.use_dec, "3-bit weight-major any arch");
    const auto fused = rpb_exl3_resolve(8, 4, true, 4, true, 2, true, true, false, 860, true);
    check(fused.nwarps == 4 && fused.ignored, "fused stays at 4");
    const auto wide = rpb_exl3_resolve(8, 16, false, 4, true, 2, true, true, false, 860, true);
    check(wide.nwarps == 4 && wide.ignored, "T16 stays at 4");
    const auto pf2 = rpb_exl3_resolve(8, 4, false, 4, true, 2, true, true, true, 860, true);
    check(pf2.nwarps == 4 && pf2.ignored, "pf2 stays at 4");
    const auto novw = rpb_exl3_resolve(8, 4, false, 4, true, 2, true, false, false, 860, true);
    check(novw.nwarps == 4 && novw.ignored, "row-major staging stays at 4");
    const auto nowm = rpb_exl3_resolve(8, 4, false, 4, true, 2, false, true, false, 860, true);
    check(nowm.nwarps == 4 && nowm.ignored, "weight-major off stays at 4");
    const auto nodec = rpb_exl3_resolve(8, 4, false, 4, false, 2, true, true, false, 860, true);
    check(nodec.nwarps == 4 && nodec.ignored, "dec off at T4 stays at 4");
    const auto nobuild = rpb_exl3_resolve(8, 4, false, 4, true, 2, true, true, false, 860, false);
    check(nobuild.nwarps == 4 && nobuild.ignored, "weight-major compiled out");
    const auto badreq = rpb_exl3_resolve(5, 4, false, 4, true, 2, true, true, false, 860, true);
    check(badreq.nwarps == 4 && badreq.ignored, "unclamped request");
}

static void test_shapes() {
    for (int nw : {2, 4, 8}) {
        check(rpb_exl3_instance_ok(nw, 1, false, false), "old T1 smem");
        for (int T = 2; T <= 8; ++T) {
            check(rpb_exl3_instance_ok(nw, T, true, true), "dec wm smem");
            check(rpb_exl3_instance_ok(nw, T, true, false), "dec x-as-A smem");
        }
        check(rpb_exl3_instance_ok(nw, 16, true, false), "dec T16 smem");
    }
    check(!rpb_exl3_instance_ok(1, 1, false, false), "nwarps 1 excluded");
    check(!rpb_exl3_instance_ok(0, 1, false, false), "nwarps 0 excluded");
    check(!rpb_exl3_instance_ok(16, 8, true, true), "nwarps 16 excluded");

    // Sweep set, default XREG stages-2 kernel, both tile sizes. Also the kernels
    // that already ship (xsmem MINB 3 stages 3, and stages 4 at warps 4 / minb 6).
    for (int w : {2, 4, 8}) {
        for (int m : {2, 4, 6, 8}) {
            // 8*8 warps and 256*8 threads exceed sm_86 (48 warps, 1536 threads per SM).
            const bool legal = !(w == 8 && m == 8);
            for (bool k5 : {false, true}) {
                check(rpb_ai4_mmv_ok(w, m, 2, true, k5) == legal, "ai4 sweep instance");
            }
        }
    }
    check(rpb_ai4_mmv_ok(4, 3, 3, false, true), "ai4 xsmem k5");
    check(rpb_ai4_mmv_ok(4, 3, 3, false, false), "ai4 xsmem");
    check(rpb_ai4_mmv_ok(4, 6, 4, true, true), "ai4 stages 4");
    check(rpb_ai4_mmv_ok(4, 6, 3, true, false), "ai4 stages 3");
    check(!rpb_ai4_mmv_ok(1, 6, 2, true, false), "ai4 warps 1 zero red");
    check(!rpb_ai4_mmv_ok(0, 6, 2, true, false), "ai4 warps 0");
    check(!rpb_ai4_mmv_ok(8, 0, 2, true, false), "ai4 minb 0");

    // Default launch geometry the report quotes. Block threads and shared bytes.
    check(4 * 32 == 128, "exl3 default block");
    check(rpb_exl3_gemv_smem(4, 1, false) > 0, "exl3 T1 smem");
    check(rpb_exl3_gemv_dec_smem(4, 8, true) <= 48 * 1024, "exl3 T8 dec smem cap");
    check(rpb_ai4_dynamic_smem(4, 2, true, false) == 8960, "ai4 default dynamic smem");
    check(rpb_ai4_static_smem(4) == 1536, "ai4 default static smem");
    check(rpb_ai4_dynamic_smem(8, 2, true, true) + rpb_ai4_static_smem(8) <= 99 * 1024, "ai4 worst smem");
}

int main() {
    test_pick();
    test_ai4_ampere_range();
    test_ksplit();
    test_exl3_resolve();
    test_shapes();
    if (g_fails != 0) {
        std::fprintf(stderr, "test-rpb-knobs: %d failed\n", g_fails);
        return 1;
    }
    std::printf("test-rpb-knobs: ok\n");
    return 0;
}
