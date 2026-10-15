#include "nmse-masked-inf.h"

#include <cmath>
#include <cstdio>
#include <limits>

static int g_fail = 0;

static void report(const char * name, double got, bool ok) {
    printf("%s: score=%.9g %s\n", name, got, ok ? "ok" : "FAIL");
    if (!ok) {
        g_fail++;
    }
}

// same comparison the backend-ops harness uses. NaN is not a pass.
static bool passes_1e6(double s) {
    return std::isfinite(s) && !(s > 1e-6);
}

static bool fails_1e6(double s) {
    return s > 1e-6;
}

static bool pos_inf(double s) {
    return std::isinf(s) && s > 0.0;
}

// what the old nmse() does. -inf makes this NaN, and NaN > 1e-6 is false.
static double plain_nmse(const float * a, const float * b, size_t n) {
    double sse = 0.0;
    double saa = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double da = (double) a[i];
        const double db = (double) b[i];
        const double d  = da - db;
        sse += d*d;
        saa += da*da;
    }
    return sse/saa;
}

template <size_t N>
static double score(const float (&a)[N], const float (&b)[N]) {
    return nmse_masked_inf(a, b, N);
}

int main() {
    const float ninf = -std::numeric_limits<float>::infinity();
    const float pinf =  std::numeric_limits<float>::infinity();
    const float nan  =  std::numeric_limits<float>::quiet_NaN();
    const float nmax = -std::numeric_limits<float>::max();

    {
        const float a[] = { 1.0f, -2.0f, ninf, 3.5f, ninf, 0.25f };
        const float b[] = { 1.0f, -2.0f, ninf, 3.5f, ninf, 0.25f };
        const double s = score(a, b);
        const double plain = plain_nmse(a, b, 6);
        const bool nan_would_pass = !(plain > 1e-6);
        report("identical with matching -inf", s, s == 0.0 && passes_1e6(s) && std::isnan(plain) && nan_would_pass);
    }

    {
        const float a[] = { 1.0f, -2.5f, 4.0f, ninf, 8.0f, ninf };
        float b[6];
        for (int i = 0; i < 6; i++) {
            if (a[i] == ninf) {
                b[i] = ninf;
                continue;
            }
            // 1e-7 is below one float32 ulp (~1.2e-7). If the multiply rounds
            // away, step one ulp so the case is a real perturbation.
            b[i] = (float) ((double) a[i] * (1.0 + 1e-7));
            if (b[i] == a[i]) {
                b[i] = std::nextafter(a[i], a[i] > 0.0f ? pinf : ninf);
            }
        }
        const double s = score(a, b);
        report("1e-7 relative noise, matching -inf", s, s > 0.0 && passes_1e6(s));
    }

    {
        const float a[] = { 1.0f, 2.0f, 4.0f, ninf, -3.0f, ninf };
        float b[]       = { 1.0f, 2.0f, 4.0f * 1.1f, ninf, -3.0f, ninf };
        const double s = score(a, b);
        const double plain = plain_nmse(a, b, 6);
        report("one finite value +10%, matching -inf", s,
               fails_1e6(s) && std::isfinite(s) && std::isnan(plain) && !(plain > 1e-6));
    }

    {
        const float a[] = { 1.0f, ninf, 2.0f, 3.0f };
        const float b[] = { 1.0f, 2.0f, ninf, 3.0f };
        report("-inf in a different position", score(a, b), pos_inf(score(a, b)));
    }

    {
        const float a[] = { 1.0f, 2.0f, ninf };
        const float b[] = { 1.0f, nan,  ninf };
        report("NaN in b", score(a, b), pos_inf(score(a, b)) && fails_1e6(score(a, b)));
    }

    {
        const float a[] = { nan,  2.0f, ninf };
        const float b[] = { 1.0f, 2.0f, ninf };
        report("NaN in a", score(a, b), pos_inf(score(a, b)) && fails_1e6(score(a, b)));
    }

    {
        const float a[] = { ninf, ninf, ninf };
        const float b[] = { ninf, ninf, ninf };
        const double s = score(a, b);
        report("all -inf matching", s, s == 0.0 && passes_1e6(s));
    }

    {
        const float a[] = { ninf, 1.0f, ninf };
        const float b[] = { -1e30f, 1.0f, ninf };
        report("-inf vs -1e30", score(a, b), pos_inf(score(a, b)));
    }

    {
        const float a[] = { ninf, 1.0f };
        const float b[] = { nmax, 1.0f };
        report("-inf vs -FLT_MAX", score(a, b), pos_inf(score(a, b)));
    }

    {
        const float a[] = { -0.0f, 1.0f, ninf, 0.0f };
        const float b[] = {  0.0f, 1.0f, ninf, -0.0f };
        const double s = score(a, b);
        report("-0.0 vs 0.0, matching -inf", s, s == 0.0 && passes_1e6(s));
    }

    {
        const float a[] = { nan, nan, 1.0f };
        const float b[] = { nan, 2.0f, 1.0f };
        report("both NaN", score(a, b), pos_inf(score(a, b)));
    }

    {
        const double s = nmse_masked_inf(nullptr, nullptr, 0);
        report("n == 0", s, s == 0.0 && passes_1e6(s));
    }

    {
        const float a[] = { pinf, 1.0f, ninf };
        const float b[] = { pinf, 2.0f, ninf };
        report("+inf vs +inf", score(a, b), pos_inf(score(a, b)));
    }

    {
        const float a[] = { pinf, 1.0f };
        const float b[] = { 1.0f, 1.0f };
        report("+inf vs finite", score(a, b), pos_inf(score(a, b)));
    }

    {
        // (1e20)^2 overflows float32. Equal values must still score 0, not NaN.
        const float a[] = { 1e20f, -1e20f, ninf, 1e20f };
        const float b[] = { 1e20f, -1e20f, ninf, 1e20f };
        const double s = score(a, b);
        report("large equal floats (float square overflows)", s, s == 0.0 && passes_1e6(s));
    }

    {
        // float32 (1e20 - -1e20)^2 and (1e20)^2 are both +inf, so inf/inf is NaN
        // and the threshold would pass. The double score must be finite and fail.
        const float a[] = { 1e20f,  1e20f, ninf };
        const float b[] = { 1e20f, -1e20f, ninf };
        const double s = score(a, b);
        report("large sign flip (float ratio would be NaN)", s, fails_1e6(s) && std::isfinite(s));
    }

    {
        const float a[] = { 0.0f, -0.0f, ninf };
        const float b[] = { 0.0f,  0.0f, ninf };
        const double s = score(a, b);
        report("all-zero finite part, matching -inf", s, s == 0.0 && passes_1e6(s));
    }

    {
        const float a[] = { 0.0f, ninf, 0.0f };
        const float b[] = { 0.0f, ninf, 1.0f };
        const double s = score(a, b);
        report("zero reference vs 1, matching -inf", s, s == 1.0 && fails_1e6(s));
    }

    if (g_fail != 0) {
        printf("%d case(s) failed\n", g_fail);
        return 1;
    }
    printf("all cases passed\n");
    return 0;
}
