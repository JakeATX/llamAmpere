#include "llama-moe-cache-policy.h"

#include "llama-impl.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// fork (INKCACHEUX): see llama-moe-cache-policy.h

static const char * env_str(const char * name) {
    const char * v = getenv(name);
    return v && v[0] ? v : nullptr;
}

bool llama_moe_cache_parse_layers(const char * text, int32_t n_layer, std::vector<uint8_t> & mask, std::string & err) {
    mask.assign(std::max(n_layer, 0), 0);
    const char * p = text;
    bool any = false;
    while (*p) {
        char * end = nullptr;
        const long a = strtol(p, &end, 10);
        if (end == p) {
            err = std::string("bad layer list: ") + text;
            return false;
        }
        long b = a;
        p = end;
        if (*p == '-') {
            ++p;
            b = strtol(p, &end, 10);
            if (end == p) {
                err = std::string("bad layer range: ") + text;
                return false;
            }
            p = end;
        }
        if (a < 0 || b < a || b >= n_layer) {
            err = std::string("layer out of range (0..") + std::to_string(n_layer - 1) + "): " + text;
            return false;
        }
        for (long i = a; i <= b; ++i) {
            mask[i] = 1;
        }
        any = true;
        if (*p == ',') {
            ++p;
        } else if (*p) {
            err = std::string("bad layer list: ") + text;
            return false;
        }
    }
    if (!any) {
        err = "empty layer list";
        return false;
    }
    return true;
}

bool llama_moe_cache_parse_mib_list(const char * text, std::vector<double> & mib) {
    mib.clear();
    const char * p = text;
    while (*p) {
        char * end = nullptr;
        const double v = strtod(p, &end);
        if (end == p || v < 0.0) {
            return false;
        }
        mib.push_back(v);
        p = end;
        if (*p == ',') {
            ++p;
        } else if (*p) {
            return false;
        }
    }
    return !mib.empty();
}

void llama_moe_cache_controls::from_env(int32_t n_layer) {
    if (const char * v = env_str("LLAMA_MOE_CACHE_DEV_MIB")) {
        if (!llama_moe_cache_parse_mib_list(v, dev_mib)) {
            LLAMA_LOG_WARN("%s: ignoring LLAMA_MOE_CACHE_DEV_MIB=%s (want a MiB list like 4096,2048)\n", __func__, v);
            dev_mib.clear();
        }
    }
    if (const char * v = env_str("LLAMA_MOE_CACHE_SLOTS")) {
        slots = std::max(0, atoi(v));
        if (slots > 0 && !dev_mib.empty()) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_SLOTS and LLAMA_MOE_CACHE_DEV_MIB are both set, using the slot count\n", __func__);
            dev_mib.clear();
        }
    }
    if (const char * v = env_str("LLAMA_MOE_CACHE_LAYERS")) {
        std::string err;
        if (!llama_moe_cache_parse_layers(v, n_layer, layers, err)) {
            LLAMA_LOG_WARN("%s: ignoring LLAMA_MOE_CACHE_LAYERS: %s\n", __func__, err.c_str());
            layers.clear();
        }
    }
    if (const char * v = env_str("LLAMA_MOE_CACHE_REQ_STATS")) {
        req_stats = atoi(v) != 0;
    }
    if (const char * v = env_str("LLAMA_MOE_CACHE_BOUNDARY")) {
        boundary = atoi(v) != 0;
    }
}

double llama_moe_cache_controls::dev_budget_mib(size_t id) const {
    if (dev_mib.empty()) {
        return -1.0;
    }
    if (dev_mib.size() == 1) {
        return dev_mib[0];
    }
    return id < dev_mib.size() ? dev_mib[id] : 0.0;
}

//
// GS4 heat profile
//

// file, little endian: "MCH1", u32 n_layer, u32 n_expert, u64 fingerprint, u64 rows (all sessions so far),
// then f32 rate[n_layer*n_expert] (decode rows routed to the key / decode rows of its layer)

void llama_moe_cache_fp::add_tensor(int32_t il, const ggml_tensor * t) {
    add(&il, sizeof(il));
    add(t->name, strnlen(t->name, GGML_MAX_NAME));
    const int32_t type = (int32_t) t->type;
    add(&type, sizeof(type));
    add(t->ne, sizeof(t->ne));
}

void llama_moe_cache_heat::init(int32_t n_layer, int32_t n_expert, uint64_t fingerprint) {
    const char * p = env_str("LLAMA_MOE_CACHE_HEAT");
    if (!p) {
        return;
    }
    path = p;
    this->n_layer     = n_layer;
    this->n_expert    = n_expert;
    this->fingerprint = fingerprint;
    if (const char * w = env_str("LLAMA_MOE_CACHE_HEAT_WEIGHT")) {
        weight = std::clamp(atof(w), 0.0, 1.0);
    }
    if (const char * s = env_str("LLAMA_MOE_CACHE_HEAT_SAVE_S")) {
        save_s = std::max(0, atoi(s));
    }
    uses.assign((size_t) n_layer*n_expert, 0);
    rows.assign(n_layer, 0);
    t_save = std::chrono::steady_clock::now();
}

bool llama_moe_cache_heat::load() {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        LLAMA_LOG_WARN("%s: [HEAT] no profile at %s yet, starting cold\n", __func__, path.c_str());
        return false;
    }
    char magic[4];
    uint32_t nl = 0, ne = 0;
    uint64_t fp = 0, nrows = 0;
    bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "MCH1", 4) == 0 &&
        fread(&nl, sizeof(nl), 1, f) == 1 && fread(&ne, sizeof(ne), 1, f) == 1 &&
        fread(&fp, sizeof(fp), 1, f) == 1 && fread(&nrows, sizeof(nrows), 1, f) == 1;
    ok = ok && (int32_t) nl == n_layer && (int32_t) ne == n_expert && fp == fingerprint;
    std::vector<float> r;
    if (ok) {
        r.resize((size_t) n_layer*n_expert);
        ok = fread(r.data(), sizeof(float), r.size(), f) == r.size();
        for (float & x : r) {
            if (!std::isfinite(x) || x < 0.0f) {
                x = 0.0f;
            }
        }
    }
    fclose(f);
    if (!ok) {
        LLAMA_LOG_WARN("%s: [HEAT] %s is not a profile of this model and cache layout, ignored (it is replaced at the next save)\n",
                __func__, path.c_str());
        return false;
    }
    rate_old = std::move(r);
    LLAMA_LOG_WARN("%s: [HEAT] loaded %s (%llu decode rows)\n", __func__, path.c_str(), (unsigned long long) nrows);
    return true;
}

void llama_moe_cache_heat::observe(int32_t il, const int32_t * ids, size_t n_ids) {
    uint64_t * u = uses.data() + (size_t) il*n_expert;
    for (size_t i = 0; i < n_ids; ++i) {
        if (ids[i] >= 0 && ids[i] < n_expert) {
            u[ids[i]]++;
        }
    }
}

std::vector<float> llama_moe_cache_heat::rates() const {
    std::vector<float> res((size_t) n_layer*n_expert, 0.0f);
    for (int32_t il = 0; il < n_layer; ++il) {
        const bool have_new = rows[il] > 0;
        for (int32_t e = 0; e < n_expert; ++e) {
            const size_t key = (size_t) il*n_expert + e;
            const double r_new = have_new ? (double) uses[key]/rows[il] : 0.0;
            const double r_old = rate_old.empty() ? 0.0 : rate_old[key];
            double r;
            if (!have_new) {
                r = r_old;
            } else if (rate_old.empty()) {
                r = r_new;
            } else {
                r = (1.0 - weight)*r_old + weight*r_new;
            }
            res[key] = (float) r;
        }
    }
    return res;
}

bool llama_moe_cache_heat::save() {
    uint64_t total = 0;
    for (uint64_t r : rows) {
        total += r;
    }
    if (total == 0 && rate_old.empty()) {
        return false; // nothing learned
    }
    const std::vector<float> r = rates();
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) {
        LLAMA_LOG_WARN("%s: [HEAT] cannot write %s\n", __func__, tmp.c_str());
        return false;
    }
    const uint32_t nl = (uint32_t) n_layer, ne = (uint32_t) n_expert;
    bool ok = fwrite("MCH1", 1, 4, f) == 4 && fwrite(&nl, sizeof(nl), 1, f) == 1 && fwrite(&ne, sizeof(ne), 1, f) == 1 &&
        fwrite(&fingerprint, sizeof(fingerprint), 1, f) == 1 && fwrite(&total, sizeof(total), 1, f) == 1 &&
        fwrite(r.data(), sizeof(float), r.size(), f) == r.size();
    ok = fclose(f) == 0 && ok;
    if (ok) {
        std::remove(path.c_str()); // Windows rename does not replace
        ok = std::rename(tmp.c_str(), path.c_str()) == 0;
    }
    if (!ok) {
        LLAMA_LOG_WARN("%s: [HEAT] failed to save %s\n", __func__, path.c_str());
        return false;
    }
    rows_saved = total;
    t_save = std::chrono::steady_clock::now();
    LLAMA_LOG_WARN("%s: [HEAT] saved %s (%llu decode rows this session)\n", __func__, path.c_str(), (unsigned long long) total);
    return true;
}

void llama_moe_cache_heat::maybe_save() {
    uint64_t total = 0;
    for (uint64_t r : rows) {
        total += r;
    }
    const auto age = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t_save).count();
    if (total > rows_saved && age >= save_s) {
        save();
    }
}

//
// #249 + GS9 miss split
//

void llama_moe_split_policy::ewls::add(const double * x, double y) {
    constexpr double lambda = 0.995;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            A[i][j] = lambda*A[i][j] + x[i]*x[j];
        }
        b[i] = lambda*b[i] + x[i]*y;
    }
    n++;
}

// nonnegative coefficients; the CPU branch keeps its contention term d (theta[2]) at or below c (theta[1])
void llama_moe_split_policy::ewls::solve(bool cpu_branch) {
    double M[3][4];
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            M[i][j] = A[i][j];
        }
        M[i][k] = b[i];
    }
    for (int i = 1; i < k; ++i) {
        M[i][i] += 1e-3*A[0][0]; // slope regularization, relative to the intercept weight
    }
    for (int c = 0; c < k; ++c) {
        int p = c;
        for (int r = c + 1; r < k; ++r) {
            if (std::fabs(M[r][c]) > std::fabs(M[p][c])) {
                p = r;
            }
        }
        if (std::fabs(M[p][c]) < 1e-12) {
            return; // singular: keep the previous fit
        }
        for (int j = 0; j <= k; ++j) {
            std::swap(M[c][j], M[p][j]);
        }
        for (int r = 0; r < k; ++r) {
            if (r != c) {
                const double f = M[r][c]/M[c][c];
                for (int j = c; j <= k; ++j) {
                    M[r][j] -= f*M[c][j];
                }
            }
        }
    }
    for (int i = 0; i < k; ++i) {
        theta[i] = std::max(0.0, M[i][k]/M[i][i]);
    }
    if (cpu_branch && k == 3) {
        theta[2] = std::min(theta[2], theta[1]);
    }
}

void llama_moe_split_policy::init(float cpu_share) {
    cpu.k = 3;
    gpu.k = 2;
    share_q_cfg = (uint32_t) std::lround(std::clamp(cpu_share, 0.0f, 1.0f)*65536.0f);
    const char * m = env_str("LLAMA_MOE_CACHE_SPLIT");
    const char * q = env_str("LLAMA_MOE_CACHE_SPLIT_Q256");
    if (!m && !q) {
        return;
    }
    if (cpu_share <= 0.0f) {
        LLAMA_LOG_WARN("%s: [SPLIT] LLAMA_MOE_CACHE_SPLIT needs --moe-cache-cpu-share > 0 (it builds the CPU part), off\n", __func__);
        return;
    }
    if (q) {
        mode = FIXED;
        frac = std::clamp(atoi(q), 0, 256)/256.0;
    } else if (strcmp(m, "minimax") == 0 || strcmp(m, "1") == 0) {
        mode = MINIMAX;
    } else if (strcmp(m, "sched") == 0 || strcmp(m, "2") == 0) {
        mode = SCHED;
        frac = std::clamp(1.0 - cpu_share, 0.10, 0.60);
    } else {
        LLAMA_LOG_WARN("%s: [SPLIT] unknown LLAMA_MOE_CACHE_SPLIT=%s (minimax|sched), off\n", __func__, m);
        return;
    }
    frac_hist.push_back(frac);
    static const char * names[] = { "off", "minimax", "sched", "fixed" };
    LLAMA_LOG_WARN("%s: [SPLIT] miss split policy %s, PCIe fraction %.3f, fallback CPU share %.3f\n", __func__,
            names[mode], mode == MINIMAX ? 0.0 : frac, cpu_share);
}

void llama_moe_split_policy::on_ids(int32_t il, int64_t n_rows) {
    const auto t = std::chrono::steady_clock::now();
    if (il <= prev_il) {
        window_end();
    }
    if (have_pending && il > prev_il && n_rows == prev_rows) {
        const double dt = std::chrono::duration<double, std::milli>(t - t_prev).count();
        const size_t n = pend_n, m = pend_m, kc = n - m;
        if (dt > 0.0 && dt < 1000.0) {
            n_obs++;
            const double xc[3] = { 1.0, (double) kc, (double) m };
            const double xg[2] = { 1.0, (double) m };
            bool to_cpu;
            if (kc == 0) {
                to_cpu = false;
            } else if (m == 0) {
                to_cpu = true;
            } else if (ready()) {
                to_cpu = cpu.predict(xc) >= gpu.predict(xg); // the branch that bounds this layer
            } else {
                to_cpu = kc >= m;
            }
            if (to_cpu) {
                cpu.add(xc, dt);
                cpu.solve(true);
            } else {
                gpu.add(xg, dt);
                gpu.solve(false);
                n_gpu_pcie += m > 0 ? 1 : 0;
            }
        } else {
            n_drop++;
        }
    }
    have_pending = false;
    prev_il   = il;
    prev_rows = n_rows;
    t_prev    = t;
}

size_t llama_moe_split_policy::n_cpu(size_t n_miss, uint32_t & carry) {
    n_lookups++;
    auto by_share = [&](uint32_t share_q) {
        const uint64_t acc = (uint64_t) n_miss*share_q + carry;
        carry = acc & 0xffff;
        return (size_t) (acc >> 16);
    };
    if (n_miss == 0) {
        return 0;
    }
    if (mode == SCHED || mode == FIXED) {
        return by_share((uint32_t) std::lround((1.0 - frac)*65536.0));
    }
    if (!ready()) {
        return by_share(share_q_cfg);
    }
    // m* = argmin over 0 <= m <= n of max(CPU cost, GPU cost); no CPU work costs the CPU nothing
    size_t best_m = n_miss;
    double best = 1e300;
    for (size_t m = 0; m <= n_miss; ++m) {
        const size_t kc = n_miss - m;
        const double xc[3] = { 1.0, (double) kc, (double) m };
        const double xg[2] = { 1.0, (double) m };
        const double c = kc > 0 ? cpu.predict(xc) : 0.0;
        const double g = gpu.predict(xg);
        const double v = std::max(c, g);
        if (v < best - 1e-6) {
            best   = v;
            best_m = m;
        }
    }
    // keep observing the alternatives: every 64th lookup steps one expert off an all-CPU or all-PCIe choice
    if (n_lookups % 64 == 0 && n_miss > 1) {
        if (best_m == 0) {
            best_m = 1;
            n_explore++;
        } else if (best_m == n_miss) {
            best_m = n_miss - 1;
            n_explore++;
        }
    }
    return n_miss - best_m;
}

void llama_moe_split_policy::record(size_t n_cpu, size_t n_pcie) {
    have_pending = true;
    pend_n = n_cpu + n_pcie;
    pend_m = n_pcie;
}

void llama_moe_split_policy::window_end() {
    if (prev_il < 0) {
        return;
    }
    have_pending = false;
    prev_il = -1;
    if (prev_rows > 8) {
        return; // only all-generation windows count
    }
    n_windows++;
    if (mode != SCHED || n_updates >= 3) {
        return;
    }
    const int64_t at[3] = { 48, 144, 240 }; // after 48, 96 and 96 windows
    if (n_windows != at[n_updates]) {
        return;
    }
    n_updates++;
    if (!ready() || cpu.theta[1] <= 0.0 || gpu.theta[1] <= 0.0) {
        return;
    }
    // PCIe share from the link rate (1/g experts per ms) against the CPU rate (1/c), CPU weighted by 1.85
    const double c = cpu.theta[1], g = gpu.theta[1];
    double f = c/(c + 1.85*g);
    f = std::round(f/0.05)*0.05;
    frac = std::clamp(f, 0.10, 0.60);
    frac_hist.push_back(frac);
    LLAMA_LOG_WARN("llama_moe_cache: [SPLIT] update %d after %lld windows: c %.4f ms, g %.4f ms per expert -> PCIe fraction %.2f\n",
            n_updates, (long long) n_windows, c, g, frac);
}

void llama_moe_split_policy::log_summary() const {
    if (mode == OFF) {
        return;
    }
    std::string hist;
    for (double f : frac_hist) {
        hist += (hist.empty() ? "" : " -> ") + std::to_string(f).substr(0, 5);
    }
    LLAMA_LOG_WARN("llama_moe_cache: [SPLIT] %lld lookups, %lld windows, %lld samples (%lld dropped), %lld explore; "
            "CPU %.4f + %.4f*cpu + %.4f*pcie ms (n %lld), GPU %.4f + %.4f*pcie ms (n %lld, %lld with PCIe)%s%s\n",
            (long long) n_lookups, (long long) n_windows, (long long) n_obs, (long long) n_drop, (long long) n_explore,
            cpu.theta[0], cpu.theta[1], cpu.theta[2], (long long) cpu.n, gpu.theta[0], gpu.theta[1], (long long) gpu.n,
            (long long) n_gpu_pcie, mode == MINIMAX ? "" : "; fraction ", mode == MINIMAX ? "" : hist.c_str());
}
