#pragma once
// fork MOEGAP CPU split: the planner's greedy choice of host experts, shared by the plan kernel (moe-cache.cu) and
// the CPU test (tests/test-moe-cs-plan.cpp), plus the AIPFIX per-type cost table (GGML_MOE_CPU_SPLIT_PROBE_PER_TYPE=1).
#include <array>
#include <cmath>
#include <map>
#include <vector>

#if defined(__CUDACC__)
#define MOE_CS_HD __host__ __device__ __forceinline__
#else
#define MOE_CS_HD inline
#endif

struct moe_cs_greedy_cfg {
    float route_ns;   // host cost of one (expert, route) on this layer
    float fixed_ns;   // host cost of one job, handshake included
    float w;          // link weight x ns per byte
    float floor_ns;   // card floor (already x distinct experts)
    float bytes[3];   // copy bytes of one expert per projection
    int max_exp, max_route;
};

// Candidates in `order` (most routes first); cnt[i] / 4 = routes of candidate i; flag[k][i] & 3 == 3 = projection k
// of candidate i is a miss. `gpu` = the card's copy time with every miss copied. Moves candidates to the host while
// the slower side gets faster. take[0..ne) = chosen candidates, first[0..ne] = their route offsets. Returns ne.
MOE_CS_HD int moe_cs_greedy(const moe_cs_greedy_cfg & c, float gpu, int ncand, const short * order, const short * cnt,
        const unsigned char * const flag[3], int * take, int * first, int * nr_out, unsigned long long * parts_out,
        unsigned long long * partial_out) {
    float cpu = 0.0f;
    int ne = 0, nr = 0;
    unsigned long long parts = 0, partial = 0;
    first[0] = 0;
    for (int r = 0; r < ncand && ne < c.max_exp; ++r) {
        const int i = order[r];
        const int n = cnt[i] / 4;
        if (nr + n > c.max_route) { break; }
        float saved = 0.0f;
        int nm = 0;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            if ((flag[k][i] & 3) == 3) { saved += c.bytes[k]; ++nm; }
        }
        const float cpu2 = c.fixed_ns + (float) (nr + n) * c.route_ns;
        const float gpu2 = gpu - saved * c.w;
        const float now = fmaxf(cpu, fmaxf(c.floor_ns, gpu));
        const float next = fmaxf(cpu2, fmaxf(c.floor_ns, gpu2));
        // without a floor, ties stop (the old rule); with one, ties inside the floor go to the CPU
        if (c.floor_ns > 0.0f ? next > now : next >= now) { break; }
        cpu = cpu2;
        gpu = gpu2;
        take[ne] = i;
        ++ne;
        nr += n;
        parts += (unsigned long long) nm;
        partial += nm < 3;
        first[ne] = nr;
    }
    *nr_out = nr;
    *parts_out = parts;
    *partial_out = partial;
    return ne;
}

// AIPFIX per-type probe (host-only template; no __CUDA_ARCH__ guard: nvcc's device pass must still see the name)
// One probe per distinct host type triple (up, gate, down) among the registered layers;
// each layer gets its own triple's cost. key[l] = {-1,-1,-1} for unregistered layers (cost left at 0). `probe(l, &r, &f)`
// times layer l (false = failed: that triple falls back to the global cost). `known` carries probes across sessions.
// Returns the number of new probes run.
template <typename Probe>
int moe_cs_type_costs(const std::vector<std::array<int, 3>> & key, Probe probe, double route_global, double fixed_global,
        std::map<std::array<int, 3>, std::array<double, 2>> & known, std::vector<double> & route, std::vector<double> & fixed) {
    int probes = 0;
    route.assign(key.size(), 0.0);
    fixed.assign(key.size(), 0.0);
    for (size_t l = 0; l < key.size(); ++l) {
        if (key[l][0] < 0) { continue; }
        auto it = known.find(key[l]);
        if (it == known.end()) {
            double r = 0.0, f = 0.0;
            const bool ok = probe((int) l, &r, &f);
            ++probes;
            it = known.emplace(key[l], ok ? std::array<double, 2>{r, f} : std::array<double, 2>{route_global, fixed_global}).first;
        }
        route[l] = it->second[0];
        fixed[l] = it->second[1];
    }
    return probes;
}
