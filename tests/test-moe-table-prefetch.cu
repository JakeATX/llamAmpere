// Standalone CUDA op/micro test. Build/run only through the project's guarded queue.
// Intentionally no CMake integration until the diagnostic build is frozen.
#include "../ggml/src/ggml-cuda/moe-table-prefetch.cuh"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tp = moe_tp;
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error( \
    std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ") + #x); } while (0)
template<class T> struct Device {
    T * p = nullptr; size_t n;
    explicit Device(size_t count) : n(count) { CUDA(cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    Device(const Device &) = delete;
    void zero(cudaStream_t s = nullptr) { CUDA(cudaMemsetAsync(p, 0, n * sizeof(T), s)); }
    void put(const T * v, size_t count, size_t offset = 0) {
        CHECK(offset + count <= n); CUDA(cudaMemcpy(p + offset, v, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get(size_t count = 0, size_t offset = 0) const {
        if (!count) count = n; CHECK(offset + count <= n); std::vector<T> v(count);
        CUDA(cudaMemcpy(v.data(), p + offset, count * sizeof(T), cudaMemcpyDeviceToHost)); return v;
    }
};
struct Mapped {
    uint8_t * host = nullptr, * gpu = nullptr;
    explicit Mapped(size_t n) {
        CUDA(cudaHostAlloc(reinterpret_cast<void **>(&host), n, cudaHostAllocMapped));
        CUDA(cudaHostGetDevicePointer(reinterpret_cast<void **>(&gpu), host, 0));
    }
    ~Mapped() { cudaFreeHost(host); }
};
struct Streams {
    cudaStream_t main, copy;
    cudaEvent_t fork[48], tail;
    Streams() {
        int least, greatest; CUDA(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        CUDA(cudaStreamCreateWithPriority(&main, cudaStreamNonBlocking, greatest));
        CUDA(cudaStreamCreateWithPriority(&copy, cudaStreamNonBlocking, least));
        for (auto & e : fork) CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&tail, cudaEventDisableTiming));
    }
    ~Streams() {
        for (auto e : fork) cudaEventDestroy(e); cudaEventDestroy(tail);
        cudaStreamDestroy(copy); cudaStreamDestroy(main);
    }
    void branch(int i) { CUDA(cudaEventRecord(fork[i], main)); CUDA(cudaStreamWaitEvent(copy, fork[i], 0)); }
    void join() { CUDA(cudaEventRecord(tail, copy)); CUDA(cudaStreamWaitEvent(main, tail, 0)); }
};
struct Graph {
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    ~Graph() { if (exec) cudaGraphExecDestroy(exec); if (graph) cudaGraphDestroy(graph); }
    void finish(cudaStream_t s) {
        CUDA(cudaStreamEndCapture(s, &graph));
        CUDA(cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority));
    }
    void run(cudaStream_t s) { CUDA(cudaGraphLaunch(exec, s)); }
};

constexpr unsigned layers = 48;
constexpr size_t payload_stride = 512, slot_stride = 640, bank_stride = tp::slots_max * slot_stride;
constexpr uint64_t sizes[3] = {67, 131, 257}, offsets[3] = {16, 112, 272};
struct Fixture {
    Device<int32_t> ids{128};
    Device<int16_t> weights{layers * 512 * 512}, bias{512}, stub_bias{512}, assignment_slots{4 * 512};
    Device<uint8_t> hot[3] = {Device<uint8_t>(512), Device<uint8_t>(512), Device<uint8_t>(512)};
    Device<const void *> table[3] = {Device<const void *>(512), Device<const void *>(512), Device<const void *>(512)};
    Device<const void *> out{512}; Device<uint8_t> out_hot{512}, landing{4 * bank_stride};
    Device<tp::word> ready{4 * 20}, assignment{4}, progress{1}, counters{3}, expert_counts{512};
    Device<tp::Plan> plans{layers}, empty{layers};
    Device<tp::Completion> completion{layers * 20};
    Device<tp::Timing> timing{layers}; Device<tp::Stats> stats{layers};
    Mapped payload{3 * 512 * payload_stride};
    std::vector<int16_t> hw = std::vector<int16_t>(512 * 512), hb = std::vector<int16_t>(512);
    std::array<std::vector<uint8_t>, 3> hh;
    std::array<std::vector<const void *>, 3> ht;
    std::vector<int32_t> hi = std::vector<int32_t>(128);
    tp::Ring ring{};
    Fixture() {
        for (unsigned s = 0; s < 512; ++s) for (unsigned e = 0; e < 512; ++e)
            hw[s * 512 + e] = int16_t(int((s * 7 + (e / 2) * 11) % 37) - 23 + (e < 12 ? 400 : 0));
        for (unsigned e = 0; e < 512; ++e) hb[e] = int16_t(int((e / 2) % 11) - 6);
        // Same source-independent control as State: stable rank of training LUT column sums.
        std::array<int64_t,512> marginal{}; std::array<unsigned,512> order;
        std::array<int16_t,512> fixed{};
        for (unsigned e=0;e<512;++e) { order[e]=e; for (unsigned s=0;s<512;++s) marginal[e] += hw[s*512+e]; }
        std::stable_sort(order.begin(),order.end(),[&](unsigned a,unsigned b) { return marginal[a] > marginal[b]; });
        for (unsigned rank=0;rank<512;++rank) fixed[order[rank]]=int16_t(512-rank);
        stub_bias.put(fixed.data(),512);
        for (unsigned t = 0; t < 128; ++t) hi[t] = int32_t((t % 10 == 4 ? 3 : t % 10) + (t / 10 % 3));
        ids.put(hi.data(), hi.size()); bias.put(hb.data(), hb.size());
        for (unsigned l = 0; l < layers; ++l) weights.put(hw.data(), hw.size(), l * hw.size());
        for (int k = 0; k < 3; ++k) {
            hh[k].resize(512); ht[k].resize(512);
            for (unsigned e = 0; e < 512; ++e) {
                hh[k][e] = e % 14 < 2 || (e + k) % 5 == 0;
                ht[k][e] = payload.gpu + (k * 512 + e) * payload_stride;
                // Never equal guard byte 0xa5, so byte-counting detects actual writes.
                std::memset(payload.host + (k * 512 + e) * payload_stride, 1 + (e + k * 31) % 127, payload_stride);
            }
            hot[k].put(hh[k].data(), 512); table[k].put(ht[k].data(), 512);
        }
        ring = {landing.p, bank_stride, slot_stride, ready.p, assignment.p, assignment_slots.p};
        reset();
    }
    void reset() {
        ready.zero(); assignment.zero(); progress.zero(); counters.zero(); expert_counts.zero();
        plans.zero(); empty.zero(); completion.zero(); timing.zero(); stats.zero();
        CUDA(cudaMemset(assignment_slots.p, 0xff, assignment_slots.n * sizeof(int16_t)));
        CUDA(cudaMemset(landing.p, 0xa5, landing.n)); CUDA(cudaDeviceSynchronize());
    }
    tp::Dispatch dispatch(unsigned l, unsigned k, int width, int lead) {
        tp::Dispatch d{}; d.ids = ids.p; d.n_ids = 10; d.n_tokens = width; d.ids_stride = 10;
        d.layer = l; d.kind = k; d.base = table[k].p; d.hot = hot[k].p;
        d.out = out.p; d.out_hot = out_hot.p; d.counters = counters.p; d.expert_counts = expert_counts.p;
        d.ring = ring; d.progress = progress.p; d.lead = lead; d.ring_offset = offsets[k];
        d.stats = stats.p + l; d.projection_bytes = sizes[k]; return d;
    }
    tp::Planner planner(unsigned source, int lead, bool positive, unsigned limit, uint64_t budget) {
        tp::Planner p{}; const unsigned target = source + lead;
        p.weights = target < layers ? weights.p + target * 512 * 512 : nullptr; p.bias = bias.p;
        for (int k = 0; k < 3; ++k) p.projection[k] = {table[k].p, hot[k].p, sizes[k], offsets[k]};
        p.plan = plans.p + source; p.completion = completion.p + source * 20;
        p.n_layers = layers; p.cold_limit = limit; p.budget_bytes = budget;
        p.positive_only = positive; p.timing = timing.p + source; return p;
    }
};
struct Expected { unsigned e, mask; uint64_t bytes; int score; };
static std::vector<Expected> reference(const Fixture & f, int width, bool positive, unsigned limit, uint64_t budget,
                                       bool bias_only = false) {
    std::vector<Expected> ranked;
    for (unsigned e = 0; e < 512; ++e) {
        int score = width * f.hb[e]; unsigned mask = 0; uint64_t bytes = 0;
        if (!bias_only) for (int i = 0; i < width * 10; ++i) score += f.hw[f.hi[i] * 512 + e];
        for (int k = 0; k < 3; ++k) if (!f.hh[k][e]) { mask |= 1u << k; bytes += sizes[k]; }
        if (mask && (!positive || score > 0)) ranked.push_back({e, mask, bytes, score});
    }
    std::sort(ranked.begin(), ranked.end(), [](const Expected & a, const Expected & b) {
        return a.score != b.score ? a.score > b.score : a.e < b.e;
    });
    if (ranked.size() > limit) ranked.resize(limit);
    uint64_t prefix = 0; size_t n = 0;
    for (auto x : ranked) { prefix += x.bytes; if (prefix > budget) break; ++n; }
    ranked.resize(n); return ranked;
}

static void check_plan_copy(Fixture & f, Streams & s, int width, int lead, bool positive,
                            unsigned limit, uint64_t budget, unsigned source = 0, bool reset = true,
                            bool bias_only = false) {
    if (reset) f.reset();
    auto p = f.planner(source, lead, positive, limit, budget);
    if (bias_only) p.weights = nullptr;
    auto d = f.dispatch(source, 0, width, lead);
    Graph g; CUDA(cudaStreamBeginCapture(s.main, cudaStreamCaptureModeGlobal));
    tp::dispatch_kernel<true><<<1,256,0,s.main>>>(d, p); s.branch(0);
    const unsigned target = source + lead;
    tp::copy_kernel<<<32,256,0,s.copy>>>(p.plan, p.completion, f.progress.p, f.ring, p.timing, f.stats.p + target);
    s.join(); g.finish(s.main); g.run(s.main); CUDA(cudaStreamSynchronize(s.main));
    const auto expected = reference(f, width, positive, limit, budget, bias_only);
    const tp::Plan actual = f.plans.get(1, source)[0];
    CHECK(actual.count == expected.size()); CHECK(actual.bank == target % (lead + 1));
    const auto tags = f.ready.get(); const auto map = f.assignment_slots.get();
    uint64_t copied[3] = {}, useful[3] = {}, served_routes = 0;
    const auto data = f.landing.get();
    std::vector<uint8_t> expected_landing(f.landing.n, 0xa5);
    std::vector<uint8_t> need(512); for (int i = 0; i < width * 10; ++i) need[f.hi[i]] = 1;
    for (unsigned slot = 0; slot < expected.size(); ++slot) {
        const auto x = expected[slot]; const auto & a = actual.item[slot];
        CHECK(a.epoch == 1 && a.target == target && a.expert == x.e && a.mask == x.mask);
        CHECK(map[actual.bank * 512 + x.e] == int(slot));
        CHECK(tags[actual.bank * 20 + slot] == ((tp::word(1) << 32) | (tp::word(target) << 12) | (x.mask << 9) | x.e));
        for (int k = 0; k < 3; ++k) {
            CHECK(a.bytes[k] == ((x.mask >> k & 1) ? sizes[k] : 0));
            copied[k] += a.bytes[k]; if (need[x.e]) useful[k] += a.bytes[k];
            CHECK(a.src[k] == f.ht[k][x.e]);
        }
        for (size_t b = 0; b < slot_stride; ++b) {
            uint8_t value = 0xa5;
            for (int k = 0; k < 3; ++k) if ((x.mask >> k & 1) && b >= offsets[k] && b < offsets[k] + sizes[k])
                value = f.payload.host[(k * 512 + x.e) * payload_stride + b - offsets[k]];
            CHECK(data[actual.bank * bank_stride + slot * slot_stride + b] == value);
            expected_landing[actual.bank * bank_stride + slot * slot_stride + b] = value;
        }
    }
    CHECK(data == expected_landing); // Includes every unused slot/bank and guard tail.
    // Consumer is a separate graph: the producer graph never waits at a target layer.
    for (unsigned k : {1u, 0u, 2u}) {
        auto consumer = f.dispatch(target, k, width, lead);
        tp::dispatch_kernel<false><<<1,256,0,s.main>>>(consumer, tp::Planner{});
        CUDA(cudaStreamSynchronize(s.main));
        const auto ptr = f.out.get(); const auto hot = f.out_hot.get();
        for (unsigned e = 0; e < 512; ++e) {
            const auto it = std::find_if(expected.begin(), expected.end(), [e](const Expected & x) { return x.e == e; });
            const bool served = !f.hh[k][e] && it != expected.end() && (it->mask >> k & 1);
            if (served) served_routes += std::count(f.hi.begin(), f.hi.begin() + width * 10, int(e));
            const void * want = served ? f.landing.p + actual.bank * bank_stride + (it - expected.begin()) * slot_stride + offsets[k] : f.ht[k][e];
            CHECK(ptr[e] == want); CHECK(hot[e] == uint8_t(served || f.hh[k][e]));
        }
    }
    const auto stats = f.stats.get(1, target)[0];
    for (int k = 0; k < 3; ++k) { CHECK(stats.copied_bytes[k] == copied[k]); CHECK(stats.useful_bytes[k] == useful[k]); }
    CHECK(stats.served_routes == served_routes); // Duplicate routes count with their exact multiplicity.
    CHECK(stats.copy_launches == 1 && stats.observed_kernel_span_ns > 0);
    const auto timing = f.timing.get(1, source)[0]; CHECK(timing.completed_ctas == 32);
    if (reset) {
        tp::word hits = 0, misses = 0;
        for (unsigned k : {0u,1u,0u,2u}) for (int i=0;i<width*10;++i) (f.hh[k][f.hi[i]] ? hits : misses)++;
        const auto base = f.counters.get(); CHECK(base[0] == hits && base[1] == misses && base[2] == 4);
        const auto counts = f.expert_counts.get();
        for (unsigned e=0;e<512;++e) CHECK(counts[e] == 4 * tp::word(std::count(f.hi.begin(),f.hi.begin()+width*10,int(e))));
    }
}

// Original-equivalent BASE demand bookkeeping, deliberately independent of new helpers.
__global__ void baseline(tp::Dispatch d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 && d.counters) atomicAdd(d.counters + 2, 1ull);
    if (i < d.n_ids * d.n_tokens) {
        const int e = d.ids[i / d.n_ids * d.ids_stride + i % d.n_ids];
        if (unsigned(e) < 512) { atomicAdd(d.expert_counts + e, 1ull); if (d.counters) atomicAdd(d.counters + (d.hot[e] ? 0 : 1), 1ull); }
    }
}
// Test-only bounded expiry trigger: wait for first expert to finish, then cancel next.
__global__ void expire_after_first(const tp::Completion * c, tp::word * progress, unsigned target, tp::word delay_ns) {
    if (threadIdx.x == 0) {
        const auto start = tp::globaltimer_ns();
        while (tp::atomic<unsigned>(*const_cast<unsigned *>(&c->done)).load(cuda::memory_order_acquire) < 32
               && tp::globaltimer_ns() - start < 100000000ull) {}
        const auto first_done = tp::globaltimer_ns();
        while (tp::globaltimer_ns() - first_done < delay_ns) {}
        tp::atomic<tp::word>(*progress).store((1ull << 32) | target, cuda::memory_order_release);
    }
}

static void reject_invalid_tags(Fixture & f, Streams & s) {
    f.reset(); const unsigned target=2,bank=2,e=2;
    const tp::word progress=1ull<<32,assignment=(1ull<<32)|target; const int16_t slot=0;
    f.progress.put(&progress,1); f.assignment.put(&assignment,1,bank);
    f.assignment_slots.put(&slot,1,bank*512+e);
    for (const tp::word tag : {(2ull<<32)|(2ull<<12)|(1ull<<9)|e,
                              (1ull<<32)|(3ull<<12)|(1ull<<9)|e,
                              (1ull<<32)|(2ull<<12)|(1ull<<9)|3ull,
                              (1ull<<32)|(2ull<<12)|(2ull<<9)|e}) {
        f.ready.put(&tag,1,bank*20);
        tp::dispatch_kernel<false><<<1,256,0,s.main>>>(f.dispatch(target,0,1,2),tp::Planner{});
        CUDA(cudaStreamSynchronize(s.main));
        CHECK(f.out.get(1,e)[0] == f.ht[0][e]); CHECK(f.out_hot.get(1,e)[0] == 0);
    }
    CHECK(f.stats.get(1,target)[0].useful_bytes[0] == 0);
    CHECK(f.stats.get(1,target)[0].served_routes == 0);
}

static void cancellation_and_bandwidth(Fixture & f, Streams & s) {
    constexpr size_t large = 128u << 20;
    Mapped src(large); Device<uint8_t> dst(large+32);
    std::memset(src.host, 0x3c, large);
    tp::Plan plan{}; plan.count = 2; plan.bank = 0;
    for (int i = 0; i < 2; ++i) {
        auto & x = plan.item[i]; x.epoch = 1; x.target = 46; x.expert = i; x.mask = 1;
        x.src[0] = src.gpu; x.dst[0] = i ? dst.p : f.landing.p; x.bytes[0] = i ? large : 67;
    }
    auto prepare = [&](unsigned progress_layer, bool sticky) {
        f.reset(); CUDA(cudaMemset(dst.p, 0xa5, large+32)); f.plans.put(&plan, 1);
        tp::word progress = (1ull << 32) | progress_layer; f.progress.put(&progress, 1);
        tp::Timing t{~tp::word(0),0,0}; f.timing.put(&t,1);
        if (sticky) { tp::Completion c{0,1}; f.completion.put(&c,1); f.completion.put(&c,1,1); }
    };
    for (bool sticky : {false, true}) {
        prepare(sticky ? 0 : 46, sticky);
        tp::copy_kernel<<<32,256,0,s.copy>>>(f.plans.p,f.completion.p,f.progress.p,f.ring,f.timing.p,f.stats.p);
        CUDA(cudaStreamSynchronize(s.copy)); const auto st = f.stats.get(1)[0];
        CHECK(st.copied_bytes[0] == 0); CHECK(f.ready.get()[0] == 0 && f.ready.get()[1] == 0);
        const auto done = f.completion.get(2); CHECK(done[0].done == 32 && done[1].done == 32);
    }
    bool partial_covered = false;
    for (const tp::word delay : {200000ull,1000000ull,50000ull}) {
        prepare(0, false);
        Graph partial; CUDA(cudaStreamBeginCapture(s.main,cudaStreamCaptureModeGlobal)); s.branch(0);
        tp::copy_kernel<<<32,256,0,s.copy>>>(f.plans.p,f.completion.p,f.progress.p,f.ring,f.timing.p,f.stats.p);
        expire_after_first<<<1,32,0,s.main>>>(f.completion.p, f.progress.p, 46, delay);
        s.join(); partial.finish(s.main); partial.run(s.main); CUDA(cudaStreamSynchronize(s.main));
        const auto written = dst.get(large); const auto st = f.stats.get(1)[0];
        CHECK(std::all_of(written.begin(),written.end(),[](uint8_t b) { return b==0x3c || b==0xa5; }));
        const size_t changed = std::count(written.begin(), written.end(), uint8_t(0x3c));
        CHECK(st.copied_bytes[0] == changed + 67);
        const auto first = f.landing.get();
        for (size_t b=0;b<first.size();++b) CHECK(first[b] == (b<67 ? 0x3c : 0xa5));
        const auto guard = dst.get(32,large); CHECK(std::count(guard.begin(),guard.end(),uint8_t(0xa5)) == 32);
        if (changed>0 && changed<large) {
            CHECK(f.ready.get()[1] == 0); CHECK(f.completion.get(2)[1].canceled == 1);
            std::printf("canceled pull: %zu/%zu bytes written after %lluns delay; exact partial-byte counter PASS\n",changed,large,delay);
            partial_covered=true; break;
        }
        std::printf("partial coverage retry: %zu/%zu bytes after %lluns (not a partial-copy pass)\n",changed,large,delay);
    }
    if (!partial_covered) throw std::runtime_error("partial cancellation coverage FAILED: no strictly partial transfer in three bounded attempts");

    // Standalone mapped pull metric; no assertion against a hardware throughput floor.
    plan.count = 1; plan.item[0].dst[0] = dst.p; plan.item[0].bytes[0] = large;
    double ms_sum = 0;
    cudaEvent_t start, end; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
    for (int repeat = 0; repeat < 6; ++repeat) {
        prepare(0, false); CUDA(cudaEventRecord(start, s.copy));
        tp::copy_kernel<<<32,256,0,s.copy>>>(f.plans.p,f.completion.p,f.progress.p,f.ring,f.timing.p,f.stats.p);
        CUDA(cudaEventRecord(end,s.copy)); CUDA(cudaEventSynchronize(end));
        float ms; CUDA(cudaEventElapsedTime(&ms,start,end)); if (repeat) ms_sum += ms;
        CHECK(f.stats.get(1)[0].copied_bytes[0] == large);
        const auto tail = dst.get(32,large-32); CHECK(std::count(tail.begin(),tail.end(),uint8_t(0x3c)) == 32);
        const auto pad = dst.get(32,large); CHECK(std::count(pad.begin(),pad.end(),uint8_t(0xa5)) == 32);
    }
    std::printf("mapped pull: %.2f GB/s, %.3f ms/128MiB (20 GB/s diagnostic target; no speed claim)\n", large * 5 / (ms_sum * 1e6), ms_sum / 5);
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
}

static bool microgates(Fixture & f, Streams & s) {
    const char * names[] = {"B BASE-count", "S snapshot", "P prediction", "C bias+real-copy", "E full+empty-copy", "V full+real-copy"};
    std::puts("Micro lower bound: repeated fixed routes/warm LUT, tiny455B triplets/~9KB per layer, no1.2ms baseline work.");
    std::puts("Control-loop overhead only: real-copy contention requires native100K copy-stub measurement.");
    bool passed = true;
    for (int width : {1,4,8}) {
        double baseline_ms = 0;
        std::vector<tp::word> baseline_counts;
        for (int mode = 0; mode < 6; ++mode) {
            f.reset(); Graph g; CUDA(cudaStreamBeginCapture(s.main,cudaStreamCaptureModeGlobal));
            for (unsigned l = 0; l < layers; ++l) for (unsigned k = 0; k < 3; ++k) {
                auto d = f.dispatch(l,k,width,2); auto p = f.planner(l,2,false,20,20 * 455);
                if (mode == 3) { p.weights = nullptr; p.bias = f.stub_bias.p; p.positive_only = true; }
                if (!mode) baseline<<<1,128,0,s.main>>>(d); // Exact original count geometry.
                else if (!k && mode >= 2) tp::dispatch_kernel<true><<<1,256,0,s.main>>>(d,p);
                else tp::dispatch_kernel<false><<<1,256,0,s.main>>>(d,tp::Planner{});
                if (!k && mode >= 3 && l+2<layers) { // Production: 46 source branches.
                    s.branch(l); const unsigned target = l+2;
                    tp::copy_kernel<<<32,256,0,s.copy>>>(mode == 4 ? f.empty.p+l : p.plan, p.completion,
                        f.progress.p,f.ring,p.timing,f.stats.p+target);
                }
            }
            if (mode >= 3) s.join(); g.finish(s.main);
            for (int i = 0; i < 2; ++i) g.run(s.main);
            CUDA(cudaStreamSynchronize(s.main));
            cudaEvent_t start,end; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
            CUDA(cudaEventRecord(start,s.main)); for (int i=0;i<20;++i) g.run(s.main);
            CUDA(cudaEventRecord(end,s.main)); CUDA(cudaEventSynchronize(end));
            float ms; CUDA(cudaEventElapsedTime(&ms,start,end)); const double per_round = ms/20;
            if (!mode) baseline_ms = per_round;
            const auto base = f.counters.get();
            if (!mode) baseline_counts = base; else CHECK(base == baseline_counts);
            std::printf("micro width=%d %-18s %.4f ms/round deltaBASE=%+.4f (48 layers x3 projections)\n",
                        width,names[mode],per_round,per_round-baseline_ms);
            if (mode == 2 || mode == 3) {
                const bool gate = per_round-baseline_ms <= 0.1; passed = passed && gate;
                std::printf("  %s main critical-path delta <=0.1ms: %s (control-loop micro gate only)\n",mode == 2 ? "P-B" : "C-B",gate ? "PASS" : "FAIL");
            }
            if (mode >= 3) {
                const auto stats = f.stats.get(); tp::word span=0,bytes=0;
                for (const auto & x : stats) { span += x.observed_kernel_span_ns; for (int k=0;k<3;++k) bytes += x.copied_bytes[k]; }
                std::printf("  observed copy CTA-span sum %.4f ms/round; %.0f copied bytes/round (not critical-path duration)\n",
                    span / (22.0 * 1e6),bytes / 22.0);
            }
            CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
        }
    }
    return passed;
}

int main(int argc, char ** argv) {
    const char * lease = std::getenv("PARENT_HOLDS_GPU_LOCK");
    if (!lease || std::strcmp(lease,"1")) { std::fprintf(stderr,"GPU test requires card queue PARENT_HOLDS_GPU_LOCK=1\n"); return 2; }
    try {
        std::puts("Synthetic CUDA op/micro checks; explicit peak allocations ~410MiB; no end-to-end speed claim.");
        CUDA(cudaSetDeviceFlags(cudaDeviceMapHost)); Fixture f; Streams s;
        for (int width : {1,4,8}) for (int lead : {1,2,3}) for (bool positive : {false,true})
            for (unsigned limit : {15u,20u}) for (uint64_t budget : {uint64_t(1100),uint64_t(20000)})
                check_plan_copy(f,s,width,lead,positive,limit,budget);
        check_plan_copy(f,s,8,2,false,20,20000,0,true,true); // Bias-only source-independent policy.
        const auto saved_bias = f.hb;
        for (int value : {0,-7}) {
            std::fill(f.hb.begin(),f.hb.end(),int16_t(value)); f.bias.put(f.hb.data(),512);
            for (bool positive : {false,true}) check_plan_copy(f,s,8,2,positive,20,20000,0,true,true);
        }
        f.hb = saved_bias; f.bias.put(f.hb.data(),512);
        // Eight tokens repeat expert3 on all ten routes. UP/GATE are cold and
        // DOWN resident: 160 served routes, but just two unique projection copies.
        const auto saved_ids = f.hi;
        std::fill(f.hi.begin(),f.hi.end(),3); f.ids.put(f.hi.data(),f.hi.size());
        std::fill(f.hb.begin(),f.hb.end(),0); f.hb[3] = 32767; f.bias.put(f.hb.data(),512);
        check_plan_copy(f,s,8,2,true,20,20000,0,true,true);
        const auto repeated = f.stats.get(1,2)[0];
        CHECK(repeated.served_routes == 160);
        CHECK(repeated.useful_bytes[0] == sizes[0] && repeated.useful_bytes[1] == sizes[1] &&
              repeated.useful_bytes[2] == 0);
        f.hi = saved_ids; f.ids.put(f.hi.data(),f.hi.size());
        f.hb = saved_bias; f.bias.put(f.hb.data(),512);
        reject_invalid_tags(f,s);
        // Reuse target2's bank for target5 after its consumer, then reject stale target2.
        check_plan_copy(f,s,4,2,false,20,20000);
        f.stats.zero(); CUDA(cudaMemset(f.landing.p,0xa5,f.landing.n)); CUDA(cudaDeviceSynchronize());
        check_plan_copy(f,s,4,2,false,20,20000,3,false);
        auto stale = f.dispatch(2,0,4,2); tp::dispatch_kernel<false><<<1,256,0,s.main>>>(stale,tp::Planner{});
        CUDA(cudaStreamSynchronize(s.main)); const auto stale_hot = f.out_hot.get();
        CHECK(stale_hot == f.hh[0]);
        // Inactive layers produce an empty plan but retain measured copy-kernel timing.
        f.reset(); auto p = f.planner(0,2,false,20,20000); p.weights = nullptr; p.bias = nullptr;
        tp::dispatch_kernel<true><<<1,256,0,s.main>>>(f.dispatch(0,0,1,2),p); s.branch(0);
        tp::copy_kernel<<<32,256,0,s.copy>>>(p.plan,p.completion,f.progress.p,f.ring,p.timing,f.stats.p); s.join();
        CUDA(cudaStreamSynchronize(s.main)); CHECK(f.plans.get(1)[0].count == 0);
        const auto inactive_stats = f.stats.get(1)[0];
        const auto inactive_timing = f.timing.get(1)[0];
        std::printf("inactive copy timing: launches=%llu span=%llu first=%llu last=%llu completed=%u\n",
            inactive_stats.copy_launches, inactive_stats.observed_kernel_span_ns,
            inactive_timing.first_cta_ns, inactive_timing.last_cta_ns, inactive_timing.completed_ctas);
        CHECK(inactive_stats.copy_launches == 1);
        CHECK(inactive_timing.completed_ctas == 32 && inactive_timing.first_cta_ns != ~tp::word(0));
        CHECK(inactive_timing.last_cta_ns >= inactive_timing.first_cta_ns);
        CHECK(inactive_stats.observed_kernel_span_ns == inactive_timing.last_cta_ns - inactive_timing.first_cta_ns);
        cancellation_and_bandwidth(f,s);
        CUDA(cudaGetLastError()); std::puts("moe-table-prefetch correctness checks PASS");
        if (argc == 1 || std::strcmp(argv[1],"--no-micro")) {
            const bool passed = microgates(f,s); CUDA(cudaGetLastError());
            std::puts(passed ? "moe-table-prefetch control-loop micro gates PASS" : "moe-table-prefetch control-loop performance FAIL");
            if (!passed) return 3;
        }
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
