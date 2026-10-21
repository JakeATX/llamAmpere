#pragma once

// Experimental, intentionally unincluded. Storage must outlive the graph-tail join.
// One main stream, one FIFO copy stream; fork after plan, join only at graph tail.
// Caller validates: 1<=lead<=3, 0<cold_limit<=20, 0<n_ids*n_tokens<=128,
// n_tokens<=128, int16 artifact coefficients in [-32767,32767] (not -32768),
// layers<65536, nonzero projection bytes, valid GPU aliases and sufficient ring strides.
// Initialize progress/assignment/ready to zero. Reject unsupported concurrent scopes;
// reset all state before epoch wraps. Only the target graph's source UP0 starts epochs.
#if defined(__CUDACC__) && !defined(__HIPCC__)
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

namespace moe_tp {
constexpr int experts = 512, routes_max = 128, slots_max = 20;
constexpr int threads = 256, copy_blocks = 32;
constexpr uint64_t copy_chunk = 1u << 20; // Aggregate over all CTAs, not per CTA.
using word = unsigned long long;
template<class T> using atomic = cuda::atomic_ref<T, cuda::thread_scope_device>;

struct Projection {
    const void * const * mapped; // GPU aliases only; never CPU copy_source pointers.
    const uint8_t * hot;
    uint64_t bytes, ring_offset;
};
struct Ring {
    uint8_t * data;
    uint64_t bank_stride, slot_stride;
    word * ready;               // [banks, slots_max], independent of assignment.
    word * assignment;          // [banks], epoch:32 | target:32.
    int16_t * expert_slot;      // [banks, experts], -1 means no assignment.
};
struct Descriptor {
    uint32_t epoch, target, expert, mask;
    const uint8_t * src[3];
    uint8_t * dst[3];
    uint64_t bytes[3];
};
struct Plan { uint32_t count, bank; Descriptor item[slots_max]; };
struct Completion { unsigned done, canceled; };
// Aggregated per target layer; useful bytes are distinct per layer/kind dispatch.
// Caller issues one such dispatch per layer/kind/epoch, resets only after graph join.
// copied_bytes includes all completed writes, even if their plan is later canceled.
// served_routes counts actual cold routes satisfied, including repeated expert IDs.
struct Stats { word useful_bytes[3], copied_bytes[3], observed_kernel_span_ns, copy_launches, served_routes; };
static_assert(sizeof(Stats) == 72 && alignof(Stats) == 8, "Stats ABI");
// Unique per copy launch, initialized by its planner, including empty/inactive plans.
// Span is first CTA sample to last CTA sample, not CE busy or CUDA event duration;
// it excludes final timing-reduction instructions and never counts inter-launch waits.
struct Timing { word first_cta_ns, last_cta_ns; unsigned completed_ctas; };
struct Dispatch {
    const int32_t * ids;
    int n_ids, n_tokens, ids_stride; // Total actual IDs <= routes_max.
    unsigned layer, kind;           // kind: UP=0, GATE=1, DOWN=2.
    const void * const * base;
    const uint8_t * hot;
    const void ** out;
    uint8_t * out_hot;
    word * counters;                // Original BASE hot, cold, dispatch count.
    word * expert_counts;
    Ring ring;
    word * progress;                // epoch:32 | current layer:32; initially zero.
    int lead;
    uint64_t ring_offset;
    Stats * stats;                  // Optional; BASE counters remain unchanged.
    uint64_t projection_bytes;
};
struct Planner {
    const int16_t * weights;         // This target's [source expert, target expert].
    const int16_t * bias;            // Added once per source token.
    Projection projection[3];
    Plan * plan;                    // Unique per source UP, immutable after launch.
    Completion * completion;        // [slots_max], also unique per source UP.
    unsigned n_layers, cold_limit;   // cold_limit <= slots_max; lead >= 1.
    uint64_t budget_bytes;
    bool positive_only;             // raw/conditional true, signed NB false.
    Timing * timing;                // Optional; pass the same pointer to copy_kernel.
};
using Sort = cub::BlockRadixSort<uint64_t, threads, 2>;
struct Scratch {
    typename Sort::TempStorage sort;
    uint64_t ranked[experts];
    int32_t ids[routes_max];
    unsigned needed[experts / 32];
    unsigned epoch;
};

__device__ inline word epoch_target(unsigned epoch, unsigned target) {
    return (word(epoch) << 32) | target;
}
// Target < 65536, expert < 512, mask < 8; epoch zero is reserved invalid.
__device__ inline word ready_tag(unsigned epoch, unsigned target, unsigned e, unsigned mask) {
    return (word(epoch) << 32) | (word(target) << 12) | (word(mask) << 9) | e;
}
__device__ inline bool expired(const word * progress, unsigned epoch, unsigned target) {
    const word p = atomic<word>(*const_cast<word *>(progress)).load(cuda::memory_order_acquire);
    return unsigned(p >> 32) != epoch || unsigned(p) >= target;
}

// Call with exactly one 256-thread CTA. Snapshots are immutable until dispatch ends.
// Every projection advances layer progress; kind ordering need not be UP first.
__device__ inline void count_snapshot(const Dispatch & d, Scratch & s) {
    const unsigned tid = threadIdx.x;
    if (d.stats && tid < experts / 32) s.needed[tid] = 0;
    if (tid == 0) {
        assert(blockDim.x == threads && gridDim.x == 1);
        assert(d.n_ids > 0 && d.n_tokens > 0 && d.n_tokens <= routes_max);
        assert(d.n_ids <= routes_max / d.n_tokens && d.ids_stride >= d.n_ids);
        assert(d.lead >= 1 && d.lead <= 3 && d.layer < 65536 && d.kind < 3);
        assert(!d.stats || d.projection_bytes > 0);
        word p = atomic<word>(*d.progress).load(cuda::memory_order_relaxed);
        unsigned epoch = unsigned(p >> 32);
        const bool new_epoch = d.layer == 0 && d.kind == 0;
        if (new_epoch) { ++epoch; assert(epoch); } // Wrap requires reinitialization.
        // A next-scope GATE0 before UP0 must not revive old-epoch late work.
        const unsigned layer = new_epoch || d.layer > unsigned(p) ? d.layer : unsigned(p);
        atomic<word>(*d.progress).store(epoch_target(epoch, layer), cuda::memory_order_release);
        s.epoch = epoch;
        if (d.counters) atomicAdd(d.counters + 2, word(1));
    }
    __syncthreads();
    const int n = d.n_ids * d.n_tokens;
    for (int i = tid; i < n; i += threads) {
        const int e = d.ids[(i / d.n_ids) * d.ids_stride + i % d.n_ids];
        if (i < routes_max) s.ids[i] = e;
        if (unsigned(e) < experts) {
            if (d.stats) atomicOr(s.needed + e / 32, 1u << (e % 32));
            if (d.expert_counts) atomicAdd(d.expert_counts + e, word(1));
            if (d.counters) atomicAdd(d.counters + (d.hot[e] ? 0 : 1), word(1));
        }
    }
    if (d.stats) __syncthreads(); // Complete the route-presence bitset before snapshot.
    const int bank = d.layer % (d.lead + 1);
    const word assignment = atomic<word>(d.ring.assignment[bank]).load(cuda::memory_order_acquire);
    for (int e = tid; e < experts; e += threads) {
        const void * ptr = d.base[e];
        uint8_t hot = d.hot[e];
        bool served = false;
        if (!hot && assignment == epoch_target(s.epoch, d.layer) && s.epoch) {
            const int slot = d.ring.expert_slot[bank * experts + e];
            if (unsigned(slot) < slots_max) {
                const word tag = atomic<word>(d.ring.ready[bank * slots_max + slot]).load(cuda::memory_order_acquire);
                const unsigned mask = unsigned(tag >> 9) & 7;
                if ((mask & (1u << d.kind)) && tag == ready_tag(s.epoch, d.layer, e, mask)) {
                    ptr = d.ring.data + bank * d.ring.bank_stride + slot * d.ring.slot_stride + d.ring_offset;
                    hot = 1;
                    served = true;
                }
            }
        }
        d.out[e] = ptr;
        d.out_hot[e] = hot;
        if (d.stats) {
            const bool useful = served && ((s.needed[e / 32] >> (e % 32)) & 1);
            const unsigned mask = __ballot_sync(0xffffffffu, useful);
            if (tid % 32 == 0 && mask) atomicAdd(d.stats->useful_bytes + d.kind,
                                                  word(__popc(mask)) * d.projection_bytes);
        }
    }
    __syncthreads();
    if (d.stats) {
        bool served = false;
        if (tid < unsigned(n)) {
            const unsigned e = unsigned(s.ids[tid]);
            served = e < experts && !d.hot[e] && d.out_hot[e];
        }
        // All lanes participate; <=128 routes permit at most four nonzero adds.
        const unsigned mask = __ballot_sync(0xffffffffu, served);
        if (tid % 32 == 0 && mask) atomicAdd(&d.stats->served_routes, word(__popc(mask)));
    }
}

// Called after count_snapshot by the same CTA. LUT sums use actual route multiplicity.
// Cold prefilter precedes ranking: signed NB remains valid even below zero.
__device__ inline void plan_up(const Dispatch & d, const Planner & p, Scratch & s) {
    const unsigned tid = threadIdx.x, target = d.layer + d.lead;
    if (tid == 0) {
        assert(p.plan && p.n_layers < 65536 && p.cold_limit > 0 && p.cold_limit <= slots_max);
        if (p.timing) {
            p.timing->first_cta_ns = ~word(0);
            p.timing->last_cta_ns = 0;
            p.timing->completed_ctas = 0;
        }
    }
    if (target >= p.n_layers || (!p.weights && !p.bias)) {
        if (tid == 0) p.plan->count = 0;
        return;
    }
    const unsigned bank = target % (d.lead + 1);
    if (tid == 0) {
        p.plan->count = 0; p.plan->bank = bank;
        atomic<word>(d.ring.assignment[bank]).store(0, cuda::memory_order_release);
    }
    for (int e = tid; e < experts; e += threads) d.ring.expert_slot[bank * experts + e] = -1;
    if (tid < slots_max) {
        atomic<word>(d.ring.ready[bank * slots_max + tid]).store(0, cuda::memory_order_release);
        p.completion[tid].done = p.completion[tid].canceled = 0;
    }
    uint64_t keys[2];
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        const unsigned e = 2 * tid + j;
        int score = p.bias ? int(p.bias[e]) * d.n_tokens : 0;
        for (int i = 0; i < d.n_ids * d.n_tokens; ++i) {
            const unsigned src = unsigned(s.ids[i]);
            if (p.weights && src < experts) score += int(p.weights[src * experts + e]);
        }
        const bool cold = !p.projection[0].hot[e] || !p.projection[1].hot[e] || !p.projection[2].hot[e];
        // <=128 row terms plus <=128 bias terms, each in [-32767,32767]:
        // |score| <= 8388352 < 2^23. Offset is strictly positive, preserving sentinel0.
        assert(score >= -8388352 && score <= 8388352);
        keys[j] = cold && (!p.positive_only || score > 0)
            ? (uint64_t(score + (1 << 23)) << 9) | (511 - e) : 0;
    }
    __syncthreads();
    Sort(s.sort).SortDescending(keys, 0, 33);
    s.ranked[2 * tid] = keys[0]; s.ranked[2 * tid + 1] = keys[1];
    __syncthreads();
    if (tid < 32) {
        const uint64_t key = s.ranked[tid];
        const unsigned e = 511 - unsigned(key & 511);
        const bool candidate = key && tid < p.cold_limit && tid < slots_max;
        unsigned mask = 0;
        uint64_t bytes = 0;
        if (candidate) for (int k = 0; k < 3; ++k) if (!p.projection[k].hot[e]) {
            mask |= 1u << k; bytes += p.projection[k].bytes;
        }
        word prefix = bytes;
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            const word add = __shfl_up_sync(0xffffffffu, prefix, offset);
            if (tid >= unsigned(offset)) prefix += add;
        }
        const bool accepted = candidate && prefix <= p.budget_bytes;
        const unsigned accepted_mask = __ballot_sync(0xffffffffu, accepted);
        if (accepted) {
            Descriptor & x = p.plan->item[tid];
            x.epoch = s.epoch; x.target = target; x.expert = e; x.mask = mask;
            for (int k = 0; k < 3; ++k) {
                x.src[k] = static_cast<const uint8_t *>(p.projection[k].mapped[e]);
                x.dst[k] = d.ring.data + bank * d.ring.bank_stride + tid * d.ring.slot_stride + p.projection[k].ring_offset;
                x.bytes[k] = mask & (1u << k) ? p.projection[k].bytes : 0;
            }
            d.ring.expert_slot[bank * experts + e] = int16_t(tid);
        }
        if (tid == 0) p.plan->count = __popc(accepted_mask);
    }
    __syncthreads();
    if (tid == 0) atomic<word>(d.ring.assignment[bank]).store(epoch_target(s.epoch, target), cuda::memory_order_release);
}

// Reusable count_snapshot/plan_up also allow an integration-specific combined kernel.
template<bool PlanUP>
__global__ __launch_bounds__(threads) void dispatch_kernel(Dispatch d, Planner p) {
    __shared__ Scratch s;
    count_snapshot(d, s);
    if (PlanUP) plan_up(d, p, s);
}

// Exact masked tail; aligned 16-byte bulk loads/stores. No padding is read or written.
__device__ inline void copy16(uint8_t * dst, const uint8_t * src, uint64_t offset, uint64_t end) {
    if (offset + 16 <= end && ((uintptr_t(src + offset) | uintptr_t(dst + offset)) & 15) == 0) {
        *reinterpret_cast<int4 *>(dst + offset) = __ldcs(reinterpret_cast<const int4 *>(src + offset));
    } else {
        #pragma unroll
        for (int j = 0; j < 16; ++j) if (offset + j < end) dst[offset + j] = src[offset + j];
    }
}

__device__ inline word globaltimer_ns() {
    word ns;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
    return ns;
}
// Exact bytes assigned to this CTA in an executed slice (including a short tail).
__device__ inline word cta_slice_bytes(uint64_t bytes) {
    word result = 0;
    #pragma unroll
    for (unsigned i = 0; i < 2; ++i) {
        const uint64_t start = blockIdx.x * threads * 64 + i * copy_blocks * threads * 64;
        if (bytes > start) result += bytes - start < threads * 64 ? bytes - start : threads * 64;
    }
    return result;
}

// Launch exactly <<<32,256>>> on the single FIFO copy stream after the plan event.
// Each CTA copies <=32KiB between deadline checks. All CTAs complete every item,
// including canceled items; nobody waits for main-stream progress or READY.
__global__ __launch_bounds__(threads) void copy_kernel(const Plan * plan, Completion * completion,
                                                       const word * progress, Ring ring,
                                                       Timing * timing, Stats * stats) {
    __shared__ unsigned canceled;
    word copied[3] = {0, 0, 0}; // Only CTA leader updates/flushes these launch totals.
    if (threadIdx.x == 0) {
        assert(gridDim.x == copy_blocks && blockDim.x == threads);
        if (timing) atomicMin(&timing->first_cta_ns, globaltimer_ns());
        if (stats && blockIdx.x == 0) atomicAdd(&stats->copy_launches, word(1));
    }
    for (unsigned slot = 0; slot < plan->count; ++slot) {
        const Descriptor & x = plan->item[slot];
        Completion & c = completion[slot];
        for (int k = 0; k < 3; ++k) for (uint64_t base = 0; base < x.bytes[k]; base += copy_chunk) {
            if (threadIdx.x == 0) {
                if (expired(progress, x.epoch, x.target)) atomic<unsigned>(c.canceled).store(1, cuda::memory_order_release);
                canceled = atomic<unsigned>(c.canceled).load(cuda::memory_order_acquire);
            }
            __syncthreads();
            if (canceled) break; // Uniform CTA branch; completion below remains mandatory.
            const uint64_t end = x.bytes[k] < base + copy_chunk ? x.bytes[k] : base + copy_chunk;
            for (uint64_t i = base + blockIdx.x * threads * 64 + threadIdx.x * 16;
                 i < base + copy_chunk; i += copy_blocks * threads * 64) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint64_t offset = i + j * threads * 16;
                    if (offset < end) copy16(x.dst[k], x.src[k], offset, end);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0 && stats) copied[k] += cta_slice_bytes(end - base);
        }
        __syncthreads(); // All CTA writes happen before this CTA's completion release.
        if (threadIdx.x == 0) {
            if (expired(progress, x.epoch, x.target)) atomic<unsigned>(c.canceled).store(1, cuda::memory_order_release);
            const unsigned done = atomic<unsigned>(c.done).fetch_add(1, cuda::memory_order_acq_rel);
            if (done == copy_blocks - 1 && !atomic<unsigned>(c.canceled).load(cuda::memory_order_acquire)
                && !expired(progress, x.epoch, x.target)) {
                atomic<word>(ring.ready[plan->bank * slots_max + slot]).store(
                    ready_tag(x.epoch, x.target, x.expert, x.mask), cuda::memory_order_release);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stats) for (int k = 0; k < 3; ++k) if (copied[k]) atomicAdd(stats->copied_bytes + k, copied[k]);
        if (timing) {
            atomicMax(&timing->last_cta_ns, globaltimer_ns());
            const unsigned done = atomic<unsigned>(timing->completed_ctas).fetch_add(1, cuda::memory_order_acq_rel);
            if (done == copy_blocks - 1 && stats) {
                const word first = atomic<word>(timing->first_cta_ns).load(cuda::memory_order_relaxed);
                const word last = atomic<word>(timing->last_cta_ns).load(cuda::memory_order_relaxed);
                atomicAdd(&stats->observed_kernel_span_ns, last - first);
            }
        }
    }
}
} // namespace moe_tp
#endif
