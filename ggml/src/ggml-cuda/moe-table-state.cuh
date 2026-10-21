#pragma once

// Unincluded preparation layer. All storage is reserved before expert pools.
#include "moe-table-model.h"
#include "moe-table-prefetch.cuh"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace moe_tp {
struct Input {
    const void * const * table; // Live DEVICE table: a cold entry is its current GPU host alias.
    const uint8_t * hot;
    uint64_t bytes;
};
struct Snapshot { const void ** pointers; uint8_t * hot; };
struct State {
    Model model;
    unsigned cold_limit = 20;
    uint64_t byte_budget = 0;
    int mode = 0; // 1 prediction, 2 fixed-prior copy stub, 3 prediction and copy.
    int device = -1;
    cudaStream_t copy_stream = nullptr, origin = nullptr, owner_stream = nullptr;
    cudaEvent_t tail = nullptr;
    std::vector<cudaEvent_t> forks;
    bool issued = false;
    bool single_stream = false;
    const void * evaluation_context = nullptr;
    Ring ring{};
    word * progress = nullptr;
    std::vector<std::array<Projection, 3>> projections;
    std::vector<std::array<Snapshot, 3>> snapshots;
    std::vector<Planner> planners;
    Stats * stats = nullptr;
    const void * owner_context = nullptr;

    static size_t aligned(size_t n) { return (n + 255) & ~size_t(255); }
    size_t slot_stride() const {
        size_t result = 0;
        for (const auto & layer : model.layers) {
            size_t bytes = 0;
            for (auto n : layer.projection_bytes) bytes += aligned(n);
            result = std::max(result, bytes);
        }
        return result;
    }
    size_t workspace_bytes() const {
        const size_t nl = model.layers.size(), banks = model.lead + 1;
        size_t n = aligned(banks * slots_max * slot_stride());
        n += aligned(banks * slots_max * sizeof(word));
        n += aligned(banks * sizeof(word)) + aligned(banks * experts * sizeof(int16_t));
        n += aligned(sizeof(word)) + aligned(nl * sizeof(Stats));
        n += nl * (aligned(sizeof(Plan)) + aligned(slots_max * sizeof(Completion)) + aligned(sizeof(Timing)));
        n += nl * 3 * (aligned(experts * sizeof(void *)) + aligned(experts));
        for (const auto & layer : model.layers) if (layer.active()) {
            n += aligned(experts * experts * sizeof(int16_t)) + 2 * aligned(experts * sizeof(int16_t));
        }
        return n;
    }
    void release_streams() {
        // Caller drains main and copy streams before any event, mapping or workspace is released.
        for (auto event : forks) if (event) CUDA_CHECK(cudaEventDestroy(event));
        forks.clear();
        if (tail) CUDA_CHECK(cudaEventDestroy(tail));
        if (copy_stream) CUDA_CHECK(cudaStreamDestroy(copy_stream));
        tail = nullptr; copy_stream = origin = owner_stream = nullptr;
        owner_context = evaluation_context = nullptr; issued = false;
    }
    void prepare(char * workspace, size_t capacity, const std::vector<std::array<Input, 3>> & input) {
        if (capacity < workspace_bytes() || input.size() != model.layers.size() ||
                cold_limit == 0 || cold_limit > slots_max || byte_budget == 0 || mode < 1 || mode > 3) {
            throw std::runtime_error("MoE transition workspace/dimension mismatch");
        }
        release_streams();
        CUDA_CHECK(cudaMemset(workspace, 0, workspace_bytes()));
        size_t used = 0;
        const auto take = [&](size_t n) {
            char * p = workspace + used;
            used += aligned(n);
            GGML_ASSERT(used <= capacity);
            return p;
        };
        const unsigned nl = unsigned(model.layers.size()), banks = model.lead + 1;
        ring.slot_stride = slot_stride(); ring.bank_stride = slots_max * ring.slot_stride;
        ring.data = reinterpret_cast<uint8_t *>(take(banks * ring.bank_stride));
        ring.ready = reinterpret_cast<word *>(take(banks * slots_max * sizeof(word)));
        ring.assignment = reinterpret_cast<word *>(take(banks * sizeof(word)));
        ring.expert_slot = reinterpret_cast<int16_t *>(take(banks * experts * sizeof(int16_t)));
        CUDA_CHECK(cudaMemset(ring.expert_slot, 0xff, banks * experts * sizeof(int16_t)));
        progress = reinterpret_cast<word *>(take(sizeof(word)));
        stats = reinterpret_cast<Stats *>(take(nl * sizeof(Stats)));
        projections.resize(nl); snapshots.resize(nl); planners.resize(nl);
        for (unsigned layer = 0; layer < nl; ++layer) {
            auto & p = planners[layer];
            p = {};
            p.plan = reinterpret_cast<Plan *>(take(sizeof(Plan)));
            p.completion = reinterpret_cast<Completion *>(take(slots_max * sizeof(Completion)));
            p.timing = reinterpret_cast<Timing *>(take(sizeof(Timing)));
            p.n_layers = nl; p.cold_limit = cold_limit; p.budget_bytes = byte_budget;
            size_t offset = 0;
            for (unsigned k = 0; k < 3; ++k) {
                const auto & in = input[layer][k];
                if (!in.hot || !in.table ||
                        in.bytes != model.layers[layer].projection_bytes[k]) {
                    throw std::runtime_error("MoE transition source geometry mismatch");
                }
                projections[layer][k] = {in.table, in.hot, in.bytes, offset};
                snapshots[layer][k].pointers = reinterpret_cast<const void **>(take(experts * sizeof(void *)));
                snapshots[layer][k].hot = reinterpret_cast<uint8_t *>(take(experts));
                offset += aligned(in.bytes);
            }
        }
        for (unsigned target = 0; target < nl; ++target) {
            if (target < model.lead || !model.layers[target].active()) continue;
            auto & p = planners[target - model.lead];
            const auto & layer = model.layers[target];
            auto weights = reinterpret_cast<int16_t *>(take(experts * experts * sizeof(int16_t)));
            auto bias = reinterpret_cast<int16_t *>(take(experts * sizeof(int16_t)));
            auto stub_bias = reinterpret_cast<int16_t *>(take(experts * sizeof(int16_t)));
            CUDA_CHECK(cudaMemcpy(weights, layer.weights.data(), experts * experts * sizeof(int16_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(bias, layer.bias.data(), experts * sizeof(int16_t), cudaMemcpyHostToDevice));
            // Fixed source-independent ranking measures copy overhead without row lookup.
            std::array<int64_t, experts> marginal{};
            std::array<int, experts> order;
            for (int e = 0; e < experts; ++e) {
                order[e] = e;
                for (int s = 0; s < experts; ++s) marginal[e] += layer.weights[s * experts + e];
            }
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return marginal[a] > marginal[b]; });
            std::array<int16_t, experts> fixed{};
            for (int rank = 0; rank < experts; ++rank) fixed[order[rank]] = int16_t(experts - rank);
            CUDA_CHECK(cudaMemcpy(stub_bias, fixed.data(), sizeof(fixed), cudaMemcpyHostToDevice));
            p.weights = mode == 2 ? nullptr : weights;
            p.bias = mode == 2 ? stub_bias : bias;
            p.positive_only = mode == 2 || layer.positive_only();
            p.n_layers = nl; p.cold_limit = cold_limit; p.budget_bytes = byte_budget;
            for (int k = 0; k < 3; ++k) p.projection[k] = projections[target][k];
        }
        GGML_ASSERT(used == workspace_bytes());
        int low, high;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&low, &high));
        CUDA_CHECK(cudaStreamCreateWithPriority(&copy_stream, cudaStreamNonBlocking, low));
        CUDA_CHECK(cudaEventCreateWithFlags(&tail, cudaEventDisableTiming));
        forks.resize(nl);
        for (auto & event : forks) CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    void graph_begin(const void * context, cudaStream_t stream, bool allow_single_stream) {
        GGML_ASSERT(!origin);
        evaluation_context = context; origin = stream;
        single_stream = allow_single_stream; issued = false;
    }
    void dispatch(const void * context, unsigned layer, unsigned kind,
                  const int32_t * ids, int n_ids, int n_tokens, int stride,
                  const void * const * base, const uint8_t * hot, word * counters, word * expert_counts) {
        GGML_ASSERT(origin && evaluation_context == context && single_stream);
        GGML_ASSERT(layer < planners.size() && kind < 3 && n_ids > 0 && n_tokens > 0 && n_ids * n_tokens <= routes_max);
        if (!owner_context) { owner_context = context; owner_stream = origin; }
        GGML_ASSERT(owner_context == context && owner_stream == origin);
        Dispatch d{};
        d.ids = ids; d.n_ids = n_ids; d.n_tokens = n_tokens; d.ids_stride = stride;
        d.layer = layer; d.kind = kind; d.base = base; d.hot = hot;
        d.out = snapshots[layer][kind].pointers; d.out_hot = snapshots[layer][kind].hot;
        d.counters = counters; d.expert_counts = expert_counts;
        d.ring = ring; d.progress = progress; d.lead = model.lead;
        d.ring_offset = projections[layer][kind].ring_offset;
        d.stats = stats + layer; d.projection_bytes = projections[layer][kind].bytes;
        Planner p = planners[layer];
        if (mode == 1) p.timing = nullptr;
        if (kind == 0) {
            dispatch_kernel<true><<<1, threads, 0, origin>>>(d, p);
            if (mode != 1 && layer + model.lead < planners.size() && (p.weights || p.bias)) {
                CUDA_CHECK(cudaEventRecord(forks[layer], origin));
                CUDA_CHECK(cudaStreamWaitEvent(copy_stream, forks[layer], 0));
                copy_kernel<<<copy_blocks, threads, 0, copy_stream>>>(p.plan, p.completion, progress, ring,
                    p.timing, stats + layer + model.lead);
                issued = true;
            }
        } else {
            dispatch_kernel<false><<<1, threads, 0, origin>>>(d, p);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    void graph_end(const void * context) {
        GGML_ASSERT(origin && evaluation_context == context);
        if (issued) {
            CUDA_CHECK(cudaEventRecord(tail, copy_stream));
            CUDA_CHECK(cudaStreamWaitEvent(origin, tail, 0));
        }
        origin = nullptr; evaluation_context = nullptr; issued = false;
    }
};
}
