// CPU writer for the versioned 3/4-bit pair stream. No global scratch state.
#pragma once
#include "ggml-aif175.h"

#define AIF175_TRACE_BYTES (32*(8192+4096))

typedef struct {
    float values[2][256];
} aif175_errors;

typedef struct {
    float rotated[16*512];
    uint8_t levels[2*65536];
    float costs[65536];
    float best[8192];
    aif175_errors tables[64];
    uint8_t trace[AIF175_TRACE_BYTES];
} aif175_workspace;

static double aif175_level_map(uint8_t * levels) {
    double square_sum = 0;
    for (int s = 0; s < 65536; ++s) {
        const int level = aif175_level_state((uint32_t)s, AIF175_MUL);
        levels[2*s] = (uint8_t)(level+128);
        levels[2*s+1] = (uint8_t)(aif175_level_state((uint32_t)s, (uint32_t)((uint64_t)AIF175_MUL*AIF175_MUL))+128);
        square_sum += (double)level*level;
    }
    return sqrt(square_sum/65536);
}

static void aif175_predecessors(const float * current, int width, int first, int count, float * best, uint8_t * trace) {
    const int tails = 1 << (16-width);
    // Only these tails feed the next live state range or a valid traceback.
    const int begin_tail = count < tails ? first%tails : 0;
    const int end_tail = begin_tail + (count < tails ? count : tails);
    // A small range in a nonzero band has one live candidate per tail.
    if (count < tails && first >= tails) {
        const int head = first/tails;
        for (int tail = begin_tail; tail < end_tail; ++tail) {
            const float value = current[head*tails+tail];
            const bool wins = value < INFINITY;
            best[tail] = wins ? value : INFINITY;
            trace[tail] = wins ? (uint8_t) head : 0;
        }
        return;
    }
    if (first < tails) {
        memcpy(best + begin_tail, current + begin_tail, (size_t)(end_tail-begin_tail)*sizeof(float));
    } else {
        for (int tail = begin_tail; tail < end_tail; ++tail) { best[tail] = INFINITY; }
    }
    memset(trace + begin_tail, 0, (size_t)(end_tail-begin_tail));
    if (count < tails) { return; }
    // Keep head zero for all-infinite ties; visit only complete reachable bands.
    const int begin = first/tails > 1 ? first/tails : 1;
    const int end = (first+count)/tails;
    for (int head = begin; head < end; ++head) {
        const float * candidate = current + head*tails;
        for (int tail = 0; tail < tails; ++tail) {
            if (candidate[tail] < best[tail]) {
                best[tail] = candidate[tail];
                trace[tail] = (uint8_t) head;
            }
        }
    }
}

static void aif175_error_tables(const float * target, const float * metric, float errors[2][256]) {
    for (int j = 0; j < 2; ++j) {
        if (target[j] == 0.0f && metric[j] == 0.0f) {
            for (int level = 0; level < 256; ++level) { errors[j][level] = metric[j]; }
            continue;
        }
        for (int level = 0; level < 256; ++level) {
            const float delta = (float)(level-128) - target[j];
            errors[j][level] = (delta*delta)*metric[j];
        }
    }
}

static void aif175_state_costs(const float * error0, const float * error1, const uint8_t * levels, int width,
                             int first, int count, bool zero_error, const float * best, float * costs) {
    // Padded rows have zero targets and metrics, hence exactly positive-zero errors.
    if (zero_error) {
        if (!best) {
            for (int s = first; s < first+count; ++s) { costs[s] = 0.0f; }
        } else {
            const int symbols = 1 << width;
            for (int begin = first; begin < first+count; begin += symbols) {
                const float value = best[begin >> width] + 0.0f;
                for (int s = begin; s < begin+symbols; ++s) { costs[s] = value; }
            }
        }
        return;
    }
    if (!best) {
        for (int s = first; s < first+count; ++s) {
            costs[s] = error0[levels[2*s]] + error1[levels[2*s+1]];
        }
        return;
    }
    // Live ranges contain whole symbol groups with one prior cost per group.
    const int symbols = 1 << width;
    for (int begin = first; begin < first+count; begin += symbols) {
        const float prior = best[begin >> width];
        for (int s = begin; s < begin+symbols; ++s) {
            const float error = error0[levels[2*s]] + error1[levels[2*s+1]];
            costs[s] = prior + error;
        }
    }
}

static float aif175_terminal_cost(const float * costs, int width, int state, float error, uint8_t * trace) {
    const int tails = 1 << (16-width), tail = state >> width;
    float prior = costs[tail];
    int winner = 0;
    for (int head = 1; head < (1 << width); ++head) {
        const float candidate = costs[head*tails+tail];
        if (candidate < prior) { prior = candidate; winner = head; }
    }
    *trace = (uint8_t) winner;
    return prior + error;
}

static float aif175_fixed_tail_cost(const float * costs, const uint8_t * levels, int width, int state,
                                   const float * error0, const float * error1, float error, uint8_t * trace, uint8_t * last_trace) {
    const int last_width = 7-width, last_tails = 1 << (16-last_width), last_tail = state >> last_width;
    float prior = INFINITY;
    int winner = 0;
    for (int head = 0; head < (1 << last_width); ++head) {
        const int s = head*last_tails+last_tail;
        const float pair_error = error0[levels[2*s]] + error1[levels[2*s+1]];
        const float candidate = aif175_terminal_cost(costs, width, s, pair_error, trace + (s >> width));
        if (!head || candidate < prior) { prior = candidate; winner = head; }
    }
    *last_trace = (uint8_t) winner;
    return prior + error;
}

static float aif175_fixed_three_cost(const float * costs, const uint8_t * levels, int width, int state,
                                    const float * error0, const float * error1, const float * next0, const float * next1,
                                    float error, uint8_t * trace, uint8_t * next_trace, uint8_t * last_trace) {
    const int tails = 1 << (16-width), tail = state >> width, middle_width = 7-width;
    float prior = INFINITY;
    int winner = 0;
    for (int head = 0; head < (1 << width); ++head) {
        const int s = head*tails+tail;
        const float pair_error = next0[levels[2*s]] + next1[levels[2*s+1]];
        const float candidate = aif175_fixed_tail_cost(costs, levels, width, s, error0, error1, pair_error,
                                                     trace, next_trace + (s >> middle_width));
        if (!head || candidate < prior) { prior = candidate; winner = head; }
    }
    *last_trace = (uint8_t) winner;
    return prior + error;
}

static bool aif175_path(const float * target, const float * metric, const aif175_errors * tables, const uint8_t * levels,
                       int offset, int boundary, float * costs, float * best, uint8_t * trace, uint16_t * path) {
    int trace_pos = 0, first = 0, count = 65536;
    for (int i = 0; i < 64; ++i) {
        const int p = (i + offset)%64, width = 3 + p%2, tails = 1 << (16-width);
        const float * error0 = tables[p].values[0], * error1 = tables[p].values[1];
        // A fixed end needs only 128 states two steps before it.
        if (i == 61 && boundary >= 0 && count == 65536) {
            const int next_p = (p+1)%64, last_p = (p+2)%64, middle_width = 7-width;
            const int middle_tails = 1 << (16-middle_width);
            const float error = tables[last_p].values[0][levels[2*boundary]] + tables[last_p].values[1][levels[2*boundary+1]];
            costs[boundary] = aif175_fixed_three_cost(costs, levels, width, boundary, error0, error1,
                                                    tables[next_p].values[0], tables[next_p].values[1], error, trace + trace_pos,
                                                    trace + trace_pos + tails, trace + trace_pos + tails + middle_tails + (boundary >> width));
            trace_pos += 2*tails + middle_tails;
            break;
        }
        if (i) {
            // Save all predecessor costs before reusing the state buffer.
            aif175_predecessors(costs, width, first, count, best, trace + trace_pos);
            // A fixed start reaches one aligned range until all states are reachable.
            if (count < tails) {
                first = (first & (tails-1)) << width;
                count <<= width;
            } else {
                first = 0; count = 65536;
            }
        } else if (boundary >= 0) {
            first = (boundary & (tails-1)) << width;
            count = 1 << width;
        }
        const float pair_zero = error0[0] + error1[0];
        uint32_t pair_zero_bits;
        memcpy(&pair_zero_bits, &pair_zero, sizeof(pair_zero_bits));
        const bool zero_error = pair_zero_bits == 0 && target[2*p] == 0.0f && target[2*p+1] == 0.0f &&
                                metric[2*p] == 0.0f && metric[2*p+1] == 0.0f;
        aif175_state_costs(error0, error1, levels, width, first, count, zero_error, i ? best : NULL, costs);
        trace_pos += tails;
    }
    int state = boundary;
    if (state < 0) {
        state = 0;
        for (int s = 1; s < 65536; ++s) { if (costs[s] < costs[state]) { state = s; } }
    }
    if (!isfinite(costs[state])) { return false; }
    for (int i = 63; i >= 0; --i) {
        const int p = (i + offset)%64, width = 3 + p%2;
        trace_pos -= 1 << (16-width);
        path[p] = (uint16_t) state;
        if (i) {
            const int tail = state >> width;
            state = tail | ((int) trace[trace_pos + tail] << (16-width));
        }
    }
    const int width = 3 + offset%2;
    return boundary < 0 || (state >> width) == (boundary & ((1 << (16-width))-1));
}

static bool aif175_pack(const uint16_t * path, uint8_t * dst) {
    uint32_t words[7] = {0};
    int end = 0;
    for (int p = 0; p < 64; p += 2) {
        const int word = end/32, left = 32-end%32;
        // Check both links before appending the seven-bit pair.
        if ((path[p] >> 3) != (path[(p+63)%64] & 8191u) ||
            (path[p+1] >> 4) != (path[p] & 4095u)) { return false; }
        const uint32_t symbol = ((path[p] & 7u) << 4) | (path[p+1] & 15u);
        if (left >= 7) {
            words[word] |= symbol << (left-7);
        } else {
            words[word] |= symbol >> (7-left);
            words[word+1] |= symbol << (25+left);
        }
        end += 7;
    }
    for (int w = 0; w < 7; ++w) {
        for (int b = 0; b < 4; ++b) { dst[4*w+b] = (uint8_t)(words[w] >> (8*b)); }
    }
    return true;
}

static bool aif175_quantize_slab_impl(const float * src, int64_t K, void * output, size_t nbytes,
                                    void * (*allocate)(size_t), void (*release)(void *)) {
    const size_t expected = ggml_ai4_slab_nbytes(GGML_TYPE_AIF175_U2, K);
    if (!src || !output || !expected || nbytes != expected || expected > SIZE_MAX-sizeof(aif175_workspace) ||
        (uint64_t)K > SIZE_MAX/(16*sizeof(float))) {
        return false;
    }
    const size_t values = 16*(size_t)K;
    bool nonzero = false;
    // Reject bad weights before any block enters the trellis search.
    for (size_t i = 0; i < values; ++i) {
        if (!isfinite(src[i])) { return false; }
        nonzero |= src[i] != 0.0f;
    }
    if (!nonzero) {
        memset(output, 0, expected);
        return true;
    }
    // Keep output unchanged on invalid input, allocation failure or an open tail.
    float * rotated = NULL;
    uint8_t * levels = NULL;
    aif175_workspace * workspace = NULL;
    uint8_t * trace = NULL;
    uint8_t * packed = NULL;
    double level_rms = 0;
    bool ok = false;
    workspace = (aif175_workspace *) allocate(sizeof(aif175_workspace) + expected);
    if (!workspace) { goto done; }
    rotated = workspace->rotated;
    levels = workspace->levels;
    trace = workspace->trace;
    packed = (uint8_t *)(workspace + 1);
    level_rms = aif175_level_map(levels);
    for (int64_t block = 0; block < K/512; ++block) {
        uint8_t * dst = packed + (size_t)block*AIF175_BLOCK_BYTES;
        float scales[16], top = 0;
        for (int r = 0; r < 16; ++r) {
            float * row = rotated + r*512;
            bool row_nonzero = false;
            for (int k = 0; k < 512; ++k) {
                const float value = src[(size_t)r*K + block*512+k];
                row[k] = value * (ai4_sign_neg(k%256) ? -1.0f : 1.0f);
                row_nonzero |= value != 0.0f;
            }
            if (!row_nonzero) {
                scales[r] = 0.0f;
                dst[2*r] = dst[2*r+1] = 0;
                continue;
            }
            for (int k = 0; k < 512; k += 256) {
                ai4_fwht256(row + k);
                for (int j = 0; j < 256; ++j) {
                    row[k+j] *= 1.0f/16.0f;
                    if (!isfinite(row[k+j])) { goto done; }
                }
            }
            double sum = 0;
            for (int k = 0; k < 512; ++k) {
                const double value = row[k];
                sum += value*value;
            }
            const uint16_t h = GGML_FP32_TO_FP16((float)(sqrt(sum/512)/level_rms));
            scales[r] = ai4_fp16_to_fp32(h);
            if (!isfinite(scales[r])) { goto done; }
            dst[2*r] = (uint8_t)h; dst[2*r+1] = (uint8_t)(h >> 8);
            if (scales[r] > top) { top = scales[r]; }
        }
        if (top == 0) { memset(dst + 32, 0, AIF175_BLOCK_BYTES-32); continue; }
        float metric[128];
        for (int r = 0; r < 16; ++r) {
            const float ratio = scales[r]/top;
            for (int k = 0; k < 8; ++k) { metric[8*r+k] = ratio*ratio; }
        }
        for (int tile = 0; tile < 64; ++tile) {
            float target[128];
            uint16_t path[64];
            const int col = (tile/16)*128 + (tile%16)*8;
            for (int r = 0; r < 16; ++r) {
                for (int k = 0; k < 8; ++k) {
                    target[8*r+k] = scales[r] ? rotated[r*512+col+k]/scales[r] : 0;
                }
            }
            for (int p = 0; p < 64; ++p) {
                aif175_error_tables(target + 2*p, metric + 2*p, workspace->tables[p].values);
            }
            if (!aif175_path(target, metric, workspace->tables, levels, 31, -1, workspace->costs, workspace->best, trace, path) ||
                !aif175_path(target, metric, workspace->tables, levels, 0, path[63], workspace->costs, workspace->best, trace, path) ||
                !aif175_pack(path, dst + 32 + tile*AIF175_TILE_BYTES)) { goto done; }
        }
    }
    memcpy(output, packed, expected);
    ok = true;
done:
    release(workspace);
    return ok;
}
