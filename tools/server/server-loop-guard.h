#pragma once

#include "llama.h"

#include <array>
#include <cstdint>

struct server_loop_guard {
    std::array<llama_token, 64> history = {};
    std::array<uint64_t, 64> matches = {};
    int next = 0;
    int size = 0;
    int period = 0;
    double repeats = 0;

    bool accept(llama_token tok, int reps, int span) {
        if (reps == 0) {
            return false;
        }
        bool stopped = false;
        for (int p = 1; p <= size; ++p) {
            auto & count = matches[p - 1];
            count = tok == history[(next + 64 - p) % 64] ? count + 1 : 0;
            const uint64_t stretch = count + p;
            if (!stopped && count > 0 && (stretch >= uint64_t(reps)*p || (stretch >= uint64_t(span) && stretch >= uint64_t(4)*p))) {
                period = p;
                repeats = double(stretch)/p;
                stopped = true;
            }
        }
        history[next] = tok;
        next = (next + 1) % 64;
        if (size < 64) {
            ++size;
        }
        return stopped;
    }
};
