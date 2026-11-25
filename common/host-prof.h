#pragma once

// [HOSTOVL] host phase ranges for an nsys timeline of the decode loop. LLAMA_HOSTPROF=1 turns them on; off (the
// default), each scope is one test of a static bool. The ranges are NVTX ranges when the build found the NVTX v3
// headers (LLAMA_HOSTPROF_NVTX), so `nsys profile -t cuda,nvtx` shows them on the host thread; otherwise no-ops.
// Header only: libllama includes it from src/ as well.

#include <cstdlib>

#if defined(LLAMA_HOSTPROF_NVTX)
#if defined(_WIN32)
// nvtx3 pulls in <windows.h>: keep its min/max macros away from std::min/std::max
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include <nvtx3/nvToolsExt.h>
#endif

namespace llama_hostprof {

inline bool on() {
    static const bool v = [] {
        const char * e = std::getenv("LLAMA_HOSTPROF");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}

inline void push(const char * name) {
#if defined(LLAMA_HOSTPROF_NVTX)
    if (on()) {
        nvtxRangePushA(name);
    }
#else
    (void) name;
#endif
}

inline void pop() {
#if defined(LLAMA_HOSTPROF_NVTX)
    if (on()) {
        nvtxRangePop();
    }
#endif
}

inline void mark(const char * name) {
#if defined(LLAMA_HOSTPROF_NVTX)
    if (on()) {
        nvtxMarkA(name);
    }
#else
    (void) name;
#endif
}

struct scope {
    const bool active;
    explicit scope(const char * name) : active(on()) {
        if (active) {
            push(name);
        }
    }
    ~scope() {
        if (active) {
            pop();
        }
    }
    scope(const scope &) = delete;
    scope & operator=(const scope &) = delete;
};

} // namespace llama_hostprof

#define LLAMA_HOSTPROF_CAT2(a, b) a##b
#define LLAMA_HOSTPROF_CAT(a, b)  LLAMA_HOSTPROF_CAT2(a, b)
#define LLAMA_HOSTPROF_SCOPE(name) llama_hostprof::scope LLAMA_HOSTPROF_CAT(hostprof_scope_, __LINE__)(name)
