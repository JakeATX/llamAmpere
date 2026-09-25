#pragma once

// Fallback ledger: host-side counters that record which route an op took and why a fast path was
// skipped (CUDA graph rebuilds, mul_mat routes by width, flash-attention kernels, fusions that did
// not fire, draft-vocab fallbacks). Off unless GGML_LEDGER=1 (or ggml_ledger_set_enabled(true)).
// Counting never touches the device and never synchronizes; a disabled ledger costs one cached
// bool test per site.
//
// Counts are (site, key) pairs. `site` groups related counters ("cuda.graph", "cuda.mul_mat", ...),
// `key` names the outcome. Both strings are copied on first use. With GGML_LEDGER=1 the table is
// printed to stderr at process exit; tools can also walk it with ggml_ledger_foreach().

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    struct ggml_ledger_slot;

    // true when counting is on; the first call reads GGML_LEDGER
    GGML_API bool ggml_ledger_enabled(void);
    GGML_API void ggml_ledger_set_enabled(bool enabled);

    // stable handle for (site, key), created on first use; cache it at hot sites
    GGML_API struct ggml_ledger_slot * ggml_ledger_slot_get(const char * site, const char * key);
    GGML_API void ggml_ledger_slot_add(struct ggml_ledger_slot * slot, int64_t n);

    // one-shot form: looks the slot up every call (a mutex and a map lookup), for cold sites
    GGML_API void ggml_ledger_add(const char * site, const char * key, int64_t n);

    // same, with a printf-style key
    GGML_API void ggml_ledger_addf(const char * site, int64_t n, const char * key_fmt, ...) GGML_ATTRIBUTE_FORMAT(3, 4);

    typedef void (*ggml_ledger_cb)(const char * site, const char * key, int64_t count, void * user_data);

    // visits every counter, sorted by site then key
    GGML_API void ggml_ledger_foreach(ggml_ledger_cb cb, void * user_data);

    // prints every non-zero counter to stderr
    GGML_API void ggml_ledger_print(void);

    // zeroes every counter (slots stay valid)
    GGML_API void ggml_ledger_reset(void);

#ifdef __cplusplus
}
#endif
