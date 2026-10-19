#pragma once

#include <cmath>
#include <cstddef>

// The measured interval includes host staging and copy completion. Keep deadline headroom.
static inline size_t ggml_moe_prefetch_budget(double lead_us, double elapsed_us, double bytes_per_us, size_t pending_bytes = 0) {
    if (!std::isfinite(lead_us) || !std::isfinite(elapsed_us) || !std::isfinite(bytes_per_us) ||
            lead_us <= elapsed_us || elapsed_us < 0.0 || bytes_per_us <= 0.0) {
        return 0;
    }
    const double bytes = 0.75 * (lead_us - elapsed_us) * bytes_per_us;
    const size_t maximum = 16u << 20;
    const size_t capped = bytes >= (double) maximum ? maximum : (size_t) bytes;
    return capped > pending_bytes ? capped - pending_bytes : 0;
}
