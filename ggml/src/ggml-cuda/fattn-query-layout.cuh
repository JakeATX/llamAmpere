#pragma once

// Map physical tensor-core columns to query tokens and heads.
#if defined(__CUDACC__) || defined(__HIPCC__)
# define GGML_FATTN_LAYOUT_HD __host__ __device__
#else
# define GGML_FATTN_LAYOUT_HD
#endif

template<int query_rows, int head_columns, bool compact_q5g6 = false>
struct ggml_fattn_query_layout {
    static_assert(query_rows > 0 && head_columns > 0, "invalid physical shape");
    static_assert(!compact_q5g6 || (query_rows == 4 && head_columns == 8),
                  "compact Q5/G6 requires the existing physical 4x8 tile");
    static constexpr int token_step = compact_q5g6 ? 5 : query_rows;
    static constexpr int mask_rows = token_step;
    GGML_FATTN_LAYOUT_HD static constexpr int token(int slot) {
        return slot / (compact_q5g6 ? 6 : head_columns);
    }
    GGML_FATTN_LAYOUT_HD static constexpr int head(int slot) {
        return slot % (compact_q5g6 ? 6 : head_columns);
    }
    // Padding slots use mask row zero; their outputs are not stored.
    GGML_FATTN_LAYOUT_HD static constexpr int mask_row(int slot) {
        return compact_q5g6 && token(slot) >= 5 ? 0 : token(slot);
    }
    GGML_FATTN_LAYOUT_HD static constexpr int query_tiles(int tokens) {
        return (tokens + token_step - 1) / token_step;
    }
};
#undef GGML_FATTN_LAYOUT_HD
