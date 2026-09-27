#pragma once
// Physical tensor-core columns need not form a rectangular token/head grid.
// Keep this header free of CUDA runtime dependencies so its address contract can
// be exhaustively tested by an ordinary C++ compiler.
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
    static constexpr int slots = query_rows * head_columns;
    static constexpr int token_step = compact_q5g6 ? 5 : query_rows;
    static constexpr int mask_rows = token_step;
    GGML_FATTN_LAYOUT_HD static constexpr int token(int slot) {
        return slot / (compact_q5g6 ? 6 : head_columns);
    }
    GGML_FATTN_LAYOUT_HD static constexpr int head(int slot) {
        return slot % (compact_q5g6 ? 6 : head_columns);
    }
    // Slots 30 and 31 are padding. Their internal arithmetic can read mask row
    // zero, but must NEVER form an address into nonexistent mask row five.
    GGML_FATTN_LAYOUT_HD static constexpr int mask_row(int slot) {
        return compact_q5g6 && token(slot) >= 5 ? 0 : token(slot);
    }
    GGML_FATTN_LAYOUT_HD static constexpr int query_tiles(int tokens) {
        return (tokens + token_step - 1) / token_step;
    }
    GGML_FATTN_LAYOUT_HD static constexpr bool live(int slot, int tokens, int gqa,
                                                   int token_tile=0, int head_tile=0) {
        return slot >= 0 && slot < slots && token_tile * token_step + token(slot) < tokens
            && head_tile * head_columns + head(slot) < gqa;
    }
};
#undef GGML_FATTN_LAYOUT_HD
