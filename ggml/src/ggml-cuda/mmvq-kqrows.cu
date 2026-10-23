// [KQROWS] switch, defaults and dispatch for the K-quant row kernel (see mmvq-kqrows.cuh).
#include "mmvq-kqrows.cuh"

#include <cstdlib>
#include <cstring>
#include <string>

// launches taken (test harness: ggml_backend_cuda_fusion_count "kqrows", read as a before/after delta)
int64_t ggml_cuda_kqrows_count = 0;

namespace {

struct kqrows_cfg {
    bool on       = false;
    bool types[3] = { true, true, true }; // q4_K, q5_K, q6_K
    int  rows     = 0;  // 0 = per-width default
    int  nwarps   = 0;  // 0 = the MMVQ table (bit-identical to MMVQ)
    int  min_n    = 2;
    int  max_n    = 8;
};

static int kqrows_type_index(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return 0;
        case GGML_TYPE_Q5_K: return 1;
        case GGML_TYPE_Q6_K: return 2;
        default:             return -1;
    }
}

static const kqrows_cfg & kqrows_get_cfg() {
    static const kqrows_cfg cfg = [] {
        kqrows_cfg c;
        const char * on = getenv("GGML_CUDA_KQ_ROWS");
        c.on = on != nullptr && atoi(on) != 0;
        if (const char * t = getenv("GGML_CUDA_KQ_ROWS_TYPES")) {
            const std::string s(t);
            c.types[0] = s.find("q4_K") != std::string::npos;
            c.types[1] = s.find("q5_K") != std::string::npos;
            c.types[2] = s.find("q6_K") != std::string::npos;
        }
        if (const char * r = getenv("GGML_CUDA_KQ_ROWS_R")) {
            const int v = atoi(r);
            c.rows = (v == 1 || v == 2 || v == 4 || v == 8) ? v : 0;
        }
        if (const char * w = getenv("GGML_CUDA_KQ_ROWS_NW")) {
            const int v = atoi(w);
            c.nwarps = (v == 2 || v == 4 || v == 8) ? v : 0;
        }
        if (const char * m = getenv("GGML_CUDA_KQ_ROWS_MIN_N")) {
            const int v = atoi(m);
            c.min_n = v < 2 ? 2 : (v > 8 ? 8 : v);
        }
        if (const char * m = getenv("GGML_CUDA_KQ_ROWS_MAX_N")) {
            const int v = atoi(m);
            c.max_n = v < 2 ? 2 : (v > 8 ? 8 : v);
        }
        if (c.on) {
            GGML_LOG_INFO("kqrows: on, types q4_K=%d q5_K=%d q6_K=%d, widths %d..%d, rows %d (0 = per width), nwarps %d (0 = MMVQ table)\n",
                c.types[0], c.types[1], c.types[2], c.min_n, c.max_n, c.rows, c.nwarps);
        }
        return c;
    }();
    return cfg;
}

// The MMVQ table's warp count (calc_nwarps, MMVQ_PARAMETERS_GENERIC, default QC4_* values): 4 at widths 2..4;
// at 5..8 Q4_K/Q5_K keep 4 (QC4_NWARPS_58_K) and the other types drop to 2. Matching it keeps the output
// bit-identical to MMVQ at the same width.
static int kqrows_table_nwarps(ggml_type type, int ncols_dst) {
    if (ncols_dst <= 4) {
        return 4;
    }
    return (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) ? 4 : 2;
}

// [KQROWS] default table: rows per CTA (0 = not taken, keep the current route) and warps per CTA (0 = the MMVQ table
// count, bit-identical to MMVQ) for each type, width 2..8 and row-count class. Small = at most 1024 rows (router,
// hyper-connection and gate-sized products; MMVQ's 8 rows per CTA leave the card under-filled there), large = more.
// Filled from AIFAST/ROWMMVQ kq_table.py --emit on the home 3090 Ti (350 W): a cell is set only where the row kernel
// beat the current route at that shape.
struct kqrows_cell { int8_t rows; int8_t nwarps; };
static constexpr int KQROWS_SMALL_MAX_ROWS = 1024;
//                                            width:   2       3       4       5       6       7       8
// results/perf3 (fixed load order), KQ_EMIT=1 KQ_RIVALS=kqtc: a cell is set only where that row count beat both
// today's route and KQTC by >= 3% on every timed shape of its class (worst case: q4_K w2 -14.3%, q5_K w2 -16.8%,
// q6_K w2 -23.7%, w3 -15.2%, w4 -12.4%). Large matrices: KQTC wins every width, so no cells.
static const kqrows_cell kqrows_tab_small[3][7] = {
    /* q4_K */ { {2, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} },
    /* q5_K */ { {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} },
    /* q6_K */ { {1, 0}, {2, 0}, {2, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} },
};
static const kqrows_cell kqrows_tab_large[3][7] = {
    /* q4_K */ { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} },
    /* q5_K */ { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} },
    /* q6_K */ { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} },
};

// rows/nwarps for this product, rows 0 = not taken
static kqrows_cell kqrows_pick(int ti, ggml_type type, int64_t nrows_x, int64_t ncols_dst) {
    const kqrows_cfg & c = kqrows_get_cfg();
    kqrows_cell cell = (nrows_x <= KQROWS_SMALL_MAX_ROWS ? kqrows_tab_small : kqrows_tab_large)[ti][ncols_dst - 2];
    if (c.rows > 0) {
        cell.rows = (int8_t) c.rows; // forced for every cell (sweeps)
    }
    if (cell.rows == 0) {
        return cell;
    }
    if (c.nwarps > 0) {
        cell.nwarps = (int8_t) c.nwarps;
    } else if (cell.nwarps == 0) {
        cell.nwarps = (int8_t) kqrows_table_nwarps(type, (int) ncols_dst);
    }
    return cell;
}

static bool kqrows_eligible(int ti, int cc, int64_t ncols_x, int64_t ncols_dst) {
    const kqrows_cfg & c = kqrows_get_cfg();
    return c.on && ti >= 0 && c.types[ti] && ncols_dst >= c.min_n && ncols_dst <= c.max_n &&
        GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_DP4A && ncols_x % QK_K == 0;
}

} // namespace

bool ggml_cuda_kqrows_wants(ggml_type type, int cc, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst) {
    const int ti = kqrows_type_index(type);
    return kqrows_eligible(ti, cc, ncols_x, ncols_dst) && kqrows_pick(ti, type, nrows_x, ncols_dst).rows > 0;
}

bool ggml_cuda_kqrows_try(ggml_type type, const void * vx, const void * vy_q8_1, float * dst,
        int ncols_x, int nrows_x, int ncols_dst, int stride_row_x, int stride_col_y, int stride_col_dst,
        int cc, cudaStream_t stream) {
    const int ti = kqrows_type_index(type);
    if (!kqrows_eligible(ti, cc, ncols_x, ncols_dst)) {
        return false;
    }
    const kqrows_cell cell = kqrows_pick(ti, type, nrows_x, ncols_dst);
    if (cell.rows == 0) {
        return false;
    }
    const int rows   = cell.rows;
    const int nwarps = cell.nwarps;
    switch (type) {
        case GGML_TYPE_Q4_K:
            ggml_cuda_kqrows_launch<GGML_TYPE_Q4_K>(vx, vy_q8_1, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, rows, nwarps, stream);
            break;
        case GGML_TYPE_Q5_K:
            ggml_cuda_kqrows_launch<GGML_TYPE_Q5_K>(vx, vy_q8_1, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, rows, nwarps, stream);
            break;
        case GGML_TYPE_Q6_K:
            ggml_cuda_kqrows_launch<GGML_TYPE_Q6_K>(vx, vy_q8_1, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, rows, nwarps, stream);
            break;
        default:
            return false;
    }
    ++ggml_cuda_kqrows_count;
    return true;
}
