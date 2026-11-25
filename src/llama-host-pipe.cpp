#include "llama-host-pipe.h"

#include "llama-impl.h"
#include "ggml-alloc.h"

#include <cstdlib>
#include <cstring>

bool llama_host_lite1_enabled() {
    static const bool on = [] {
        const char * v = getenv("LLAMA_HOST_LITE1");
        return v && strcmp(v, "1") == 0;
    }();
    return on;
}

int64_t llama_host_lite1_rows() {
    static const int64_t rows = [] {
        const char * v = getenv("LLAMA_HOST_LITE1_ROWS");
        const long long n = v && *v ? atoll(v) : 64;
        return (int64_t) (n > 0 ? (n < 65536 ? n : 65536) : 64);
    }();
    return rows;
}

static bool llama_host_op_is_light(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_GET_ROWS:
        case GGML_OP_SCALE:
            return true;
        default:
            return false;
    }
}

bool llama_host_cpu_part_is_lite(ggml_backend_sched_t sched, ggml_backend_t cpu, ggml_cgraph * gf, int64_t max_rows) {
    if (sched == nullptr || cpu == nullptr || gf == nullptr) {
        return false;
    }
    const int n = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n; ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (ggml_backend_sched_get_tensor_backend(sched, node) != cpu) {
            continue;
        }
        if (!llama_host_op_is_light(node->op)) {
            return false;
        }
        // views move no data; the others are bounded by rows and by bytes (a flattened copy has one row)
        const bool no_data = node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                             node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE;
        if (!no_data && (ggml_nrows(node) > max_rows || ggml_nbytes(node) > (size_t) max_rows * 64 * 1024)) {
            return false;
        }
    }
    return true;
}

bool llama_dspark_embd_dev_enabled() {
    static const bool on = [] {
        const char * v = getenv("LLAMA_DSPARK_EMBD_DEV");
        return v && strcmp(v, "1") == 0;
    }();
    return on;
}

ggml_tensor * llama_dspark_embd_dev_copy(const ggml_tensor * src, ggml_backend_dev_t dev, ggml_context_ptr & ctx_out,
        ggml_backend_buffer_ptr & buf_out) {
    if (src == nullptr || src->buffer == nullptr || dev == nullptr || !ggml_backend_buffer_is_host(src->buffer) ||
            ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return nullptr;
    }
    {
        // the copy only helps if the device can run the row lookup on this weight type
        ggml_init_params pp = { 3 * ggml_tensor_overhead(), nullptr, true };
        ggml_context_ptr probe(ggml_init(pp));
        if (!probe) {
            return nullptr;
        }
        ggml_tensor * w   = ggml_dup_tensor(probe.get(), src);
        ggml_tensor * ids = ggml_new_tensor_1d(probe.get(), GGML_TYPE_I32, 1);
        if (!ggml_backend_dev_supports_op(dev, ggml_get_rows(probe.get(), w, ids))) {
            LLAMA_LOG_WARN("%s: LLAMA_DSPARK_EMBD_DEV: %s cannot look up %s rows, the draft keeps the host embeddings\n",
                    __func__, ggml_backend_dev_name(dev), ggml_type_name(src->type));
            return nullptr;
        }
    }
    ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
    ctx_out.reset(ggml_init(ip));
    if (!ctx_out) {
        return nullptr;
    }
    ggml_tensor * dst = ggml_dup_tensor(ctx_out.get(), src);
    ggml_set_name(dst, "token_embd.weight (draft device copy)");
    buf_out.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_out.get(), ggml_backend_dev_buffer_type(dev)));
    if (!buf_out) {
        LLAMA_LOG_WARN("%s: LLAMA_DSPARK_EMBD_DEV: no room for %.1f MiB on %s, the draft keeps the host embeddings\n",
                __func__, ggml_nbytes(src) / 1048576.0, ggml_backend_dev_name(dev));
        ctx_out.reset();
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buf_out.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    LLAMA_LOG_INFO("%s: LLAMA_DSPARK_EMBD_DEV: draft token embeddings on %s (%.1f MiB)\n", __func__,
            ggml_backend_dev_name(dev), ggml_nbytes(src) / 1048576.0);
    return dst;
}
