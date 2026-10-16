#include "nemotron-moe.cuh"
#include "mmvq.cuh"

#include <climits>

static bool nemotron_moe_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t pa = (uintptr_t) a->data;
    const uintptr_t pb = (uintptr_t) b->data;
    return pa <= pb ? pb - pa < ggml_nbytes(a) : pa - pb < ggml_nbytes(b);
}

int ggml_cuda_fuse_nemotron_moe(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, int index) {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_NEMOTRON_MOE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    if (!enabled || index < 0 || index + 6 > graph->n_nodes || !ctx.stream_context().concurrent_events.empty()) {
        return -1;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (cc != 800 && cc != 860) {
        return -1;
    }
    const ggml_tensor * mm = graph->nodes[index];
    if (mm->op != GGML_OP_MUL_MAT_ID || mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm)) {
        return -1;
    }
    const ggml_tensor * matrix = mm->src[0];
    const ggml_tensor * activation = mm->src[1];
    const ggml_tensor * ids = mm->src[2];
    if (!matrix || !activation || !ids || !ggml_cuda_moe_sum_type_supported(matrix->type) ||
            activation->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
            !ggml_is_contiguous(matrix) || !ggml_is_contiguous(activation) || !ggml_is_contiguous(ids) ||
            matrix->ne[0] <= 0 || matrix->ne[0] > 65536 || matrix->ne[0] % ggml_blck_size(matrix->type) != 0 || matrix->ne[1] <= 0 || matrix->ne[1] > 65536 ||
            matrix->ne[2] <= 0 || matrix->ne[2] > 1024 || matrix->ne[3] != 1 ||
            ggml_nbytes(matrix) / ggml_type_size(matrix->type) > INT_MAX ||
            ids->ne[0] < 2 || ids->ne[0] > 32 || ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            ids->ne[0] > matrix->ne[2] || activation->ne[0] != matrix->ne[0] ||
            activation->ne[1] != ids->ne[0] || activation->ne[2] != 1 || activation->ne[3] != 1 ||
            mm->ne[0] != matrix->ne[1] || mm->ne[1] != ids->ne[0] || mm->ne[2] != 1 || mm->ne[3] != 1 ||
            ggml_get_op_params_i32(mm, 3) == GGML_PREC_F32) {
        return -1;
    }
    for (int j = 3; j < GGML_MAX_SRC; ++j) {
        if (mm->src[j]) {
            return -1;
        }
    }
    auto is_weights = [mm](const ggml_tensor * tensor) {
        return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
            tensor->ne[0] == 1 && tensor->ne[1] == mm->ne[1] && tensor->ne[2] == 1 && tensor->ne[3] == 1;
    };
    auto match_mul = [is_weights](const ggml_tensor * mul, const ggml_tensor * input, const ggml_tensor *& weights) {
        if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul) ||
                !ggml_are_same_shape(mul, input) || mul->src[0] != input || !is_weights(mul->src[1])) {
            return false;
        }
        weights = mul->src[1];
        return true;
    };
    int offset = 1;
    const ggml_tensor * weights = nullptr;
    const ggml_tensor * scales = nullptr;
    if (!match_mul(graph->nodes[index + offset], mm, weights)) {
        return -1;
    }
    const ggml_tensor * weighted = graph->nodes[index + offset++];
    if (index + offset < graph->n_nodes && graph->nodes[index + offset]->op == GGML_OP_MUL) {
        scales = weights;
        if (!match_mul(graph->nodes[index + offset], weighted, weights)) {
            return -1;
        }
        weighted = graph->nodes[index + offset++];
    }
    if (index + offset + 4 > graph->n_nodes) {
        return -1;
    }
    const ggml_tensor * permute = graph->nodes[index + offset++];
    const ggml_tensor * cont = graph->nodes[index + offset++];
    const ggml_tensor * sum = graph->nodes[index + offset++];
    ggml_tensor * dst = graph->nodes[index + offset++];
    if (permute->op != GGML_OP_PERMUTE || permute->type != GGML_TYPE_F32 || permute->src[0] != weighted ||
            permute->ne[0] != weighted->ne[1] || permute->ne[1] != weighted->ne[0] ||
            permute->ne[2] != 1 || permute->ne[3] != 1 ||
            permute->nb[0] != weighted->nb[1] || permute->nb[1] != weighted->nb[0] ||
            cont->op != GGML_OP_CONT || cont->src[0] != permute || !ggml_is_contiguous(cont) ||
            !ggml_are_same_shape(cont, permute) || cont->type != GGML_TYPE_F32 ||
            sum->op != GGML_OP_SUM_ROWS || sum->src[0] != cont || sum->type != GGML_TYPE_F32 ||
            sum->ne[0] != 1 || sum->ne[1] != matrix->ne[1] || sum->ne[2] != 1 || sum->ne[3] != 1 ||
            dst->op != GGML_OP_RESHAPE || dst->src[0] != sum || dst->type != GGML_TYPE_F32 ||
            dst->ne[0] != matrix->ne[1] || dst->ne[1] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            !ggml_is_contiguous(sum) || !ggml_is_contiguous(dst)) {
        return -1;
    }
    ggml_op ops[7];
    for (int j = 0; j < offset; ++j) {
        ops[j] = graph->nodes[index + j]->op;
    }
    const int output = index + offset - 1;
    if (!ggml_can_fuse_subgraph(graph, index, offset, ops, &output, 1)) {
        return -1;
    }
    const ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(ctx.device);
    for (const ggml_tensor * tensor : { matrix, activation, ids, weights, scales, (const ggml_tensor *) dst }) {
        if (tensor && (!tensor->data || !tensor->buffer || ggml_backend_buffer_get_type(tensor->buffer) != buft)) {
            return -1;
        }
    }
    // The allocator can reuse a dead intermediate for dst while a fused input is still live.
    for (const ggml_tensor * input : { matrix, activation, ids, weights, scales }) {
        if (input && (!input->data || nemotron_moe_overlap(dst, input))) {
            return -1;
        }
    }
    ggml_cuda_mul_mat_vec_q_moe_sum(ctx, matrix, activation, ids, weights, scales, dst);
    return offset - 1;
}
