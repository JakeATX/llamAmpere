#include "nemotron-ssm.cuh"
#include "unary.cuh"
#include "ggml-backend-impl.h"

#include <cstring>

bool ggml_cuda_nemotron_ssm_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_NEMOTRON_SSM");
        return value != nullptr && strcmp(value, "1") == 0;
    }();
    return enabled;
}

namespace {

struct nemotron_ssm_match {
    ggml_tensor * concat = nullptr;
    ggml_tensor * conv = nullptr;
    ggml_tensor * conv_bias = nullptr;
    ggml_tensor * silu = nullptr;
    ggml_tensor * dt_cont = nullptr;
    ggml_tensor * dt_add = nullptr;
    ggml_tensor * scan = nullptr;
    ggml_tensor * conv_copy = nullptr;
    ggml_tensor * state_copy = nullptr;
    ggml_tensor * mul = nullptr;
    ggml_tensor * add = nullptr;
    ggml_tensor * z_cont = nullptr;
    ggml_tensor * glu = nullptr;
    int last = -1;
    std::vector<ggml_tensor *> reads;
};

static bool metadata(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

static bool shape(const ggml_tensor * t, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
    return t != nullptr && t->type == GGML_TYPE_F32 && t->ne[0] == n0 && t->ne[1] == n1 && t->ne[2] == n2 && t->ne[3] == n3;
}

static bool view(const ggml_tensor * t, const ggml_tensor * source, size_t offset) {
    return t != nullptr && t->op == GGML_OP_VIEW && t->view_src == source && t->view_offs == offset && t->nb[0] == sizeof(float);
}

static bool match_at(ggml_cgraph * graph, int first, nemotron_ssm_match & m) {
    m = {};
    ggml_tensor * concat = graph->nodes[first];
    if (concat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(concat, 0) != 0 || concat->src[0] == nullptr || concat->src[1] == nullptr) {
        return false;
    }
    const int64_t channels = concat->ne[1];
    if (!shape(concat, 4, channels) || !shape(concat->src[0], 3, channels) || !shape(concat->src[1], 1, channels) ||
        concat->src[0]->nb[0] != sizeof(float) || concat->src[0]->nb[1] % sizeof(float) != 0 ||
        concat->src[1]->nb[1] != sizeof(float)) {
        return false;
    }
    m.concat = concat;
    // A decode block is small; a longer interval contains unrelated graph work.
    for (int j = first + 1; j < graph->n_nodes && j - first < 64; ++j) {
        ggml_tensor * t = graph->nodes[j];
        if (t->op == GGML_OP_SSM_CONV && t->src[0] == concat) {
            m.conv = t;
        }
        if (t->op == GGML_OP_SSM_SCAN) {
            if (m.scan != nullptr) {
                return false;
            }
            m.scan = t;
        }
        if (t->op == GGML_OP_GLU) {
            m.glu = t;
            m.last = j;
            break;
        }
    }
    if (m.conv == nullptr || m.scan == nullptr || m.glu == nullptr || ggml_get_glu_op(m.glu) != GGML_GLU_OP_SWIGLU ||
        ggml_get_op_params_i32(m.glu, 1) != 0 || ggml_get_op_params_i32(m.scan, 0) != 1) {
        return false;
    }
    ggml_tensor * x = m.scan->src[1];
    ggml_tensor * B = m.scan->src[4];
    ggml_tensor * C = m.scan->src[5];
    ggml_tensor * state = m.scan->src[0];
    const int64_t head_dim = x->ne[0];
    const int64_t heads = x->ne[1];
    const int64_t groups = B->ne[1];
    const int64_t d_state = B->ne[0];
    if ((d_state != 96 && d_state != 128) || head_dim < 4 || head_dim > 256 || head_dim % 4 != 0 ||
        heads < 1 || heads > 1024 || groups < 1 || groups > heads || heads % groups != 0) {
        return false;
    }
    const int64_t inner = heads * head_dim;
    const int64_t state_size = inner * d_state;
    if (inner > 65536 || channels != inner + 2 * groups * d_state || !shape(x, head_dim, heads) ||
        !shape(B, d_state, groups) || !shape(C, d_state, groups) || !shape(state, d_state, head_dim, heads, state->ne[3]) ||
        state->ne[3] < 1 || ggml_nelements(m.scan) != inner + state_size || !ggml_is_contiguous(state) || !shape(m.scan->src[3], 1, heads) ||
        !ggml_is_contiguous(m.scan->src[3]) || !shape(m.conv, channels) || !shape(m.conv->src[1], 4, channels) ||
        !ggml_is_contiguous(m.conv->src[1])) {
        return false;
    }
    const ggml_tensor * ids = m.scan->src[6];
    if (ids->type != GGML_TYPE_I32 || ggml_nelements(ids) != 1 || !ggml_is_contiguous(ids)) {
        return false;
    }
    m.silu = x->view_src;
    if (m.silu == nullptr || m.silu->op != GGML_OP_UNARY || ggml_get_unary_op(m.silu) != GGML_UNARY_OP_SILU ||
        !shape(m.silu, channels) || !ggml_is_contiguous(m.silu) || !view(x, m.silu, 0) ||
        !view(B, m.silu, inner * sizeof(float)) || !view(C, m.silu, (inner + groups * d_state) * sizeof(float)) ||
        x->nb[1] != head_dim * sizeof(float) || B->nb[1] != d_state * sizeof(float) || C->nb[1] != d_state * sizeof(float)) {
        return false;
    }
    ggml_tensor * conv_out = m.silu->src[0];
    if (conv_out->op == GGML_OP_ADD && conv_out->src[0] == m.conv && shape(conv_out->src[1], channels) && ggml_is_contiguous(conv_out->src[1])) {
        m.conv_bias = conv_out;
    } else if (conv_out != m.conv) {
        return false;
    }
    m.dt_add = m.scan->src[2];
    if (m.dt_add->op != GGML_OP_ADD || !shape(m.dt_add, heads) || !shape(m.dt_add->src[1], heads) || !ggml_is_contiguous(m.dt_add->src[1])) {
        return false;
    }
    m.dt_cont = m.dt_add->src[0];
    if (m.dt_cont->op != GGML_OP_CONT || !shape(m.dt_cont, heads) || !shape(m.dt_cont->src[0], heads) ||
        m.dt_cont->src[0]->nb[0] != sizeof(float)) {
        return false;
    }
    m.z_cont = m.glu->src[0];
    m.add = m.glu->src[1];
    if (m.z_cont == nullptr || m.add == nullptr || m.z_cont->op != GGML_OP_CONT || m.add->op != GGML_OP_ADD ||
        !shape(m.z_cont, head_dim, heads) || !shape(m.z_cont->src[0], head_dim, heads) ||
        !ggml_is_contiguous_1(m.z_cont->src[0]) || !shape(m.add, head_dim, heads) ||
        !view(m.add->src[0], m.scan, 0) || !shape(m.add->src[0], head_dim, heads) ||
        m.add->src[0]->nb[1] != head_dim * sizeof(float) || !shape(m.glu, head_dim, heads) || !ggml_is_contiguous(m.glu)) {
        return false;
    }
    m.mul = m.add->src[1];
    if (m.mul->op != GGML_OP_MUL || m.mul->src[0] != x || !shape(m.mul, head_dim, heads) || !shape(m.mul->src[1], 1, heads) ||
        !ggml_is_contiguous(m.mul->src[1])) {
        return false;
    }
    std::vector<int> nodes;
    std::vector<ggml_op> ops;
    std::vector<int> outputs = { m.last };
    for (int j = first; j <= m.last; ++j) {
        ggml_tensor * t = graph->nodes[j];
        if (t->op == GGML_OP_CPY && ggml_nelements(t) > 0) {
            ggml_tensor * src = t->src[0];
            if (view(src, concat, sizeof(float)) && shape(src, 3, channels) && src->nb[1] == 4 * sizeof(float) &&
                ggml_nelements(t->src[1]) == 3 * channels && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t->src[1]) && m.conv_copy == nullptr) {
                m.conv_copy = t;
            } else if (view(src, m.scan, inner * sizeof(float)) && ggml_nelements(src) == state_size && ggml_is_contiguous(src) &&
                       ggml_nelements(t->src[1]) == state_size && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t->src[1]) && m.state_copy == nullptr) {
                m.state_copy = t;
            } else {
                return false;
            }
            outputs.push_back(j);
        } else if (metadata(t)) {
            // External views have no work to replace and keep their original lifetime.
            const ggml_tensor * base = t->view_src;
            if (base != concat && base != m.silu && base != m.scan) {
                continue;
            }
        } else if (ggml_nelements(t) == 0) {
            if (t->flags & GGML_TENSOR_FLAG_OUTPUT) {
                return false;
            }
            outputs.push_back(j);
        } else if (t != concat && t != m.conv && t != conv_out && t != m.silu && t != m.dt_cont && t != m.dt_add &&
                   t != m.scan && t != m.mul && t != m.add && t != m.z_cont && t != m.glu) {
            return false;
        }
        nodes.push_back(j);
        ops.push_back(t->op);
    }
    if (m.conv_copy == nullptr || m.state_copy == nullptr ||
        !ggml_can_fuse_subgraph_ext(graph, nodes.data(), nodes.size(), ops.data(), outputs.data(), outputs.size())) {
        return false;
    }
    m.reads = { concat->src[0], concat->src[1], m.conv->src[1], state, m.scan->src[3], m.scan->src[6],
                m.dt_cont->src[0], m.dt_add->src[1], m.z_cont->src[0], m.mul->src[1] };
    if (m.conv_bias != nullptr) {
        m.reads.push_back(m.conv_bias->src[1]);
    }
    for (ggml_tensor * input : m.reads) {
        if (input->type != GGML_TYPE_F32 && input != ids) {
            return false;
        }
        // Views of graph inputs can occur inside this interval; computed inputs must already exist.
        const ggml_tensor * base = input;
        while (metadata(base)) {
            base = base->src[0];
        }
        for (int j = first; j <= m.last; ++j) {
            if (graph->nodes[j] == base && ggml_nelements(base) > 0) {
                return false;
            }
        }
    }
    return true;
}

static bool overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t ap = (uintptr_t) a->data;
    const uintptr_t bp = (uintptr_t) b->data;
    const size_t an = ggml_nbytes(a);
    const size_t bn = ggml_nbytes(b);
    return ap < bp ? bp - ap < an : ap - bp < bn;
}

static bool memory_ok(ggml_backend_cuda_context & ctx, const nemotron_ssm_match & m) {
    const ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(ctx.device);
    const ggml_tensor * writes[] = { m.conv_copy->src[1], m.state_copy->src[1], m.glu };
    auto allocated = [buft](const ggml_tensor * t) {
        return t->data != nullptr && t->buffer != nullptr && t->buffer->buft == buft;
    };
    for (const ggml_tensor * t : m.reads) {
        if (!allocated(t)) {
            return false;
        }
        for (const ggml_tensor * dst : writes) {
            // Each warp owns the same state coordinates for every source and destination row.
            if (t == m.scan->src[0] && dst == m.state_copy->src[1]) {
                if (overlap(t, dst)) {
                    const uintptr_t base = (uintptr_t) t->data;
                    const uintptr_t out = (uintptr_t) dst->data;
                    const size_t row = t->nb[3];
                    if (out < base || (out - base) % row != 0 || out - base > ggml_nbytes(t) - row) {
                        return false;
                    }
                }
                continue;
            }
            if (overlap(t, dst)) {
                return false;
            }
        }
    }
    for (int j = 0; j < 3; ++j) {
        if (!allocated(writes[j])) {
            return false;
        }
        for (int k = 0; k < j; ++k) {
            if (overlap(writes[j], writes[k])) {
                return false;
            }
        }
    }
    return true;
}

struct nemotron_ssm_args {
    const float * conv_state;
    const float * xnew;
    const float * conv_weight;
    const float * conv_bias;
    const float * state;
    const int32_t * ids;
    const float * A;
    const float * D;
    const float * dt;
    const float * dt_bias;
    const float * z;
    float * conv_dst;
    float * state_dst;
    float * y;
    int64_t conv_stride;
    int64_t state_stride;
    int heads;
    int head_dim;
    int groups;
    int64_t z_head_stride;
};

__device__ __forceinline__ float conv_channel(const nemotron_ssm_args & a, int channel, bool save) {
    const float * old = a.conv_state + channel * a.conv_stride;
    const float * w = a.conv_weight + 4 * channel;
    const float x0 = old[0];
    const float x1 = old[1];
    const float x2 = old[2];
    const float x3 = a.xnew[channel];
    float value = 0.0f;
    value += x0 * w[0];
    value += x1 * w[1];
    value += x2 * w[2];
    value += x3 * w[3];
    value += a.conv_bias != nullptr ? a.conv_bias[channel] : 0.0f;
    if (save) {
        a.conv_dst[3 * channel + 0] = x1;
        a.conv_dst[3 * channel + 1] = x2;
        a.conv_dst[3 * channel + 2] = x3;
    }
    return ggml_cuda_op_silu_single(value);
}

template<int d_state>
__global__ void nemotron_ssm_decode(nemotron_ssm_args a) {
    ggml_cuda_pdl_lc();
    constexpr int n_warps = 4;
    constexpr int c_factor = d_state / WARP_SIZE;
    __shared__ float B[d_state];
    __shared__ float C[d_state];
    const int head = blockIdx.y;
    const int dim = blockIdx.x * n_warps + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int group = head / (a.heads / a.groups);
    const int inner = a.heads * a.head_dim;
    const bool save_BC = blockIdx.x == 0 && head == group * (a.heads / a.groups);
    ggml_cuda_pdl_sync();
    if (threadIdx.x < d_state) {
        B[threadIdx.x] = conv_channel(a, inner + group * d_state + threadIdx.x, save_BC);
        C[threadIdx.x] = conv_channel(a, inner + a.groups * d_state + group * d_state + threadIdx.x, save_BC);
    }
    __syncthreads();
    const int channel = head * a.head_dim + dim;
    float x = 0.0f;
    if (lane == 0) {
        x = conv_channel(a, channel, true);
    }
    x = __shfl_sync(0xffffffff, x, 0);
    const float dt_value = a.dt[head] + a.dt_bias[head];
    const float dt_softplus = dt_value <= 20.0f ? log1pf(expf(dt_value)) : dt_value;
    const float dA = expf(dt_softplus * a.A[head]);
    const float x_dt = x * dt_softplus;
    const float * state = a.state + (int64_t) a.ids[0] * a.state_stride + (int64_t) channel * d_state;
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < c_factor; ++j) {
        const int n = j * WARP_SIZE + lane;
        const float next = (state[n] * dA) + (B[n] * x_dt);
        sum += next * C[n];
        a.state_dst[(int64_t) channel * d_state + n] = next;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        // MUL and ADD are separate baseline kernels; keep their rounding boundary.
        const float y = __fadd_rn(sum, __fmul_rn(x, a.D[head]));
        const float z = a.z[head * a.z_head_stride + dim];
        a.y[channel] = ggml_cuda_op_silu_single(z) * y;
    }
}

} // namespace

void ggml_cuda_nemotron_ssm_prepare(ggml_cgraph * graph, ggml_backend_graph_optimize_params * params) {
    if (!ggml_cuda_nemotron_ssm_enabled()) {
        return;
    }
    nemotron_ssm_match m;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (match_at(graph, i, m)) {
            for (ggml_tensor * input : m.reads) {
                params->add_alloc_dep(params->user_data, input, m.glu);
            }
            i = m.last;
        }
    }
}

int ggml_cuda_fuse_nemotron_ssm(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int node_idx) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(ctx);
    GGML_UNUSED(graph);
    GGML_UNUSED(node_idx);
    return -1;
#else
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_nemotron_ssm_enabled() || !ctx.stream_context().concurrent_events.empty() || (cc != GGML_CUDA_CC_AMPERE && cc != 860)) {
        return -1;
    }
    nemotron_ssm_match m;
    if (!match_at(graph, node_idx, m) || !memory_ok(ctx, m)) {
        return -1;
    }
    nemotron_ssm_args a{};
    a.conv_state = (const float *) m.concat->src[0]->data;
    a.xnew = (const float *) m.concat->src[1]->data;
    a.conv_weight = (const float *) m.conv->src[1]->data;
    a.conv_bias = m.conv_bias != nullptr ? (const float *) m.conv_bias->src[1]->data : nullptr;
    a.state = (const float *) m.scan->src[0]->data;
    a.ids = (const int32_t *) m.scan->src[6]->data;
    a.A = (const float *) m.scan->src[3]->data;
    a.D = (const float *) m.mul->src[1]->data;
    a.dt = (const float *) m.dt_cont->src[0]->data;
    a.dt_bias = (const float *) m.dt_add->src[1]->data;
    a.z = (const float *) m.z_cont->src[0]->data;
    a.conv_dst = (float *) m.conv_copy->src[1]->data;
    a.state_dst = (float *) m.state_copy->src[1]->data;
    a.y = (float *) m.glu->data;
    a.conv_stride = m.concat->src[0]->nb[1] / sizeof(float);
    a.state_stride = m.scan->src[0]->nb[3] / sizeof(float);
    a.heads = m.scan->src[1]->ne[1];
    a.head_dim = m.scan->src[1]->ne[0];
    a.groups = m.scan->src[4]->ne[1];
    a.z_head_stride = m.z_cont->src[0]->nb[1] / sizeof(float);
    const dim3 blocks(a.head_dim / 4, a.heads, 1);
    const ggml_cuda_kernel_launch_params launch(blocks, 4 * WARP_SIZE, 0, ctx.stream());
    if (m.scan->src[0]->ne[0] == 96) {
        ggml_cuda_kernel_launch(nemotron_ssm_decode<96>, launch, a);
    } else {
        ggml_cuda_kernel_launch(nemotron_ssm_decode<128>, launch, a);
    }
    return m.last - node_idx;
#endif
}
