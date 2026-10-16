#include "autofuse.cuh"
#include "common.cuh"
#include "convert.cuh"
#include "unary.cuh"

#include <climits>
#include <unordered_set>

namespace {

constexpr int block_size = 256;
constexpr int min_elements = 256;

struct tensor_arg {
    const char * data;
    int64_t ne[4];
    size_t nb[4];
    ggml_type type;
    bool contiguous;
    int broadcast;
    uint3 div[4];
};

struct instruction {
    tensor_arg other;
    ggml_op op;
    int subop;
    ggml_type type;
    bool lhs;
    float p0, p1;
};

struct program {
    tensor_arg head;
    tensor_arg dst;
    instruction ops[GGML_CUDA_AUTOFUSE_MAX_OPS];
    int length;
};

static_assert(sizeof(program) < 4096, "AUTOFUSE kernel arguments must fit Ampere parameter memory");

static bool scalar_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static bool tensor_layout(const ggml_tensor * t) {
    if (!t || !scalar_type(t->type)) {
        return false;
    }
    // Check the byte span and reject overlapping elements, including zero strides.
    size_t span = ggml_type_size(t->type);
    int dims[4] = { 0, 1, 2, 3 };
    std::sort(dims, dims + 4, [t](int a, int b) { return t->nb[a] < t->nb[b]; });
    int64_t n = 1;
    for (int k = 0; k < 4; ++k) {
        const int d = dims[k];
        if (t->ne[d] <= 0 || t->ne[d] > INT_MAX / n || t->nb[d] % ggml_type_size(t->type)) {
            return false;
        }
        n *= t->ne[d];
        if (t->ne[d] > 1) {
            if (t->nb[d] < span || size_t(t->ne[d] - 1) > (SIZE_MAX - span) / t->nb[d]) {
                return false;
            }
            span += size_t(t->ne[d] - 1) * t->nb[d];
        }
    }
    return true;
}

static bool binary(const ggml_tensor * t) {
    return t->op == GGML_OP_ADD || t->op == GGML_OP_SUB || t->op == GGML_OP_MUL ||
           t->op == GGML_OP_DIV || t->op == GGML_OP_GLU;
}

static bool eligible(const ggml_tensor * t) {
    if (!tensor_layout(t) || !tensor_layout(t->src[0]) || t->view_src ||
        !(t->flags & GGML_TENSOR_FLAG_COMPUTE) || (t->flags & GGML_TENSOR_FLAG_INPUT) ||
        !ggml_are_same_shape(t, t->src[0])) {
        return false;
    }
    const ggml_tensor * a = t->src[0];
    const ggml_tensor * b = t->src[1];
    if (binary(t)) {
        if (!tensor_layout(b)) {
            return false;
        }
        for (int d = 0; d < 4; ++d) {
            if (t->ne[d] % b->ne[d]) {
                return false;
            }
        }
        if (t->op == GGML_OP_GLU) {
            if (!ggml_are_same_shape(t, b) || t->type != a->type || t->type != b->type ||
                !ggml_is_contiguous_1(a) || !ggml_is_contiguous_1(b) || !ggml_is_contiguous(t)) {
                return false;
            }
            switch (ggml_get_glu_op(t)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK: return true;
                default: return false;
            }
        }
        if (a->type == GGML_TYPE_BF16 || b->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_BF16) {
            return a->type == GGML_TYPE_BF16 && t->type == GGML_TYPE_BF16 &&
                (b->type == GGML_TYPE_BF16 || b->type == GGML_TYPE_F32);
        }
        return (a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && t->type == GGML_TYPE_F32) ||
               (a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16 && t->type == GGML_TYPE_F16) ||
               (a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F32 &&
                (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32));
    }
    if (t->op == GGML_OP_CPY) {
        // ggml_cast has a fresh destination and a self-reference. State copies do not.
        return b == t && a->type != t->type;
    }
    if (a->type != t->type) {
        return false;
    }
    if (t->op == GGML_OP_CLAMP) {
        return t->type != GGML_TYPE_BF16 && ggml_is_contiguous_rows(a) && ggml_is_contiguous_rows(t);
    }
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(t)) {
        return false;
    }
    switch (t->op) {
        case GGML_OP_SCALE: return t->type != GGML_TYPE_F16;
        case GGML_OP_SQR:
        case GGML_OP_SQRT: return true;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_SOFTPLUS: return true;
                default: return false;
            }
        default: return false;
    }
}

static void tensor_signature(std::vector<uint64_t> & key, const ggml_tensor * t) {
    key.push_back(uintptr_t(t));
    key.push_back(t->op);
    key.push_back(t->type);
    key.push_back(t->flags);
    key.push_back(uintptr_t(t->view_src));
    for (int d = 0; d < 4; ++d) {
        key.push_back(t->ne[d]);
        key.push_back(t->nb[d]);
    }
    for (int k = 0; k < GGML_MAX_OP_PARAMS / int(sizeof(int32_t)); ++k) {
        key.push_back(uint32_t(t->op_params[k]));
    }
    for (const auto * src : t->src) {
        key.push_back(uintptr_t(src));
    }
}

static std::vector<const ggml_tensor *> graph_tensors(const ggml_cgraph * graph) {
    std::vector<const ggml_tensor *> tensors;
    std::unordered_set<const ggml_tensor *> seen;
    auto add = [&](const ggml_tensor * t) {
        if (t && seen.insert(t).second) {
            tensors.push_back(t);
        }
    };
    for (int i = 0; i < graph->n_nodes; ++i) { add(graph->nodes[i]); }
    for (int i = 0; i < graph->n_leafs; ++i) { add(graph->leafs[i]); }
    for (int i = 0; i < graph->n_nodes; ++i) {
        for (const auto * src : graph->nodes[i]->src) { add(src); }
    }
    for (size_t i = 0; i < tensors.size(); ++i) { add(tensors[i]->view_src); }
    return tensors;
}

static bool on_device(const ggml_tensor * t, int device) {
    return t->data && t->buffer && ggml_backend_buffer_get_type(t->buffer) == ggml_backend_cuda_buffer_type(device);
}

static bool overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t x = uintptr_t(a->data), y = uintptr_t(b->data);
    return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
}

static tensor_arg argument(const ggml_tensor * t, const ggml_tensor * shape) {
    tensor_arg arg = {};
    if (t) {
        arg.data = static_cast<const char *>(t->data);
        arg.type = t->type;
        arg.contiguous = ggml_is_contiguous(t);
        memcpy(arg.ne, t->ne, sizeof(arg.ne));
        memcpy(arg.nb, t->nb, sizeof(arg.nb));
        arg.broadcast = ggml_are_same_shape(t, shape) ? 0 : ggml_nelements(t) == 1 ? 1 :
            (t->ne[0] == shape->ne[0] && ggml_nrows(t) == 1) ? 2 : 3;
        for (int d = 0; d < 4; ++d) { arg.div[d] = init_fastdiv_values(t->ne[d]); }
    }
    return arg;
}

__device__ __forceinline__ float load(const tensor_arg & t, int64_t i, const uint32_t * coord) {
    size_t offset = i * (t.type == GGML_TYPE_F32 ? sizeof(float) : sizeof(half));
    if (!t.contiguous) {
        offset = 0;
        for (int d = 0; d < 4; ++d) { offset += coord[d] * t.nb[d]; }
    }
    const char * ptr = t.data + offset;
    switch (t.type) {
        case GGML_TYPE_F16:  return ggml_cuda_cast<float>(*reinterpret_cast<const half *>(ptr));
        case GGML_TYPE_BF16: return ggml_cuda_cast<float>(*reinterpret_cast<const nv_bfloat16 *>(ptr));
        default:            return *reinterpret_cast<const float *>(ptr);
    }
}

__device__ __forceinline__ float round_value(float x, ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:  return ggml_cuda_cast<float>(ggml_cuda_cast<half>(x));
        case GGML_TYPE_BF16: return ggml_cuda_cast<float>(ggml_cuda_cast<nv_bfloat16>(x));
        default:            return x;
    }
}

// A call boundary prevents contraction or reassociation across graph nodes under -use_fast_math.
static __device__ __noinline__ float evaluate(float v, float other, ggml_op opcode, int subop, ggml_type type, bool lhs, float p0, float p1) {
    const float a = lhs ? v : other;
    const float b = lhs ? other : v;
    switch (opcode) {
        case GGML_OP_ADD:   v = a + b; break;
        case GGML_OP_SUB:   v = a - b; break;
        case GGML_OP_MUL:   v = a * b; break;
        case GGML_OP_DIV:   v = a / b; break;
        case GGML_OP_SCALE: v = p0 * v + p1; break;
        case GGML_OP_SQR:   v = v * v; break;
        case GGML_OP_SQRT:  v = sqrtf(v); break;
        case GGML_OP_CLAMP: v = fminf(fmaxf(v, round_value(p0, type)), round_value(p1, type)); break;
        case GGML_OP_CPY:   break;
        case GGML_OP_GLU:
            switch (subop) {
                case GGML_GLU_OP_REGLU:       v = fmaxf(a, 0.0f) * b; break;
                case GGML_GLU_OP_GEGLU:       v = ggml_cuda_op_gelu_single(a) * b; break;
                case GGML_GLU_OP_SWIGLU:      v = ggml_cuda_op_silu_single(a) * b; break;
                case GGML_GLU_OP_GEGLU_ERF:   v = (0.5f*a*(1.0f + erff(a*0.70710678118654752440084436210484f))) * b; break;
                case GGML_GLU_OP_GEGLU_QUICK: v = (a * (1.0f / (1.0f + expf(-1.702f * a)))) * b; break;
            }
            break;
        case GGML_OP_UNARY:
            switch (subop) {
                case GGML_UNARY_OP_ABS:      v = fabsf(v); break;
                case GGML_UNARY_OP_NEG:      v = -v; break;
                case GGML_UNARY_OP_RELU:     v = fmaxf(v, 0.0f); break;
                case GGML_UNARY_OP_SILU:     v = ggml_cuda_op_silu_single(v); break;
                case GGML_UNARY_OP_GELU:     v = ggml_cuda_op_gelu_single(v); break;
                case GGML_UNARY_OP_SIGMOID:  v = 1.0f / (1.0f + expf(-v)); break;
                case GGML_UNARY_OP_EXP:      v = expf(v); break;
                case GGML_UNARY_OP_TANH:     v = tanhf(v); break;
                case GGML_UNARY_OP_SOFTPLUS: v = v > 20.0f ? v : logf(1.0f + expf(v)); break;
            }
            break;
        default: break;
    }
    return round_value(v, type);
}

static __global__ void interpret(const program p, const int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * block_size + threadIdx.x;
    if (i >= n) { return; }
    uint32_t coord[4], rest = i;
    for (int d = 0; d < 4; ++d) {
        const uint2 dm = fast_div_modulo(rest, p.dst.div[d]);
        coord[d] = dm.y;
        rest = dm.x;
    }
    float v = load(p.head, i, coord);
#pragma unroll 1
    for (int k = 0; k < p.length; ++k) {
        const instruction & op = p.ops[k];
        float other = 0.0f;
        if (op.other.data) {
            uint32_t bc[4] = {};
            int64_t index = 0;
            if (op.other.broadcast == 0) {
                index = i;
                for (int d = 0; d < 4; ++d) { bc[d] = coord[d]; }
            } else if (op.other.broadcast == 2) {
                index = bc[0] = coord[0];
            } else if (op.other.broadcast == 3) {
                int64_t stride = 1;
                for (int d = 0; d < 4; ++d) {
                    bc[d] = fastmodulo(coord[d], op.other.div[d]);
                    index += bc[d] * stride;
                    stride *= op.other.ne[d];
                }
            }
            other = load(op.other, index, bc);
        }
        v = evaluate(v, other, op.op, op.subop, op.type, op.lhs, op.p0, op.p1);
    }
    size_t offset = i * (p.dst.type == GGML_TYPE_F32 ? sizeof(float) : sizeof(half));
    if (!p.dst.contiguous) {
        offset = 0;
        for (int d = 0; d < 4; ++d) { offset += coord[d] * p.dst.nb[d]; }
    }
    char * out = const_cast<char *>(p.dst.data) + offset;
    switch (p.dst.type) {
        case GGML_TYPE_F16:  *reinterpret_cast<half *>(out) = ggml_cuda_cast<half>(v); break;
        case GGML_TYPE_BF16: *reinterpret_cast<nv_bfloat16 *>(out) = ggml_cuda_cast<nv_bfloat16>(v); break;
        default:            *reinterpret_cast<float *>(out) = v; break;
    }
}

} // namespace

bool ggml_cuda_autofuse_enabled() {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return false;
#else
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_AUTOFUSE");
        return value && strcmp(value, "1") == 0;
    }();
    return enabled;
#endif
}

ggml_cuda_autofuse_plan * ggml_cuda_autofuse_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph) {
    if (graph->n_nodes == 0) { return nullptr; }
    if (graph->uid != 0) {
        for (auto & plan : ctx.autofuse_plans) {
            if (plan.uid == graph->uid) { return &plan; }
        }
    }
    const auto tensors = graph_tensors(graph);
    std::vector<uint64_t> topology;
    if (graph->uid == 0) {
        topology.push_back(graph->n_nodes);
        topology.push_back(graph->n_leafs);
        for (int i = 0; i < graph->n_nodes; ++i) { topology.push_back(uintptr_t(graph->nodes[i])); }
        for (int i = 0; i < graph->n_leafs; ++i) { topology.push_back(uintptr_t(graph->leafs[i])); }
        for (const auto * t : tensors) { tensor_signature(topology, t); }
        for (int i = 0; i < graph->n_nodes; ++i) { topology.push_back(ggml_node_get_use_count(graph, i)); }
    }
    uint64_t hash = 0xcbf29ce484222325ull;
    for (uint64_t value : topology) { hash = (hash ^ value) * 0x100000001b3ull; }
    if (graph->uid == 0) {
        for (auto & plan : ctx.autofuse_plans) {
            if (plan.uid == 0 && plan.topology_hash == hash && plan.topology == topology) { return &plan; }
        }
    }
    if (ctx.autofuse_plans.size() >= 64) { ctx.autofuse_plans.clear(); }
    ctx.autofuse_plans.emplace_back();
    auto & plan = ctx.autofuse_plans.back();
    plan.uid = graph->uid;
    plan.topology_hash = hash;
    plan.topology = std::move(topology);
    plan.lengths.resize(graph->n_nodes);
    std::unordered_map<const ggml_tensor *, int> uses;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto * t = graph->nodes[i];
        for (const auto * src : t->src) {
            if (src && src != t) { ++uses[src]; }
        }
    }
    for (const auto * t : tensors) {
        if (t->view_src) { ++uses[t->view_src]; }
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * first = graph->nodes[i];
        if (!eligible(first) || ggml_nelements(first) < min_elements) { continue; }
        int length = 1;
        while (i + length < graph->n_nodes && length < GGML_CUDA_AUTOFUSE_MAX_OPS) {
            const ggml_tensor * prev = graph->nodes[i + length - 1];
            const ggml_tensor * next = graph->nodes[i + length];
            const int self_uses = prev->op == GGML_OP_CPY && prev->src[1] == prev ? 1 : 0;
            if (uses[prev] != 1 || ggml_node_get_use_count(graph, i + length - 1) != 1 + self_uses ||
                (prev->flags & GGML_TENSOR_FLAG_OUTPUT) || !eligible(next) ||
                !ggml_are_same_shape(first, next)) { break; }
            const ggml_tensor * other = nullptr;
            if (next->src[0] == prev) {
                other = binary(next) ? next->src[1] : nullptr;
            } else if (binary(next) && next->src[1] == prev) {
                other = next->src[0];
            } else { break; }
            bool internal = false;
            for (int j = i; j < i + length; ++j) { internal |= other == graph->nodes[j]; }
            if (internal) { break; }
            ++length;
        }
        if (length >= 2) { plan.lengths[i] = length; }
    }
    return &plan;
}

int ggml_cuda_autofuse_launch(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, int i, ggml_cuda_autofuse_plan & plan) {
    const int length = plan.lengths[i];
    if (!length) { return 0; }
    program p = {};
    p.length = length;
    const ggml_tensor * out = graph->nodes[i + length - 1];
    const ggml_tensor * head = graph->nodes[i]->src[0];
    if (!on_device(out, ctx.device) || !on_device(head, ctx.device)) { return 0; }
    p.head = argument(head, out);
    p.dst = argument(out, out);
    for (int k = 0; k < length; ++k) {
        const ggml_tensor * t = graph->nodes[i + k];
        const ggml_tensor * prev = k ? graph->nodes[i + k - 1] : head;
        const bool lhs = t->src[0] == prev;
        const ggml_tensor * other = binary(t) ? t->src[lhs ? 1 : 0] : nullptr;
        if (!on_device(t, ctx.device) || (other && !on_device(other, ctx.device))) { return 0; }
        // No skipped write may change an operand that a later instruction loads from memory.
        const ggml_tensor * read = k ? other : head;
        for (int j = 0; read && j < k; ++j) {
            if (overlap(read, graph->nodes[i + j])) { return 0; }
        }
        if (other && overlap(out, other) && (other->data != out->data || !ggml_are_same_layout(out, other))) { return 0; }
        auto & op = p.ops[k];
        op.other = argument(other, out);
        op.op = t->op;
        op.subop = t->op == GGML_OP_UNARY ? int(ggml_get_unary_op(t)) : t->op == GGML_OP_GLU ? int(ggml_get_glu_op(t)) : 0;
        op.type = t->type;
        op.lhs = lhs;
        op.p0 = ggml_get_op_params_f32(t, 0);
        op.p1 = ggml_get_op_params_f32(t, 1);
    }
    if (overlap(out, head) && (head->data != out->data || !ggml_are_same_layout(out, head))) { return 0; }
    const int64_t n = ggml_nelements(out);
    interpret<<<(n + block_size - 1) / block_size, block_size, 0, ctx.stream()>>>(p, n);
    CUDA_CHECK(cudaGetLastError());
    ++ctx.fusion_stats.autofuse;
    if (!plan.reported) { ++plan.histogram[length]; }
    return length - 1;
}

void ggml_cuda_autofuse_report(ggml_backend_cuda_context & ctx, ggml_cuda_autofuse_plan & plan) {
    if (plan.reported) { return; }
    int64_t count = 0;
    for (int k = 2; k <= GGML_CUDA_AUTOFUSE_MAX_OPS; ++k) { count += plan.histogram[k]; }
    GGML_LOG_INFO("CUDA%d AUTOFUSE graph uid=%llu key=%llx: %lld chains; lengths 2..8 = [%lld,%lld,%lld,%lld,%lld,%lld,%lld]\n",
        ctx.device, (unsigned long long) plan.uid, (unsigned long long) (plan.uid ? plan.uid : plan.topology_hash), (long long) count,
        (long long) plan.histogram[2], (long long) plan.histogram[3], (long long) plan.histogram[4],
        (long long) plan.histogram[5], (long long) plan.histogram[6], (long long) plan.histogram[7], (long long) plan.histogram[8]);
    plan.reported = true;
}
