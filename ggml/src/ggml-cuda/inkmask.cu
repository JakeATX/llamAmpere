#include "inkmask.cuh"
#include "convert.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>

static bool ink_relmask_enabled() {
    static const bool v = [] { const char * e = getenv("GGML_CUDA_INK_MASK"); return e == nullptr || atoi(e) != 0; }();
    return v;
}

namespace {
struct ink_relmask_counter {
    std::atomic<long long> n{0};
    std::atomic<long long> matched{0};
    ~ink_relmask_counter() {
        if (n.load() > 0 || matched.load() > 0) {
            fprintf(stderr, "[INKMASK] fused rel-bias masks: %lld (matched %lld)\n", n.load(), matched.load());
        }
    }
};
}
static ink_relmask_counter g_ink_relmask_count;

static bool ink_is_view_op(ggml_op op) {
    return op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// node positions of a matched chain, relative to the GET_ROWS node
struct ink_relmask_chain {
    int n_view = 0;   // view nodes between GET_ROWS and CONT
    int cont   = 0;   // CONT
    int mcast  = -1;  // CPY F16->F32 of the mask, or -1
    int add    = 0;   // ADD
    int out    = 0;   // CPY F32->F16 (last node)
    int stage  = -1;  // -1: write the output directly; else a chain node (GET_ROWS, CONT or ADD) whose buffer holds the
                      // result first, copied to the output after (the output reuses the table, rel or mask memory)
};

static int ink_relmask_log_level() {
    static const int v = [] { const char * e = getenv("GGML_CUDA_INK_MASK_LOG"); return e ? atoi(e) : 0; }();
    return v;
}
// GGML_CUDA_INK_MASK_LOG=2: print why a GET_ROWS F32 node did not match (first 64), step = order of the checks
static std::atomic<int> g_ink_fail_logs{0};
#define INK_FAIL(step) do { if (ink_relmask_log_level() >= 2 && (step) > 2 && g_ink_fail_logs.fetch_add(1) < 64) { \
        GGML_LOG_WARN("[INKMASK] no match at node %d (%s): step %d\n", i, cgraph->nodes[i]->name, (step)); } return false; } while (0)

static bool ink_relmask_parse(const ggml_cgraph * cgraph, int i, ink_relmask_chain & ch) {
    const ggml_tensor * G = cgraph->nodes[i];
    if (G->op != GGML_OP_GET_ROWS || G->type != GGML_TYPE_F32 || G->view_src != nullptr) {
        INK_FAIL(1);
    }
    const ggml_tensor * A = G->src[0];
    const ggml_tensor * I = G->src[1];
    if (A == nullptr || I == nullptr || A->type != GGML_TYPE_F32 || I->type != GGML_TYPE_I32 ||
            A->nb[0] != sizeof(float) || I->nb[0] != sizeof(int32_t) || A->ne[3] != 1 || I->ne[2] != 1 || I->ne[3] != 1 ||
            A->ne[2] != I->ne[1] || A->ne[0] > 256 || !ggml_is_contiguous(G) ||
            A->nb[1] % sizeof(float) != 0 || A->nb[2] % sizeof(float) != 0 || I->nb[1] % sizeof(int32_t) != 0) {
        INK_FAIL(2);
    }
    const int64_t H = G->ne[0], n_kv = G->ne[1], T = G->ne[2];
    if (T > 65535 || G->ne[3] != 1) {
        INK_FAIL(3);
    }

    int j = i + 1;
    const ggml_tensor * cur = G;
    while (j < cgraph->n_nodes && ink_is_view_op(cgraph->nodes[j]->op) && cgraph->nodes[j]->src[0] == cur && j - i <= 3) {
        cur = cgraph->nodes[j];
        ++j;
    }
    ch.n_view = j - (i + 1);
    if (ch.n_view == 0 || j >= cgraph->n_nodes) {
        INK_FAIL(4);
    }
    const ggml_tensor * P = cur;
    const ggml_tensor * C = cgraph->nodes[j];
    if (C->op != GGML_OP_CONT || C->src[0] != P || C->type != GGML_TYPE_F32 || !ggml_is_contiguous(C)) {
        INK_FAIL(5);
    }
    ch.cont = j - i;

    // the permuted view must be out[k, ts, h, s] = G[h, k, ts + Ts*s]
    const int64_t Ts = P->ne[1], ns = P->ne[3];
    if (P->view_src != G || P->view_offs != 0 || P->ne[0] != n_kv || P->ne[2] != H || Ts*ns != T ||
            P->nb[0] != (size_t) H*sizeof(float) || P->nb[1] != (size_t) H*n_kv*sizeof(float) ||
            P->nb[2] != sizeof(float) || P->nb[3] != (size_t) H*n_kv*Ts*sizeof(float)) {
        INK_FAIL(6);
    }

    ++j;
    if (j >= cgraph->n_nodes) {
        INK_FAIL(7);
    }
    if (cgraph->nodes[j]->op == GGML_OP_CPY) {
        const ggml_tensor * mc = cgraph->nodes[j];
        if (mc->type != GGML_TYPE_F32 || mc->src[0] == nullptr || mc->src[0]->type != GGML_TYPE_F16 ||
                !ggml_are_same_shape(mc, mc->src[0])) {
            INK_FAIL(8);
        }
        ch.mcast = j - i;
        ++j;
        if (j >= cgraph->n_nodes) {
            INK_FAIL(9);
        }
    }
    const ggml_tensor * D = cgraph->nodes[j];
    if (D->op != GGML_OP_ADD || D->type != GGML_TYPE_F32 || !ggml_are_same_shape(D, C) || !ggml_is_contiguous(D)) {
        INK_FAIL(10);
    }
    const ggml_tensor * M = D->src[0] == C ? D->src[1] : D->src[1] == C ? D->src[0] : nullptr;
    if (M == nullptr || M == C) {
        INK_FAIL(11);
    }
    if (ch.mcast >= 0 && M != cgraph->nodes[i + ch.mcast]) {
        INK_FAIL(12);
    }
    if (ch.mcast < 0 && M->type != GGML_TYPE_F32) {
        INK_FAIL(13);
    }
    const ggml_tensor * Mr = ch.mcast >= 0 ? M->src[0] : M; // the tensor the kernel reads
    const size_t mts = ggml_type_size(Mr->type);
    if (!ggml_can_repeat(M, C) || Mr->nb[0] != mts || Mr->nb[1] % mts || Mr->nb[2] % mts || Mr->nb[3] % mts) {
        INK_FAIL(14);
    }
    ch.add = j - i;

    ++j;
    if (j >= cgraph->n_nodes) {
        INK_FAIL(15);
    }
    const ggml_tensor * O = cgraph->nodes[j];
    if (O->op != GGML_OP_CPY || O->src[0] != D || O->type != GGML_TYPE_F16 || !ggml_are_same_shape(O, D) ||
            !ggml_is_contiguous(O)) {
        INK_FAIL(16);
    }
    ch.out = j - i;

    // the output must not overlap anything the kernel reads: the table, and also the graph inputs (rel indices,
    // mask), which ggml_cuda_check_fusion_memory_ranges skips (op NONE) but gallocr may free after their last use
    const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * ap = (const char *) a->data, * bp = (const char *) b->data;
        return ap < bp + ggml_nbytes(b) && bp < ap + ggml_nbytes(a);
    };
    if (!O->data) {
        INK_FAIL(17);
    }
    // gallocr frees the table after GET_ROWS, so in prefill the output often reuses its memory. The chain's own
    // F32 intermediates (GET_ROWS, CONT, ADD: twice the output's size, dead once fused) then stage the result.
    ch.stage = -1;
    if (overlaps(O, A) || overlaps(O, I) || overlaps(O, Mr)) {
        const int cands[3] = { 0, ch.cont, ch.add };
        for (const int c : cands) {
            const ggml_tensor * S = cgraph->nodes[i + c];
            if (S->data && ggml_nbytes(S) >= ggml_nbytes(O) && !overlaps(S, A) && !overlaps(S, I) && !overlaps(S, Mr) &&
                    !overlaps(S, O)) {
                ch.stage = c;
                break;
            }
        }
        if (ch.stage < 0) {
            INK_FAIL(20);
        }
    }

    // the mask cast records itself as src[1] (ggml_cast), so its use count is 2 (itself + the ADD): checked here and
    // left out of the subgraph check, which counts only uses by later listed nodes
    if (ch.mcast >= 0) {
        const int mi = i + ch.mcast;
        const ggml_tensor * mc = cgraph->nodes[mi];
        if (mc->src[1] != mc || (mc->flags & GGML_TENSOR_FLAG_OUTPUT) || !(mc->flags & GGML_TENSOR_FLAG_COMPUTE) ||
                ggml_node_get_use_count(cgraph, mi) != 2) {
            INK_FAIL(18);
        }
    }

    enum ggml_op ops[12];
    int idxs[12];
    int count = 0;
    if (ch.out + 1 > 12) {
        INK_FAIL(19);
    }
    for (int k = 0; k <= ch.out; ++k) {
        if (k == ch.mcast) {
            continue;
        }
        idxs[count] = i + k;
        ops[count]  = cgraph->nodes[i + k]->op;
        ++count;
    }
    const int outs[1] = { i + ch.out };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, outs, 1)) { INK_FAIL(99); }
    return true;
}

int ggml_cuda_ink_relmask_match(const ggml_cgraph * cgraph, int i) {
    if (!ink_relmask_enabled()) {
        return 0;
    }
    ink_relmask_chain ch;
    if (!ink_relmask_parse(cgraph, i, ch)) {
        return 0;
    }
    g_ink_relmask_count.matched.fetch_add(1, std::memory_order_relaxed);
    return ch.out;
}

// one thread per key cell k of query row t (blockIdx.y = ts + Ts*s); loops over the heads
template <typename mask_t, bool mask_per_head>
static __global__ void k_ink_relmask(
        const float * __restrict__ A, const int64_t a_s1, const int64_t a_s2,
        const int32_t * __restrict__ I, const int64_t i_s1,
        const mask_t * __restrict__ M, const int64_t m_ne0, const int64_t m_ne1, const int64_t m_ne2, const int64_t m_ne3,
        const int64_t m_s0, const int64_t m_s1, const int64_t m_s2, const int64_t m_s3,
        half * __restrict__ O, const int H, const int64_t n_kv, const int64_t Ts) {
    const int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= n_kv) {
        return;
    }
    const int64_t t  = blockIdx.y;
    const int64_t ts = t % Ts;
    const int64_t s  = t / Ts;

    const int32_t r = I[k + t*i_s1];
    const float * a = A + r*a_s1 + t*a_s2;

    const int64_t m_base = (k % m_ne0)*m_s0 + (ts % m_ne1)*m_s1 + (s % m_ne3)*m_s3;
    float m0 = 0.0f;
    if constexpr (!mask_per_head) {
        m0 = ggml_cuda_cast<float>(M[m_base]);
    }
    half * o = O + k + n_kv*(ts + Ts*H*s);
    const int64_t o_sh = n_kv*Ts;
    for (int h = 0; h < H; ++h) {
        float m = m0;
        if constexpr (mask_per_head) {
            m = ggml_cuda_cast<float>(M[m_base + (h % m_ne2)*m_s2]);
        }
        o[h*o_sh] = ggml_cuda_cast<half>(a[h] + m);
    }
}

template <typename mask_t>
static void ink_relmask_launch_t(const ggml_tensor * A, const ggml_tensor * I, const ggml_tensor * Mr, const ggml_tensor * M,
        half * out, int64_t H, int64_t n_kv, int64_t T, int64_t Ts, cudaStream_t stream) {
    const size_t mts = ggml_type_size(Mr->type);
    const int block = 256;
    const dim3 grid((unsigned) ((n_kv + block - 1)/block), (unsigned) T, 1);
    // M (the logical mask) and Mr (what is read) have the same shape
    const bool per_head = M->ne[2] > 1;
    auto launch = [&](auto kern) {
        kern<<<grid, block, 0, stream>>>(
                (const float *) A->data, (int64_t) (A->nb[1]/sizeof(float)), (int64_t) (A->nb[2]/sizeof(float)),
                (const int32_t *) I->data, (int64_t) (I->nb[1]/sizeof(int32_t)),
                (const mask_t *) Mr->data, M->ne[0], M->ne[1], M->ne[2], M->ne[3],
                (int64_t) (Mr->nb[0]/mts), (int64_t) (Mr->nb[1]/mts), (int64_t) (Mr->nb[2]/mts), (int64_t) (Mr->nb[3]/mts),
                out, (int) H, n_kv, Ts);
    };
    if (per_head) {
        launch(k_ink_relmask<mask_t, true>);
    } else {
        launch(k_ink_relmask<mask_t, false>);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_ink_relmask_launch(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int n) {
    ink_relmask_chain ch;
    GGML_ASSERT(ink_relmask_parse(cgraph, i, ch) && ch.out == n);
    const ggml_tensor * G = cgraph->nodes[i];
    const ggml_tensor * A = G->src[0];
    const ggml_tensor * I = G->src[1];
    const ggml_tensor * C = cgraph->nodes[i + ch.cont];
    const ggml_tensor * D = cgraph->nodes[i + ch.add];
    const ggml_tensor * M = D->src[0] == C ? D->src[1] : D->src[0];
    const ggml_tensor * Mr = ch.mcast >= 0 ? M->src[0] : M;
    ggml_tensor * O = cgraph->nodes[i + ch.out];
    const ggml_tensor * P = cgraph->nodes[i + ch.n_view];

    const int64_t H = G->ne[0], n_kv = G->ne[1], T = G->ne[2], Ts = P->ne[1];
    cudaStream_t stream = ctx.stream();
    half * dst = (half *) (ch.stage >= 0 ? cgraph->nodes[i + ch.stage]->data : O->data);
    if (Mr->type == GGML_TYPE_F16) {
        ink_relmask_launch_t<half>(A, I, Mr, M, dst, H, n_kv, T, Ts, stream);
    } else {
        ink_relmask_launch_t<float>(A, I, Mr, M, dst, H, n_kv, T, Ts, stream);
    }
    if (ch.stage >= 0) {
        CUDA_CHECK(cudaMemcpyAsync(O->data, dst, ggml_nbytes(O), cudaMemcpyDeviceToDevice, stream));
    }
    // GGML_CUDA_INK_MASK_LOG=1: one line per new (n_kv, T) shape (diagnostic: shows the prompt ubatch size)
    static const bool log_shapes = [] { const char * e = getenv("GGML_CUDA_INK_MASK_LOG"); return e && atoi(e) != 0; }();
    if (log_shapes) {
        static int64_t last_kv = -1, last_t = -1;
        static int n_logged = 0;
        if ((n_kv != last_kv || T != last_t) && n_logged < 256) {
            last_kv = n_kv; last_t = T; ++n_logged;
            GGML_LOG_WARN("[INKMASK] shape n_kv %lld T %lld%s\n", (long long) n_kv, (long long) T, ch.stage >= 0 ? " staged" : "");
        }
    }
    if (g_ink_relmask_count.n.fetch_add(1) == 0) {
        GGML_LOG_WARN("[INKMASK] fused rel-bias mask on: H %lld, n_kv %lld, T %lld (%lld x %lld), mask %s%s, %d nodes\n",
                (long long) H, (long long) n_kv, (long long) T, (long long) Ts, (long long) (T/Ts), ggml_type_name(Mr->type),
                M->ne[2] > 1 ? " per head" : "", n + 1);
    }
}
