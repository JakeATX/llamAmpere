#include "models.h"

#include "llama-impl.h"
#include "llama-memory-recurrent.h"

// utility to get one slice from the third dimension
// input dim:  [x, y, c, b]
// output dim: [x, y, 1, b]
static ggml_tensor * get_slice_2d(ggml_context * ctx0, ggml_tensor * t, int64_t c) {
    return ggml_view_4d(ctx0, t, t->ne[0], t->ne[1], 1, t->ne[3],
        t->nb[1], t->nb[2], t->nb[3], t->nb[2] * c);
}

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_chunking(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    const bool kda = (g->ne[0] == S_k && g->ne[1] == H_k);

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf(S_k);

    q = ggml_scale(ctx0, q, scale);

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(b, "b_in", il);
    cb(g, "g_in", il);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]
    g = ggml_permute(ctx0, g, 0, 2, 1, 3); // [g_0, n_tokens, H_v, n_seqs]
    b = ggml_permute(ctx0, b, 0, 2, 1, 3); // [  1, n_tokens, H_v, n_seqs]

    const int CS = kda ? 16 : 64; // chunk size

    const int pad = (CS - n_tokens % CS) % CS;
    const int n_chunks = (n_tokens + pad) / CS;

    q = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = ggml_pad(ctx0, v, 0, pad, 0, 0);
    g = ggml_pad(ctx0, g, 0, pad, 0, 0);
    b = ggml_pad(ctx0, b, 0, pad, 0, 0);

    ggml_tensor * v_b = ggml_mul(ctx0, v, b);
    ggml_tensor * k_b = ggml_mul(ctx0, k, b);

    cb(v_b, "v_b", il);
    cb(k_b, "k_b", il);

    q   = ggml_reshape_4d(ctx0, q,   S_k, CS, n_chunks, H_k * n_seqs);
    k   = ggml_reshape_4d(ctx0, k,   S_k, CS, n_chunks, H_k * n_seqs);
    k_b = ggml_reshape_4d(ctx0, k_b, S_k, CS, n_chunks, H_v * n_seqs);
    v   = ggml_reshape_4d(ctx0, v,   S_v, CS, n_chunks, H_v * n_seqs);
    v_b = ggml_reshape_4d(ctx0, v_b, S_v, CS, n_chunks, H_v * n_seqs);

    g = ggml_reshape_4d(ctx0, g, g->ne[0], CS, n_chunks, H_v * n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        CS, n_chunks, H_v * n_seqs);

    // [CS, g_0, n_chunks, H_v * n_seqs]
    // TODO: extend ggml_cumsum with axis parameter to avoid transpose
    ggml_tensor * g_cs = ggml_cumsum(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, g)));
    cb(g_cs, "g_cs", il);

    ggml_tensor * kb = nullptr;
    ggml_tensor * kq = nullptr;
    if (kda) {
        const int64_t CHB = n_chunks * H_k * n_seqs;

        ggml_tensor * g_cs_i = ggml_reshape_4d(ctx0, g_cs, CS, 1, S_k, CHB);  // [chunk_size, 1, S_k, CHB]
        ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, S_k, CHB);  // [1, chunk_size, S_k, CHB]

        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, S_k, CHB);  // [1, chunk_size, S_k, CHB] -> [chunk_size, chunk_size, S_k, CHB]

        // decay_mask [chunk_size,chunk_size,S_k,CHB]
        ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);
        cb(decay_mask, "decay_mask", il);

        // decay_mask [S_k,BT_j,BT_i,CHB] *Note* second and third chunk_sizes are switched
        decay_mask = ggml_cont_4d(ctx0, ggml_permute(ctx0, decay_mask, 2, 1, 0, 3), S_k, CS, CS, CHB);

        ggml_tensor * k_b_i = ggml_reshape_4d(ctx0, k_b, S_k, CS,  1, CHB);
        ggml_tensor * k_j   = ggml_reshape_4d(ctx0, k,   S_k,  1, CS, CHB);
        ggml_tensor * q_i   = ggml_reshape_4d(ctx0, q,   S_k, CS,  1, CHB);

        ggml_tensor * decay_k_b_i = ggml_mul(ctx0, decay_mask, k_b_i);
        ggml_tensor * decay_q_i   = ggml_mul(ctx0, decay_mask, q_i);

        // decay_k_b_i [S,BT,BT,CHB] @ k_j [S,1,BT,CHB] = Akk [BT,1,BT,CHB]
        kb = ggml_mul_mat(ctx0, decay_k_b_i, k_j);
        kq = ggml_mul_mat(ctx0, decay_q_i,   k_j);

        kb = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kb, CS, CS, n_chunks, H_v * n_seqs)));
        kq = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kq, CS, CS, n_chunks, H_v * n_seqs)));
    } else {
        ggml_tensor * g_cs_i = g_cs;
        ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, n_chunks, H_v * n_seqs);

        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, n_chunks, H_v * n_seqs);

        // [CS, CS, n_chunks, H_v * n_seqs]
        ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);
        cb(decay_mask, "decay_mask", il);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kb = ggml_mul_mat(ctx0, k,  k_b);
        kb = ggml_mul    (ctx0, kb, decay_mask);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_mul(ctx0, kq, decay_mask);
    }

    kq = ggml_tri(ctx0, kq, GGML_TRI_TYPE_LOWER_DIAG);
    cb(kq, "kq", il);

    // [CS, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * attn;
    attn = ggml_tri(ctx0, kb, GGML_TRI_TYPE_LOWER);
    cb(attn, "attn", il);

    ggml_tensor * identity;
    identity = ggml_view_1d(ctx0, attn, CS, 0);
    identity = ggml_fill   (ctx0, identity, 1.0f);
    identity = ggml_diag   (ctx0, identity);

    ggml_tensor * lhs = ggml_add(ctx0, attn, identity);
    cb(lhs, "dnet_add_ch_lhs", il);

    attn = ggml_neg(ctx0, attn);
    cb(attn, "attn_pre_solve", il);

    ggml_tensor * lin_solve = ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn = ggml_add(ctx0, lin_solve, identity);
    cb(attn, "dnet_add_ch_attn_solved", il); // [CS, CS, n_chunks, H_k * n_seqs]

    // [S_v, CS, n_chunks, H_v * n_seqs]
    v = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_b)), attn);

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_exp = ggml_exp(ctx0, g_cs);

    k_b = ggml_cont(ctx0, ggml_transpose(ctx0, k_b));

    // [CS, S_k, n_chunks, H_k * n_seqs]
    ggml_tensor * kbg = ggml_mul(ctx0, k_b, g_exp);
    cb(kbg, "k_beta_g_exp", il);

    // [S_k, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * k_cd = ggml_mul_mat(ctx0, kbg, attn);
    cb(k_cd, "k_cumdecay", il);

    // [1, CS, n_chunks, H_k * n_seqs] KDA: [S_k, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * g_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_exp));
    ggml_tensor * q_g_exp = ggml_mul(ctx0, q, g_exp_t);

    // vectorized calculation of key_gdiff
    // improved from the chunked version:
    //   g_last = torch.clamp(g_cum[:, :, -1], max=50.0).exp().unsqueeze(-1).unsqueeze(-1)
    //   g_diff = torch.clamp(g_cum[:, :, -1:] - g_cum, max=50.0).exp()
    //   key_gdiff = key * g_diff.unsqueeze(-1)
    //   kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
    //   last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew

    // get last element in g_cumsum along CS dimension (ne0)
    // example: [[x, y, z, ..., last], ...] -> [[last], ...]
    // [1, 1, n_chunks, H_v * n_seqs] KDA: [1, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_last = ggml_view_4d(ctx0, g_cs, 1, g_cs->ne[1], g_cs->ne[2], g_cs->ne[3],
            g_cs->nb[1],
            g_cs->nb[2],
            g_cs->nb[3],
            ggml_row_size(g_cs->type, g_cs->ne[0] - 1));
    cb(g_last, "g_last", il);

    // TODO: remove this cont when CUDA supports non-cont unary ops
    g_last = ggml_cont(ctx0, g_last);

    // [1, 1, n_chunks, H_v * n_seqs] KDA: [S_k, 1, n_chunks, H_v * n_seqs]
    ggml_tensor * g_last_exp_t = ggml_transpose(ctx0, ggml_exp(ctx0, g_last));
    cb(g_last_exp_t, "g_last_exp_t", il);

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_diff = ggml_neg(ctx0, ggml_sub(ctx0, g_cs, g_last));
    cb(g_diff, "g_diff", il);

    ggml_tensor * g_diff_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_exp(ctx0, g_diff)));

    // [S_k, CS, n_chunks, H_v * n_seqs]
    ggml_tensor * kg = ggml_mul(ctx0, k, g_diff_exp_t);
    cb(kg, "key_gdiff", il);

    // [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * kg_t = ggml_cont(ctx0, ggml_transpose(ctx0, kg));
    cb(kg_t, "key_gdiff_t", il);

    s = ggml_reshape_4d(ctx0, s, S_v, S_v, 1, H_v * n_seqs);
    cb(s, "dnet_add_ch_state", il);

    // [CS, S_v, n_chunks, H_v * n_seqs]
    ggml_tensor * v_t = ggml_cont(ctx0, ggml_transpose(ctx0, v));

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        ggml_tensor * ch_k_cd    = get_slice_2d(ctx0, k_cd,    chunk); // [S_k,  CS, 1, H_k * n_seqs]
        ggml_tensor * ch_v_t     = get_slice_2d(ctx0, v_t,     chunk); // [ CS, S_v, 1, H_v * n_seqs]
        ggml_tensor * ch_kq      = get_slice_2d(ctx0, kq,      chunk); // [ CS,  CS, 1, H_k * n_seqs]
        ggml_tensor * ch_q_g_exp = get_slice_2d(ctx0, q_g_exp, chunk); // [S_k,  CS, 1, H_k * n_seqs]
        ggml_tensor * ch_kg_t    = get_slice_2d(ctx0, kg_t,    chunk); // [ CS, S_k, 1, H_v * n_seqs]

        // [CS, S_v, 1, H_v * n_seqs]
        ggml_tensor * v_t_p = ggml_mul_mat(ctx0, ch_k_cd, s);
        cb(v_t_p, "v_prime", il);

        // [CS, S_v, 1, H_v * n_seqs]
        ggml_tensor * v_t_new = ggml_sub(ctx0, ch_v_t, v_t_p);
        cb(v_t_new, "v_t_new", il);

        // [S_v, CS, 1, H_v * n_seqs]
        ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_t_new, ch_kq);
        cb(v_attn, "v_attn", il);

        // [S_v, CS, 1, H_v * n_seqs]
        ggml_tensor * attn_inter = ggml_mul_mat(ctx0, s, ch_q_g_exp);
        cb(attn_inter, "attn_inter", il);

        // [S_v, CS, 1, H_v * n_seqs]
        ggml_tensor * o_ch = ggml_add(ctx0, attn_inter, v_attn);
        cb(o_ch, "dnet_add_ch_attn_out", il);

        v = ggml_set_inplace(ctx0, v, o_ch, v->nb[1], v->nb[2], v->nb[3], chunk * v->nb[2]);

        // kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
        // TODO: head broadcast might not work here - probably will need a transpose
        ggml_tensor * kgv = ggml_mul_mat(ctx0, ch_kg_t, v_t_new); // [S_k, S_v, 1, H_k * n_seqs]

        // last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew
        ggml_tensor * ch_g_last_exp_t = get_slice_2d(ctx0, g_last_exp_t, chunk);

        s = ggml_mul(ctx0, s, ch_g_last_exp_t);
        s = ggml_add(ctx0, s, kgv);
        cb(s, "dnet_add_ch_state", il);
    }

    // truncate padded tokens
    ggml_tensor * o = ggml_view_4d(ctx0, v,
            S_v, n_tokens, H_v, n_seqs,
            ggml_row_size(v->type, S_v),
            ggml_row_size(v->type, S_v * CS * n_chunks),
            ggml_row_size(v->type, S_v * CS * n_chunks * H_v), 0);
    o = ggml_permute  (ctx0, o, 0, 2, 1, 3); // [S_v, H_v, n_tokens, n_seqs]
    s = ggml_reshape_4d(ctx0, s, S_v, S_v, H_v, n_seqs);
    cb(s, "output_state", il);

    return {o, s};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_autoregressive(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b, // beta
        ggml_tensor * s, // state
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(n_tokens == 1);

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf(S_k);

    q = ggml_scale(ctx0, q, scale);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(b, "b_in", il);
    cb(g, "g_in", il);

    // GDA: [1,  1,  H_v, n_seqs]
    // KDA: [1, S_k, H_v, n_seqs]
    g = ggml_reshape_4d(ctx0, g, 1, g->ne[0], H_v, n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        1, H_v, n_seqs);

    // [S_v, S_v, H_v, n_seqs]
    g = ggml_exp(ctx0, g);
    s = ggml_mul(ctx0, s, g);

    // [1, S_v, H_v, n_seqs]
    ggml_tensor * sk;
    sk = ggml_mul     (ctx0, s, k);
    sk = ggml_sum_rows(ctx0, sk);

    // [S_v, 1, H_v, n_seqs]
    ggml_tensor * d;
    d = ggml_sub(ctx0, v, ggml_transpose(ctx0, sk));
    d = ggml_mul(ctx0, d, b);

    // [1, S_v, H_v, n_seqs]
    ggml_tensor * d_t;
    d_t = ggml_transpose(ctx0, d);

    // [S_v, S_v, H_v, n_seqs]
    ggml_tensor * kd;
    k  = ggml_repeat(ctx0, k, s);
    kd = ggml_mul   (ctx0, k, d_t);

    s = ggml_add(ctx0, s, kd);

    cb(s, "dnet_add_ar_state", il);

    ggml_tensor * s_q = ggml_mul     (ctx0, s, q);
    ggml_tensor * o   = ggml_sum_rows(ctx0, s_q);

    o = ggml_permute  (ctx0, o, 2, 0, 1, 3); // [S_v, H_v, n_tokens, n_seqs]

    return {o, s};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_fused(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    // K=1: output carries the final state only. state s is 4D [S_v, S_v, H_v, n_seqs].
    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*K=*/1, /*emit_mode=*/0);
    if (n_tokens == 1) {
        res->add_fused_node({LLM_FUSED_OP_GDN_AR, result, il});
    } else {
        res->add_fused_node({LLM_FUSED_OP_GDN_CH, result, il});
    }

    ggml_tensor * output = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * S_v),
            ggml_row_size(result->type, S_v * S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs));

    return {output, new_state};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t n_seq_tokens = q->ne[2];

    if (n_seq_tokens == 1) {
        if (cparams.fused_gdn_ar) {
            return build_delta_net_fused(q, k, v, g, b, s, il);
        }
        return build_delta_net_autoregressive(q, k, v, g, b, s, il);
    }

    if (cparams.fused_gdn_ch) {
        return build_delta_net_fused(q, k, v, g, b, s, il);
    }

    return build_delta_net_chunking(q, k, v, g, b, s, il);
}

// [TAG_RECURRENT_ROLLBACK_SHIFT] Move the older snapshot groups of the ubatch's sequences back by
// n_seq_tokens so that group g keeps meaning "the state g tokens behind the head" after a ubatch
// shorter than the group count K. The op writes only the newest min(n, K) groups (ggml.h,
// emit_mode == 0), so groups [n, K) would otherwise still describe the head BEFORE this ubatch.
// Group g of the logical state is plane (plane0 + g) of `all`, read through the same s_copy
// gather as the state itself (so a relocated or seq_cp'd cell reads its source): plane0 is the
// pending rollback in replay mode (rs_idx stays 0 there, replay_len selects the conv group), and
// 0 otherwise, since s_copy already carries rs_idx * mem_size. All gathers are materialized before
// any write (gather() then write()) because the source planes overlap the destinations.
static void snapshot_shift_gather(ggml_context * ctx0, ggml_cgraph * gf, const llm_graph_input_rs * inp,
        ggml_tensor * all, int64_t row_elems, uint32_t plane0, uint32_t mem_size,
        std::vector<ggml_tensor *> & gathered) {
    gathered.clear();
    for (uint32_t j = 0; j < inp->snap_shift; ++j) {
        const size_t rows_off = (size_t) (plane0 + j) * mem_size;
        GGML_ASSERT(rows_off + mem_size <= (size_t) all->ne[1]);
        ggml_tensor * planes = ggml_view_2d(ctx0, all, row_elems, all->ne[1] - rows_off, all->nb[1], rows_off * all->nb[1]);
        ggml_tensor * g = ggml_get_rows(ctx0, planes, inp->s_copy_main);
        ggml_build_forward_expand(gf, g);
        gathered.push_back(g);
    }
}

static void snapshot_shift_write(ggml_context * ctx0, ggml_cgraph * gf,
        ggml_tensor * all, int64_t row_elems, int64_t n_seq_tokens, uint32_t kv_head, uint32_t mem_size,
        const std::vector<ggml_tensor *> & gathered) {
    for (size_t j = 0; j < gathered.size(); ++j) {
        const size_t rows_off = ((size_t) n_seq_tokens + j) * mem_size + kv_head;
        ggml_tensor * dst = ggml_view_2d(ctx0, all, row_elems, gathered[j]->ne[1], all->nb[1], rows_off * all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, gathered[j], dst));
    }
}

ggml_tensor * llm_build_delta_net_base::build_conv_state(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        qkv_mixed,
        int64_t              conv_kernel_size,
        int64_t              conv_channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head  = mctx_cur->get_head();
    const auto mem_size = mctx_cur->get_size();

    const int64_t n_seqs = ubatch.n_seqs;

    // gdn_replay rolls the GDN recurrent state back by replaying ingredients, so seq_rm records
    // replay_len INSTEAD of calling set_rs_idx (llama-memory-recurrent.cpp) -- rs_idx stays pinned
    // at 0 for the whole run. The conv state has no replay path: it still keeps its (1 + n_rs_seq)
    // rollback groups and is selected purely by rs_idx through s_copy_idx(). With rs_idx stuck at
    // 0, build_rs would always hand back the OPTIMISTIC group 0, i.e. a convolution window that
    // still contains the rejected draft tokens, while the recurrent state around it was correctly
    // rewound. The wanted depth is the very same `rollback` value seq_rm saw, so select the group
    // explicitly here.
    ggml_tensor * conv_src = conv_states_all;

    const uint32_t conv_rollback = mctx_cur->get_replay_len();
    if (conv_rollback > 0) {
        GGML_ASSERT((uint32_t) conv_states_all->ne[1] >= (conv_rollback + 1) * mem_size);
        conv_src = ggml_view_2d(ctx0, conv_states_all,
                conv_states_all->ne[0], mem_size,
                conv_states_all->nb[1],
                (size_t) conv_rollback * mem_size * conv_states_all->nb[1]);
    }

    ggml_tensor * conv_states = build_rs(inp, conv_src, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    const int64_t row_count = (conv_kernel_size - 1) * conv_channels;

    const size_t row_size  = ggml_row_size(conv_states_all->type, row_count);

    if (cparams.n_rs_seq == 0) {
        const int64_t s_idx  = conv_input->ne[0] - conv_states->ne[0];
        const int64_t s_slot = 0;

        ggml_tensor * conv_state_last =
            ggml_view_3d(ctx0, conv_input,
                    conv_kernel_size - 1, conv_channels, n_seqs,
                    conv_input->nb[1], conv_input->nb[2],
                    ggml_row_size(conv_input->type, s_idx));
        cb(conv_state_last, "conv_state_last", il);

        ggml_tensor * conv_state_update =
            ggml_view_2d(ctx0, conv_states_all,
                    row_count, n_seqs, conv_states_all->nb[1],
                    (s_slot * mem_size + kv_head) * row_size);
        cb(conv_state_update, "conv_state_update", il);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_state_last, conv_state_update));
    } else {
        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // this logic assumes that the last (n_rs_seq + 1) tokens of a sequence in a batch are inside
        //   the same ubatch, which `split_equal()` guarantees via its n_keep_tail argument

        const int64_t K = (int64_t) cparams.n_rs_seq + 1;

        // tokens of this sequence in the ubatch
        const int64_t n_seq_tokens = conv_input->ne[0] - conv_states->ne[0];

        // only the newest min(n_seq_tokens, K) snapshots can be produced from this ubatch; the older
        // slots are caller-owned and must be left untouched -- the same convention the delta-net
        // state snapshots follow (ggml_gated_delta_net emit_mode 0: "when n_tokens < K only slots
        // 0..n_tokens-1 are written; older slots are caller-owned").
        // Writing them anyway with s_idx clamped to 0 overwrites every older conv snapshot with the
        // pre-ubatch window, which destroys the rollback history of a sequence that is decoded one
        // token at a time and makes a later rollback restore a state that never existed.
        const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

        // ... and the older groups move back by n_seq_tokens (no-op when n_seq_tokens >= K)
        std::vector<ggml_tensor *> older;
        snapshot_shift_gather(ctx0, gf, inp, conv_states_all, row_count, conv_rollback, mem_size, older);

        for (int64_t t = K - n_written + 1; t <= K; ++t) {
            const int64_t s_idx  = n_seq_tokens - K + t; // >= 0 by construction
            const int64_t s_slot = K - t;

            ggml_tensor * conv_state_last =
                ggml_view_3d(ctx0, conv_input,
                        conv_kernel_size - 1, conv_channels, n_seqs,
                        conv_input->nb[1], conv_input->nb[2],
                        ggml_row_size(conv_input->type, s_idx));

            ggml_tensor * conv_state_update =
                ggml_view_2d(ctx0,
                        conv_states_all, row_count, n_seqs,
                        conv_states_all->nb[1],
                        (s_slot * mem_size + kv_head) * row_size);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_state_last, conv_state_update));
        }

        snapshot_shift_write(ctx0, gf, conv_states_all, row_count, n_seq_tokens, kv_head, mem_size, older);
    }

    return conv_input;
}

ggml_tensor * llm_build_delta_net_base::build_recurrent_attn(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il) {
    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t S_v          = s->ne[0];
    const int64_t H_v          = s->ne[2];
    const int64_t n_seqs       = s->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0;

    if (!keep) {
        auto attn_out = build_delta_net(q, k, v, g, b, s, il);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);

        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

        return output;
    }

    ggml_tensor * ingr_all = mctx_cur->get_ingr_l(il);
    const bool gdn_replay = ingr_all != nullptr;

    // helper: extract the [S_v,S_v,H_v,n_seqs] final-state block from a K=1, emit_mode=0
    // gated_delta_net output computed over `ntok` tokens.
    auto extract_state_k1 = [&](ggml_tensor * out, int64_t ntok) {
        const int64_t attn_elems = S_v * H_v * ntok * n_seqs;
        return ggml_view_4d(ctx0, out, S_v, S_v, H_v, n_seqs,
            ggml_row_size(out->type, S_v),
            ggml_row_size(out->type, S_v * S_v),
            ggml_row_size(out->type, S_v * S_v * H_v),
            attn_elems * ggml_element_size(out));
    };

    if (!gdn_replay) {
        const int64_t D = S_v * S_v * H_v;
        const int64_t K = cparams.n_rs_seq + 1;

        // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
        ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, K, /*emit_mode=*/0);
        if (n_seq_tokens > 1) {
            res->add_fused_node({LLM_FUSED_OP_GDN_CH, gdn_out, il});
        } else {
            res->add_fused_node({LLM_FUSED_OP_GDN_AR, gdn_out, il});
        }

        const int64_t attn_score_elems    = S_v * H_v * n_seq_tokens * n_seqs;
        const int64_t state_size_per_snap = S_v * S_v * H_v * n_seqs;

        ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
            S_v, H_v, n_seq_tokens, n_seqs,
            ggml_row_size(gdn_out->type, S_v),
            ggml_row_size(gdn_out->type, S_v * H_v),
            ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
            0);
        cb(output, "attn_output", il);

        const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);

        // op writes the last min(n_seq_tokens, K) snapshots; trailing slots are left unwritten
        const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

        // the older groups move back by n_seq_tokens (see snapshot_shift_gather; no-op for n >= K)
        std::vector<ggml_tensor *> older;
        snapshot_shift_gather(ctx0, gf, inp, ssm_states_all, hparams.n_embd_s(), 0, mem_size, older);

        // write the produced snapshots into the recurrent cache (snapshot slot i -> rollback group i)
        ggml_tensor * src = ggml_view_3d(ctx0, gdn_out,
            D, n_seqs, n_written,
            ggml_row_size(gdn_out->type, D),
            ggml_row_size(gdn_out->type, state_size_per_snap),
            ggml_row_size(gdn_out->type, attn_score_elems));

        ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all,
            D, n_seqs, n_written,
            ssm_states_all->nb[1],
            (size_t) mem_size * row_size,
            (size_t) kv_head * row_size);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));

        snapshot_shift_write(ctx0, gf, ssm_states_all, hparams.n_embd_s(), n_seq_tokens, kv_head, mem_size, older);

        return output;
    }

    // --- DRC phase 2: gdn_replay path ---
    //
    // Bookkeeping (llama_memory_recurrent::ckpt_span): s_ckpt is a checkpoint state, the ring
    // holds the ingredients of the S = ckpt_span tokens decoded after it (slot 0 = oldest,
    // chronological), and the logical state of the sequence is s_ckpt advanced by the accepted
    // prefix of m = S - r slots, r = replay_len being the rollback pending for this decode. `s`
    // (the optimistic state) is the checkpoint advanced by all S slots, so it IS the logical
    // state exactly when nothing is pending (r == 0) and it is not stale (a state restore leaves
    // s_l holding the writer's optimistic state, with the checkpoint and the ring carrying the
    // truth). The delta-net rank-1 update is not stably invertible (Sherman-Morrison's
    // denominator 1 - beta*|k|^2 sits near zero for normalized k and beta near 1), so a rollback
    // is never undone from s: the state is re-derived by replaying forward from the checkpoint.
    //
    // After this ubatch of n tokens the ring must hold the last min(m + n, K) tokens behind a
    // checkpoint that precedes them, K = n_rs_seq being the ring capacity:
    //   m + n <= K : room left -- keep the checkpoint, carry the accepted prefix, append n.
    //   n >= K     : the ubatch alone fills the ring -- the checkpoint moves to K tokens before
    //                its end (the op emits that state as a trailing block when n > K; it is the
    //                base state itself when n == K).
    //   otherwise  : short ubatch on a full ring -- advance the checkpoint by e = m + n - K
    //                accepted slots (one replay call over slots [0, e)), carry slots [e, m) and
    //                append n. This is the only shape that costs an extra kernel launch, and it
    //                never occurs on the speculative verify path, whose batches are n_draft + 1
    //                > K = n_draft tokens long.
    // build_rs_inp_impl computes the same span_new = min(m + n, K) and hands it back to the
    // memory once the graph has run.
    ggml_tensor * ckpt_all = mctx_cur->get_s_ckpt_l(il);
    GGML_ASSERT(ckpt_all != nullptr);

    const uint32_t K     = cparams.n_rs_seq;
    const uint32_t r     = mctx_cur->get_replay_len();
    const uint32_t S     = mctx_cur->get_ckpt_span();
    const bool     stale = mctx_cur->get_s_stale();
    GGML_ASSERT(r <= S && S <= K);
    const uint32_t m         = S - r;                   // accepted prefix, in ring slots
    const uint32_t n         = (uint32_t) n_seq_tokens;
    const bool     need_base = r > 0 || stale;          // s is not the logical state

    const size_t ingr_elemsize   = ggml_element_size(ingr_all);
    const size_t ingr_row        = (size_t) hparams.n_embd_s_ingredient(); // elements per slot
    const size_t ring_row        = ingr_row * K;                            // elements per cell
    const size_t state_row_bytes = (size_t) hparams.n_embd_s() * ggml_element_size(ssm_states_all);
    GGML_ASSERT((int64_t) ingr_row == 4 * S_v * H_v); // the op's per-slot ingredient block

    ggml_tensor * s_ckpt = build_rs(inp, ckpt_all, hparams.n_embd_s(), n_seqs);
    s_ckpt = ggml_reshape_4d(ctx0, s_ckpt, S_v, S_v, H_v, n_seqs);

    // the whole ring of every sequence in the ubatch, gathered (and, after seq_cp or a cell
    // move, relocated) by the same s_copy mechanism as the state itself: [ring_row, n_seqs]
    ggml_tensor * ring = build_rs(inp, ingr_all, (int32_t) ring_row, n_seqs);

    const bool kda = (g->ne[0] == S_v);

    // replay ring slots [first, first + count) forward from state4d in ONE batched K=1 call --
    // the slots are chronological, so the accepted prefix is a plain forward-order view -- and
    // return the resulting state. ggml_gated_delta_net needs g/beta (and, through q_dummy, q)
    // fully contiguous, so the strided views into the packed ring are materialized.
    auto replay_from = [&](ggml_tensor * state4d, uint32_t first, uint32_t count) -> ggml_tensor * {
        GGML_ASSERT(count > 0);
        auto comp = [&](int64_t ne0, uint32_t c) {
            return ggml_cont(ctx0, ggml_view_4d(ctx0, ring, ne0, H_v, count, n_seqs,
                4 * S_v * ingr_elemsize, ingr_row * ingr_elemsize, ring->nb[1],
                ((size_t) first * ingr_row + (size_t) c * S_v) * ingr_elemsize));
        };
        ggml_tensor * k_b = comp(S_v, 0);
        ggml_tensor * v_b = comp(S_v, 1);
        ggml_tensor * g_b = comp(kda ? S_v : 1, 2);
        ggml_tensor * b_b = comp(1, 3);
        ggml_tensor * q_dummy = ggml_scale(ctx0, k_b, 0.0f); // q only shapes the (discarded) attn output
        ggml_tensor * out = ggml_gated_delta_net(ctx0, q_dummy, k_b, v_b, g_b, b_b, state4d, /*K=*/1, /*emit_mode=*/0);
        return extract_state_k1(out, (int64_t) count);
    };

    ggml_tensor * base_state = s;
    ggml_tensor * ckpt_new   = nullptr; // when set, the checkpoint row is rewritten with it
    bool          ckpt_from_op = false; // ckpt_new comes from the main call's trailing block
    uint32_t      keep_first = 0;       // gathered ring slots [keep_first, keep_first + keep_count)
    uint32_t      keep_count = 0;       //   are carried over into ring slots [0, keep_count)
    uint32_t      new_first  = 0;       // this ubatch's ingredients land in slots [new_first, new_first + n_new)
    const uint32_t n_new     = std::min(n, K); // the op emits the last min(n, K) tokens

    if (m + n <= K) {
        if (need_base) {
            base_state = m == 0 ? s_ckpt : replay_from(s_ckpt, 0, m);
        }
        keep_first = 0;
        keep_count = m;
        new_first  = m;
    } else if (n >= K) {
        if (need_base) {
            base_state = m == 0 ? s_ckpt : replay_from(s_ckpt, 0, m);
        }
        ckpt_from_op = n > K;
        ckpt_new     = ckpt_from_op ? nullptr : base_state;
        keep_count   = 0;
        new_first    = 0;
    } else {
        const uint32_t e = m + n - K; // 1 <= e < m
        ckpt_new = replay_from(s_ckpt, 0, e);
        if (need_base) {
            base_state = replay_from(ckpt_new, e, m - e); // m - e == K - n >= 1
        }
        keep_first = e;
        keep_count = m - e;
        new_first  = K - n;
    }
    GGML_ASSERT(keep_count + n_new <= K && new_first == keep_count);

    // main call: emit_mode=1 records ingredients (+ the trailing final-state block, + the
    // before-the-window block when n > K) instead of K full snapshots.
    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, base_state, K, /*emit_mode=*/1);
    if (n_seq_tokens > 1) {
        res->add_fused_node({LLM_FUSED_OP_GDN_CH, gdn_out, il});
    } else {
        res->add_fused_node({LLM_FUSED_OP_GDN_AR, gdn_out, il});
    }

    ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
        S_v, H_v, n_seq_tokens, n_seqs,
        ggml_row_size(gdn_out->type, S_v),
        ggml_row_size(gdn_out->type, S_v * H_v),
        ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
        0);
    cb(output, "attn_output", il);

    const int64_t attn_score_elems   = S_v * H_v * n_seq_tokens * n_seqs;
    const int64_t ingr_size_per_snap = (int64_t) ingr_row * n_seqs;
    const int64_t ingr_elems_total   = (int64_t) K * ingr_size_per_snap;
    const size_t  ring_row_bytes     = ingr_all->nb[1];

    // ring, carried slots: gathered [keep_first, keep_first + keep_count) -> head cells [0, keep_count)
    if (keep_count > 0) {
        ggml_tensor * keep_src = ggml_view_3d(ctx0, ring,
            ingr_row, keep_count, n_seqs,
            ingr_row * ingr_elemsize, ring->nb[1],
            (size_t) keep_first * ingr_row * ingr_elemsize);
        ggml_tensor * keep_dst = ggml_view_3d(ctx0, ingr_all,
            ingr_row, keep_count, n_seqs,
            ingr_row * ingr_elemsize, ring_row_bytes,
            (size_t) kv_head * ring_row_bytes);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, keep_src, keep_dst));
    }

    // ring, new slots: the op's slots [K - n_new, K) (right-aligned, chronological; see the
    // emit_mode == 1 contract in ggml.h) -> head cells [new_first, new_first + n_new)
    {
        ggml_tensor * ingr_src = ggml_view_3d(ctx0, gdn_out,
            ingr_row, n_seqs, n_new,
            ggml_row_size(gdn_out->type, ingr_row),
            ggml_row_size(gdn_out->type, ingr_size_per_snap),
            (attn_score_elems + (int64_t) (K - n_new) * ingr_size_per_snap) * ggml_element_size(gdn_out));
        ggml_tensor * ingr_dst = ggml_view_3d(ctx0, ingr_all,
            ingr_row, n_seqs, n_new,
            ring_row_bytes,
            ingr_row * ingr_elemsize,
            (size_t) kv_head * ring_row_bytes + (size_t) new_first * ingr_row * ingr_elemsize);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ingr_src, ingr_dst));
    }

    // the trailing final-state block -> s_l (single row, no widening).
    ggml_tensor * final_state = ggml_view_2d(ctx0, gdn_out, S_v * H_v, S_v * n_seqs,
        ggml_row_size(gdn_out->type, S_v * H_v),
        (attn_score_elems + ingr_elems_total) * ggml_element_size(gdn_out));
    ggml_tensor * final_dst = ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs,
        ssm_states_all->nb[1], (size_t) kv_head * state_row_bytes);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, ggml_reshape_2d(ctx0, final_state, hparams.n_embd_s(), n_seqs), final_dst));

    // the checkpoint for the NEXT decode's replay base.
    //
    // Perf note: n > K is the common shape, not the rare one -- the verify batch is one token
    // longer than the retained window by construction (n_draft + 1 vs n_rs_seq = n_draft), so
    // it fires on every speculative decode. The main call captures the before-the-window state
    // as a trailing block (ggml.h's emit_mode == 1 contract), so extracting it is a view, not a
    // second launch; an earlier version recomputed it with a second K=1 call and measured as a
    // +0.6 to +2.0 ms/round regression across --spec-chain 2/4/6/8 on the real model.
    if (ckpt_from_op) {
        ckpt_new = ggml_view_2d(ctx0, gdn_out, S_v * H_v, S_v * n_seqs,
            ggml_row_size(gdn_out->type, S_v * H_v),
            (attn_score_elems + ingr_elems_total + S_v * S_v * H_v * n_seqs) * ggml_element_size(gdn_out));
    }
    if (ckpt_new != nullptr) {
        ggml_tensor * ckpt_dst = ggml_view_2d(ctx0, ckpt_all, hparams.n_embd_s(), n_seqs,
            ckpt_all->nb[1], (size_t) kv_head * state_row_bytes);
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, ggml_reshape_2d(ctx0, ckpt_new, hparams.n_embd_s(), n_seqs), ckpt_dst));
    }

    return output;
}
