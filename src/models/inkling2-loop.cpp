#include "models.h"

int llama_model_inkling2_loop::layer_phys(int il) const {
    return il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start);
}

void llama_model_inkling2_loop::load_arch_hparams(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_hparams(ml);

    ml.get_key(LLM_KV_LOOP_START,     core_start);
    ml.get_key(LLM_KV_LOOP_END,       core_end);
    ml.get_key(LLM_KV_LOOP_COUNT,     n_passes);
    ml.get_key(LLM_KV_LOOP_GATE_RANK, gate_rank);
    ml.get_key(LLM_KV_LOOP_VARYING,   varying, false);

    n_phys = hparams.n_layer();
    const uint32_t block = hparams.attn_res_block_size;
    // a fixed loop needs at least 2 passes, a variable loop at least 1
    if (n_passes < (varying ? 1u : 2u) || !gate_rank || !core_start || core_start >= core_end || core_end > (uint32_t) n_phys ||
        core_start % block || core_end % block || hparams.n_layer_nextn) {
        throw std::runtime_error(format("inkling2_loop: invalid loop core [%u, %u) with block size %u, %u passes or gate rank %u for %d layers",
                core_start, core_end, block, n_passes, gate_rank, n_phys));
    }
    const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);
    if (n_slot > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("inkling2_loop: %zu cache slots exceed the maximum of %d", n_slot, LLAMA_MAX_LAYERS));
    }

    // expand to the cache slots before load_tensors() allocates the layers and the memory
    hparams.n_layer_all = n_slot;
    for (int il = n_phys; il < (int) n_slot; ++il) {
        const int p = layer_phys(il);
        hparams.n_head_arr[il]        = hparams.n_head_arr[p];
        hparams.n_head_kv_arr[il]     = hparams.n_head_kv_arr[p];
        hparams.n_ff_arr[il]          = hparams.n_ff_arr[p];
        hparams.n_ff_exp_arr[il]      = hparams.n_ff_exp_arr[p];
        hparams.n_expert_used_arr[il] = hparams.n_expert_used_arr[p];
        hparams.is_swa_impl[il]       = hparams.is_swa_impl[p];
        hparams.attn_mode_impl[il]    = hparams.attn_mode_impl[p];
        hparams.lsa_far_bias_impl[il] = hparams.lsa_far_bias_impl[p];
        hparams.is_recr_impl[il]      = hparams.is_recr_impl[p];
    }
}

void llama_model_inkling2_loop::load_arch_tensors(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_tensors(ml);

    const int64_t n_embd = hparams.n_embd;
    const int64_t rank   = gate_rank;
    const llm_tensor down[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_DOWN,   LLM_TENSOR_LOOP_K_CTRL_DOWN,   LLM_TENSOR_LOOP_V_CTRL_DOWN,   LLM_TENSOR_LOOP_FFN_CTRL_DOWN   };
    const llm_tensor gain[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_GAIN,   LLM_TENSOR_LOOP_K_CTRL_GAIN,   LLM_TENSOR_LOOP_V_CTRL_GAIN,   LLM_TENSOR_LOOP_FFN_CTRL_GAIN   };
    const llm_tensor forget[LOOP_CTRL_COUNT] = { LLM_TENSOR_LOOP_Q_CTRL_FORGET, LLM_TENSOR_LOOP_K_CTRL_FORGET, LLM_TENSOR_LOOP_V_CTRL_FORGET, LLM_TENSOR_LOOP_FFN_CTRL_FORGET };

    loop_layers.resize(core_end - core_start);
    for (uint32_t il = core_start; il < core_end; ++il) {
        auto & l = loop_layers[il - core_start];
        // the q, k, v and ffn message queries
        l.msg_query = create_tensor(tn(LLM_TENSOR_LOOP_MSG_QUERY, "weight", il), {n_embd, LOOP_CTRL_COUNT}, 0);
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            l.ctrl_down[c]   = create_tensor(tn(down[c],   "weight", il), {2*n_embd, rank}, 0);
            l.ctrl_gain[c]   = create_tensor(tn(gain[c],   "weight", il), {rank, n_embd}, 0);
            l.ctrl_forget[c] = create_tensor(tn(forget[c], "weight", il), {rank, n_embd}, 0);
        }
    }

    // share the physical weights, each cache slot keeps its own KV and conv state
    for (int il = n_phys; il < (int) hparams.n_layer(); ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

// fork (INKFLAGS): the generic pass loads the optional per-expert ffn_*_exps.scale after load_arch_tensors, so the repeat
// slots must copy the physical layers again or they run the NVFP4 experts without their scale
void llama_model_inkling2_loop::load_arch_tensors_post(llama_model_loader & ml) {
    GGML_UNUSED(ml);
    for (int il = n_phys; il < (int) hparams.n_layer(); ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

void llama_model_inkling2_loop::load_arch_data_post() {
    const ggml_tensor * q0 = loop_layers[0].msg_query;
    for (const auto & l : loop_layers) {
        if (l.msg_query->type != GGML_TYPE_F32 || !ggml_is_contiguous(l.msg_query) || !l.msg_query->buffer ||
                ggml_backend_buffer_get_type(l.msg_query->buffer) != ggml_backend_buffer_get_type(q0->buffer)) {
            return;
        }
    }
    const size_t size = ggml_nbytes(q0);
    msg_query_all = create_derived_tensor(ggml_backend_buffer_get_type(q0->buffer), GGML_TYPE_F32, q0->ne[0],
            q0->ne[1]*loop_layers.size(), "loop_msg_query_all");
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < loop_layers.size(); ++i) {
        ggml_backend_tensor_get(loop_layers[i].msg_query, data.data(), 0, size);
        ggml_backend_tensor_set(msg_query_all, data.data(), i*size, size);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// fork (INKFLAGS): a message depends only on the previous pass, so all core layers' messages are mixed at the pass start in
// one op, [n_embd, 4*n_core, n_tokens]; layer ip reads its 4 rows. LLAMA_INKLING2_MSG_BATCH=0 = one mix per layer.
static bool inkling2_msg_batch() {
    static const bool on = [] { const char * e = getenv("LLAMA_INKLING2_MSG_BATCH"); return !e || atoi(e) != 0; }();
    return on;
}

// fork (INKB1): layer-major prompt. Graph position p (the visit order, = the normal graph's n_visit) runs one visit for
// one ubatch; what the normal graph keeps alive between visits lives in that ubatch's carry [n_embd, n_tokens, n_rows]
// F32, one contiguous [n_embd, n_tokens] row per role:
//   pre(j)       prelude blocks (j < Pb)
//   prel         the prelude output: block Pb of every pass and the input of each pass
//   core(q, k)   block k >= 1 of pass q, two sets by pass parity (pass q reads the set of q - 1 as its bank)
//   cur(q)       the running state, two by pass parity (pass q reads cur(q - 1) as its bank output); the prelude uses
//                cur(0), the tail cur(n_passes); the first tail block is cur(n_passes - 1) itself
//   tail(j)      tail blocks j >= 1
// A block visit copies its input into its block row first and computes from that copy, so the write of the visit
// output into the same cur row depends on every read of it. Inkling 2 Mini (prelude 6, core 6..18 x 3, tail 6,
// blocks of 6): 6 rows, 55,296 B per token.
namespace {
struct ink_b1_layout {
    int start, end, B, n_phys, Q, core, Pb, Kc, Tb, R0, n_rows, n_pos;

    explicit ink_b1_layout(const llama_model_inkling2_loop & m) {
        start  = (int) m.core_start;
        end    = (int) m.core_end;
        B      = (int) m.hparams.attn_res_block_size;
        n_phys = m.n_phys;
        Q      = (int) m.n_passes;
        core   = end - start;
        Pb     = start/B;
        Kc     = core/B;
        Tb     = (n_phys - end + B - 1)/B;
        R0     = Pb + 1 + 2*(Kc - 1);
        n_rows = R0 + 2 + std::max(0, Tb - 1);
        n_pos  = start + Q*core + (n_phys - end);
    }

    int pre(int j)              const { return j; }
    int prel()                  const { return Pb; }
    int core_blk(int q, int k)  const { return Pb + 1 + (q % 2)*(Kc - 1) + (k - 1); }
    int cur(int q)              const { return R0 + (q < 0 ? 0 : q % 2); }
    int tail(int j)             const { return R0 + 2 + (j - 1); }

    // q: -1 prelude, 0..Q-1 pass, Q tail; ip: physical layer; il: cache slot
    void visit(int p, int & q, int & ip, int & il) const {
        if (p < start) {
            q = -1; ip = p; il = p;
        } else if (p < start + Q*core) {
            q  = (p - start)/core;
            ip = start + (p - start) % core;
            il = q == 0 ? ip : n_phys + (q - 1)*core + (ip - start);
        } else {
            q = Q; ip = end + (p - start - Q*core); il = ip;
        }
    }

    // the block rows of the normal graph when it reaches position p (before p's own push; p = n_pos: the output's),
    // and the bank rows of p's pass (pass >= 1)
    std::vector<int> blocks(int p, std::vector<int> & bank) const {
        std::vector<int> blk;
        bank.clear();
        int pos = 0;
        for (int ip = 0; ip < start; ++ip, ++pos) {
            if (pos == p) { return blk; }
            if (ip % B == 0) { blk.push_back(pre(ip/B)); }
        }
        const std::vector<int> prel_blk = blk;
        for (int q = 0; q < Q; ++q) {
            if (q > 0) {
                bank.assign(blk.begin() + prel_blk.size() + 1, blk.end());
                blk = prel_blk;
            }
            for (int ip = start; ip < end; ++ip, ++pos) {
                if (pos == p) { return blk; }
                if (ip % B == 0) { blk.push_back(ip == start ? prel() : core_blk(q, (ip - start)/B)); }
            }
        }
        bank.clear();
        for (int ip = end; ip < n_phys; ++ip, ++pos) {
            if (pos == p) { return blk; }
            if (ip % B == 0) { blk.push_back(ip == end ? cur(Q - 1) : tail((ip - end)/B)); }
        }
        return blk;
    }
};
} // namespace

int32_t llama_model_inkling2_loop::ink_b1_carry_rows() const {
    const ink_b1_layout L(*this);
    // the row map needs whole blocks in the prelude and the core
    if (L.B <= 0 || L.start % L.B || L.core % L.B || L.Kc < 1 || L.Q < 1 || L.n_pos != (int) hparams.n_layer() || L.n_pos < 2) {
        return 0;
    }
    return L.n_rows;
}

int32_t llama_model_inkling2_loop::ink_b1_visit_slot(int32_t p) const {
    const ink_b1_layout L(*this);
    int q, ip, il;
    L.visit(p, q, ip, il);
    return il;
}

void llama_model_inkling2_loop::graph::build_visit(const llama_model_inkling2_loop & m) {
    const ink_b1_layout L(m);
    const int p = b1_il0;
    GGML_ASSERT(b1_il1 == p + 1 && p >= 0 && p < n_layer && L.n_pos == n_layer && "INKB1: one visit per graph");
    GGML_ASSERT(b1_carry->type == GGML_TYPE_F32 && b1_carry->ne[0] == n_embd && b1_carry->ne[1] == n_tokens &&
            b1_carry->ne[2] == L.n_rows && ggml_is_contiguous(b1_carry));
    const bool first = p == 0;
    const bool last  = p == n_layer - 1;
    auto row = [&](int r) {
        return ggml_view_2d(ctx0, b1_carry, n_embd, n_tokens, b1_carry->nb[1], (size_t) r*b1_carry->nb[2]);
    };

    // the first visit reads the tokens, the last one makes the outputs
    ggml_tensor * embd = build_inputs(first, last);
    n_visit = p;

    int q, ip, il;
    L.visit(p, q, ip, il);
    std::vector<int> bank_rows;
    std::vector<ggml_tensor *> blocks;
    for (int r : L.blocks(p, bank_rows)) {
        blocks.push_back(row(r));
    }

    ggml_tensor * cur = first ? embd :
        q >= 1 && q < L.Q && ip == L.start ? row(L.prel()) :     // a later pass starts from the prelude output
        q == L.Q && ip == L.end            ? row(L.cur(L.Q - 1)) : // the tail starts from the last pass output
                                             row(L.cur(q < L.Q ? q : L.Q));
    if (ip % L.B == 0) {
        // the block this visit pushes; a pass >= 1 pushes the prelude row and the tail the last pass row as they are
        int br = -1;
        if (q < 0) {
            br = L.pre(ip/L.B);
        } else if (q < L.Q) {
            br = ip == L.start ? (q == 0 ? L.prel() : -1) : L.core_blk(q, (ip - L.start)/L.B);
        } else if (ip != L.end) {
            br = L.tail((ip - L.end)/L.B);
        }
        if (br >= 0) {
            cur = ggml_cpy(ctx0, cur, row(br));
        }
    }

    ggml_tensor * messages = nullptr;
    const loop_layer * loop = nullptr;
    if (q >= 1 && q < L.Q) {
        loop = &m.loop_layers[ip - L.start];
        std::vector<ggml_tensor *> bank;
        for (int r : bank_rows) {
            bank.push_back(row(r));
        }
        // the batched pass mix computes each query on its own (depth_mix), so its 4 rows alone give the same values
        ggml_tensor * query = inkling2_msg_batch() && m.msg_query_all ?
            ggml_view_2d(ctx0, m.msg_query_all, n_embd, LOOP_CTRL_COUNT, m.msg_query_all->nb[1],
                    (size_t) (ip - L.start)*LOOP_CTRL_COUNT*m.msg_query_all->nb[1]) :
            loop->msg_query;
        messages = mix(bank, row(L.cur(q - 1)), query);
    }

    cur = build_layer(il, ip, cur, blocks, messages, loop);

    if (!last) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, cur, row(L.cur(q < 0 ? 0 : q))));
        return;
    }
    build_output(cur, blocks);
}

llama_model_inkling2_loop::graph::graph(const llm_graph_params & params) : llama_model_inkling2::graph(params, no_build{}) {
    const auto & m = static_cast<const llama_model_inkling2_loop &>(model);
    if (b1_carry) {
        build_visit(m);
        return;
    }
    const int start = m.core_start;
    const int end   = m.core_end;

    auto * cur = build_inputs();
    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < start; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }

    // every pass starts from the prelude output
    auto * prelude = cur;
    const auto prelude_blocks = blocks;
    for (uint32_t pass = 0; pass < m.n_passes; ++pass) {
        // the bank of the previous pass: its blocks after the first core block, and its output
        std::vector<ggml_tensor *> bank;
        ggml_tensor * bank_out = nullptr;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
        }
        // fork (INKFLAGS): all core layers' messages of the pass in one mix (inkling2_msg_batch)
        ggml_tensor * msgs = pass > 0 && inkling2_msg_batch() && m.msg_query_all ? mix(bank, bank_out, m.msg_query_all) : nullptr;
        for (int ip = start; ip < end; ++ip) {
            const int il = pass == 0 ? ip : m.n_phys + (pass - 1)*(end - start) + (ip - start);
            if (pass == 0) {
                cur = build_layer(il, ip, cur, blocks);
                continue;
            }
            const auto & loop = m.loop_layers[ip - start];
            ggml_tensor * messages = msgs ? ggml_view_3d(ctx0, msgs, n_embd, LOOP_CTRL_COUNT, n_tokens, msgs->nb[1], msgs->nb[2],
                    (ip - start)*LOOP_CTRL_COUNT*msgs->nb[1]) : mix(bank, bank_out, loop.msg_query);
            cur = build_layer(il, ip, cur, blocks, messages, &loop);
        }
    }

    for (int il = end; il < m.n_phys; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }
    build_output(cur, blocks);
}
