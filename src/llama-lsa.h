#pragma once

// Latent sparse attention (LSA) quality probe, M0.
//
// Opt-in and lossy. Off unless LLAMA_LSA_MODE is set (or --attn-sparse is passed). For each full-attention layer
// that has a learned index map, each query row scores every KV cell in a small 64-dim index space, keeps the first
// `sinks` positions, the last `window` positions and the `keep` best-scoring others, and masks the rest out of the
// normal attention through the KQ mask. M0 stores no new state: the index keys are recomputed from the K cache in
// every step, so prompt-cache save/restore is unchanged.
//
// Modes:
//   key    : learned map. Index key = normalize(Wk^T k_nope), index query = Wq^T (sum of the group's q_nope heads).
//   pos    : like key, but cells are scored by the centroid of fixed blocks of `bucket` consecutive cells
//            (graph-level stand-in for the cluster step; the per-cell score breaks ties inside a block).
//   oracle : exact head-summed logits q.k; no map needed, upper bound for the selection.
//   mass   : exact attention mass; no map needed. Both diagnostics use LLAMA_LSA_LAYERS when no map is set.
//   quest  : training-free, real keys: blocks of `bucket` cells ranked per KV head by the upper bound
//            sum_c (Q+_c max_c + Q-_c min_c) from the block's per-channel K max/min (Q+/Q- = summed positive/negative
//            parts of the group's q heads). Plain K only, recomputed from K each step (probe; map = layer list only).
//
// The map lives in a sidecar GGUF written by scripts/lsa_convert_map.py: per layer `lsa.<il>.wq` and `lsa.<il>.wk`,
// f32, ne = [n_head_kv*head_dim, 64], with zero rows for the rotary dims so they apply to the cached (post-RoPE)
// keys and the post-RoPE queries directly.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>

struct llama_lsa_config {
    enum mode_t { OFF = 0, KEY, POS, ORACLE, MASS, QUEST };

    mode_t      mode      = OFF;
    std::string map_path;           // LLAMA_LSA_MAP
    int64_t     keep      = 4096;   // LLAMA_LSA_KEEP: long-range cells kept per query row
    double      keep_frac = 0.0;    // LLAMA_LSA_KEEP_FRAC: if > 0, keep = frac * n_kv instead
    int64_t     sinks     = 128;    // LLAMA_LSA_SINKS: first positions always kept
    int64_t     window    = 8192;   // LLAMA_LSA_WINDOW: last positions always kept (query position included)
    int64_t     bucket    = 8;      // LLAMA_LSA_BUCKET: block size for mode=pos
    int64_t     max_rows  = 16;     // LLAMA_LSA_MAX_ROWS: ubatches with more rows stay dense (prefill)
    int64_t     min_pos   = 0;      // LLAMA_LSA_MIN_POS: ubatches starting before this position stay dense
    bool        fa_sparse = false;  // LLAMA_LSA_FA_SPARSE: pass the kept count to FA (n_kv_max) for its sparse path
    bool        shared    = false;  // LLAMA_LSA_SHARED: one long-range pick per ubatch (summed row scores), not per row
    bool        shared_max = false; // LLAMA_LSA_SHARED=2 (block pick only): rank blocks by their best row's normalized
                                    // log share instead of the summed scores (FUSED_SEL: select op mode 2; else graph chain)
    bool        shared_union = false; // LLAMA_LSA_SHARED=3 (block or fused unit pick, one-sequence ubatch, T > 1): every row
                                    // picks its own keep budget, the rows' picks are OR-ed into one plane for all rows (each
                                    // row sees a superset of its own pick; the kernel reads the union, so BLKLIST applies)
    bool        blk_list  = false;  // LLAMA_LSA_BLKLIST=1 (SJ-KVaRN decode, shared block pick): the FA node takes the block
                                    // pick (ggml_flash_attn_ext_set_lsa_blocks) plus one per-graph mask, no per-layer mask
    bool        vk        = false;  // LLAMA_LSA_VK=1 (with BLKLIST=1 and COMP=2): compensation as virtual keys inside the
                                    // SJ-KVaRN stream kernel (ggml_flash_attn_ext_set_lsa_vk): no LSE output, no merge op;
                                    // the block means and rings of SJ-KVaRN layers are then kept in the rotated cache basis
    bool        packed    = false;  // LLAMA_LSA_PACKED=1: keep the packed I32 KQ mask for non-LSA layers (LSA layers get an F16 input)
    int64_t     pick_every = 1;     // LLAMA_LSA_PICK_EVERY=N (block pick, N > 1): reuse a layer's block pick for up to N - 1
                                    // decode/verify graphs; re-pick on invalidate, rewind, n_kv shrink or when a block
                                    // leaves the newest row's window. 1 = pick every graph (default)
    bool        blist_lean = false; // LLAMA_LSA_BLIST_LEAN=1: block-list compensation counts broadcast row 0 instead of
                                    // repeating it to [nu, T] (same values, one fewer full-width copy per layer)
    bool        pick_check = false; // LLAMA_LSA_PICK_CHECK=1 (diagnostic, unit pick + PICK_EVERY): a reuse graph also builds
                                    // the fresh pick and logs its overlap with the reused one; attention still uses the reused pick
    bool        elig_drop = false;  // LLAMA_LSA_ELIG=1 (block pick): a block may take a pick iff it has >= 1 visible droppable cell
                                    // (any row), not by its first cell's position (plain cache: holes / multi-seq cells)
    bool        smax_chain = false; // LLAMA_LSA_SMAX_CHAIN=1: SHARED=2 + FUSED_SEL keeps the graph chain (A/B; default fused op)
    bool        fast      = false;  // LLAMA_LSA_FAST: copy the kept cells out and attend over them only (llama-lsa-gather.h);
                                    // one pick per ubatch (as shared) with the window widened to cover every row
    bool        per_head  = false;  // LLAMA_LSA_PER_HEAD: each KV head picks its own cells (oracle/mass/key: one FA per group; SJ key + index cache: per-head index keys, per-head mask at decode widths)
    bool        sj_compact = false; // LLAMA_LSA_SJ_COMPACT: SJ-KVaRN only, pass the kept-union bound to FA (n_kv_max) so the
                                    // CUDA compact path gathers the kept cells and attends over them only (not the whole cache)
    bool        idx_cache = false;  // LLAMA_LSA_IDX_CACHE: keep the normalized 64-dim index keys in a side cache (M1a),
                                    // written as K is stored, instead of recomputing them from all of K every step
    bool        idx_q8    = false;  // LLAMA_LSA_IDX_Q8: index cache stored as q8_0 (34 B per 32 values instead of 64 B), so the
                                    // per-cell pick reads ~half the bytes; lossy in the pick only. Not with LLAMA_LSA_BLOCK_SUM.
    double      topp      = 0.0;    // LLAMA_LSA_TOPP: if > 0, inside the keep cap keep only the smallest set of middle cells
                                    // whose index-softmax mass reaches this share (adaptive keep per row and layer)
    double      topp_tau  = 2.0;    // LLAMA_LSA_TOPP_TAU: softmax scale on row z-scores of the index score
    double      topp_min  = 0.02;   // LLAMA_LSA_TOPP_MIN: floor, this share of n_kv is always kept (best index scores)
    int64_t     block     = 0;      // LLAMA_LSA_BLOCK: key mode picks whole blocks of this many cells (aligned to cell 0),
                                    // ranked by the block's share of softmax(block_tau * index score); 0 = per-cell pick
    double      block_tau = 2.0;    // LLAMA_LSA_BLOCK_TAU: softmax scale for the block mass (box 11-04: 0.125 cost ~2.6x
                                    // KLD at m25 on SJ-KVaRN, 0.5 and 2.0 match the per-cell pick; was 0.125)
    bool        comp      = false;  // LLAMA_LSA_COMP: q8_0/f16 KV probe, T <= comp_max_t: every dropped block of `block` cells
                                    // (128 if no block mode) re-enters the softmax as one stand-in key/value (block mean K,
                                    // mean V) with log(dropped cells in it) added to its logit (centroid compensation)
    bool        comp_fast = false;  // LLAMA_LSA_COMP=2 (needs LLAMA_LSA_IDX_CACHE=1): same math without dense scores: masked
                                    // FA also writes its row log-sum-exp, block means come from a per-layer cache refreshed
                                    // for the blocks each ubatch writes (full rebuild when the index cache is invalid)
    double      comp_bias = 0.0;    // LLAMA_LSA_COMP_BIAS: added to the stand-in logits
    int64_t     comp_max_t = 64;    // LLAMA_LSA_COMP_MAX_T: wider ubatches skip the compensation
    bool        comp_lb   = false;  // LLAMA_LSA_COMP_LB=1 (opt-in, A5): fast compensation also on the 3/2 and 3/3 trellis
                                    // SJ-KVaRN bodies (the low-bit tile kernel writes the LSE in its uniform fixup)
    bool        block_sum = false;  // LLAMA_LSA_BLOCK_SUM: with LLAMA_LSA_BLOCK and the index cache, rank blocks by an upper
                                    // bound from per-block channel max/min of the index keys (kept next to the index cache,
                                    // updated for the blocks each ubatch writes) instead of scoring every cell
    int         block_sum_mode = 1; // LLAMA_LSA_BLOCK_SUM=2: the two planes hold the channel mean and variance instead of
                                    // max/min; block score = tau*(q.mean) + tau^2/2*(q^2.var), the Gaussian estimate of the
                                    // block's log softmax mass (minus log B), i.e. of the exact unit logsumexp pick
    bool        sum_op = false;     // LLAMA_LSA_SUM_OP=1 (with BLOCK_SUM + FUSED_SEL): the select op scores the blocks from the
                                    // summary planes itself (one launch; no matmuls / [nbs, T] score tensor in the graph)
    bool        blk_mask_legacy = false; // LLAMA_LSA_BLK_MASK_LEGACY=1: block-sum mask built per layer the old way
    bool        comp_cnt_legacy = false; // LLAMA_LSA_COMP_CNT_LEGACY=1: COMP=2 dropped counts from the full-width step/mul build
                                         // (control for the shared base mask; masks are bit-identical)
    bool        comp_cnt_exp = false;    // LLAMA_LSA_COMP_CNT_EXP=1: COMP=2 dropped counts from exp(mask) even when the block
                                         // pick hands them over (control for the block-count shortcut; same integers)
    bool        blk_lse   = false;  // LLAMA_LSA_BLK_LSE=1: per-cell block pick ranks blocks by the floor-free block logsumexp (REC_BLK always does)
    bool        fused_sel = false;  // LLAMA_LSA_FUSED_SEL=1: block pick (eligibility + top-k + select) as one op (GGML_OP_LSA_BLOCK_SELECT)
    bool        graph_reuse = false; // LLAMA_LSA_GRAPH_REUSE=1 (opt-in): an LSA graph may be reused like a dense one once
                                    // its index and mean caches are settled (llama_lsa_graph_reuse_on + the checks in
                                    // llm_graph_input_attn_kv); default: every LSA ubatch rebuilds its graph
    bool        unit_op   = false;  // LLAMA_LSA_UNIT_OP=1 (with FUSED_SEL + REFINE): index scoring + unit logsumexp in one op (GGML_OP_LSA_UNIT_LSE), both selects read its [n_kv/F, T] output; no score tensor
    bool        fused_comp = false; // LLAMA_LSA_FUSED_COMP=1: COMP=2 stand-in scores + softmax + merge as one op (GGML_OP_LSA_COMP_MERGE)
    bool        ring_legacy = false; // LLAMA_LSA_RING_LEGACY=1: SJ-KVaRN COMP=2 staging ring of n_ubatch + 2 blocks per layer (old)
    int64_t     ring_rows  = 0;     // LLAMA_LSA_RING_ROWS: persistent per-layer carry ring (default 4 blocks, >= 2 blocks); the
                                    // ubatch's own rows are staged in graph scratch, shared by all layers
    bool        rec_blk   = false;  // LLAMA_LSA_REC_BLK=1 (with LLAMA_LSA_BLOCK, key mode): the record pick ranks whole blocks
                                    // by their tau block score (top-k over n_kv/B blocks, not n_kv cells) and builds the mask
                                    // from the shared per-graph base; drops the per-cell top-k's one partial block
    int64_t     refine    = 0;      // LLAMA_LSA_REFINE=F (with LLAMA_LSA_BLOCK=B, key mode, per-cell pick; 0 = off): two-stage
                                    // pick. Rank the middle blocks by the block score, keep REFINE_MULT x keep/B candidate
                                    // blocks, then take the best keep cells inside them by the logsumexp score of F-cell
                                    // units (F = 1: per cell; F = 16: SJ-KVaRN strips). F must divide B.
    int64_t     blist_min_unit = 16; // LLAMA_LSA_BLKLIST_FINE=1: units below 16 cells (REFINE = 1/2/4/8, power of 2) also go to
                                    // the SJ-KVaRN block list (the kernels resolve the pick per position; 16-cell strips with
                                    // no picked cell are skipped). Default 16: finer unit picks keep the per-layer mask
    bool        refine_sh = false;  // LLAMA_LSA_REFINE_SH=1 (REFINE + FUSED_SEL + SHARED, T > 1): both refine stages pick once
                                    // per ubatch (select op shared mode), so the unit pick can go to the SJ-KVaRN block list
    double      refine_mult = 2.0;  // LLAMA_LSA_REFINE_MULT: candidate blocks = mult x keep/B (>= 1)
    std::string dump;               // LLAMA_LSA_DUMP=<prefix>: probe only, plain K/V: the first decode graph (T <= 8,
                                    // n_kv >= LLAMA_LSA_DUMP_MIN_KV) of each LSA layer writes q, K, V, index/exact score,
                                    // mask and positions as f32 files <prefix>.L<il>.<name>.f32; attention stays dense
                                    // Oracle/mass mode needs no map, so calibration can precede the first fitted map.
    int64_t     dump_min_kv = 32768; // LLAMA_LSA_DUMP_MIN_KV
    std::string score_log;          // LLAMA_LSA_SCORE_LOG=<prefix> (diagnostic, needs LLAMA_LSA_UNIT_OP=1 + REFINE + FUSED_SEL):
                                    // every graph with n_kv >= LLAMA_LSA_SCORE_LOG_MIN_KV appends, per LSA layer, row 0's
                                    // position and the max over rows of each F-cell unit's logsumexp score (f16) to
                                    // <prefix>.L<il>.f16, so residency/keep studies can replay picks offline at any keep
    int64_t     score_log_min_kv = 32768; // LLAMA_LSA_SCORE_LOG_MIN_KV
    double      resident  = 0.0;    // LLAMA_LSA_RESIDENT=<pct> (opt-in, SJ-KVaRN, one stream, D 256, G 128):
                                    // the sealed body of each LSA layer lives in pinned host RAM and the card keeps <pct>%
                                    // of its 16-cell strips (scalar 4/4, ggml_backend_cuda_qsa_strip_buffer_type) or of
                                    // its whole 128-cell records (3/3t, 3/2t and other low-bit bodies,
                                    // ggml_backend_cuda_qsa_page_buffer_type) in CLOCK slots, filled from the pick before
                                    // each decode attention; records without a slot are read from the host master, so the
                                    // output does not depend on residency. Also turns on the exact tile skip.
                                    // 0 = off (body on the card, byte-identical)
    bool        sj_skip   = false;  // LLAMA_LSA_SJ_SKIP: SJ-KVaRN decode kernel walks only the body records some row keeps
                                    // (sets GGML_SJKVARN_LIVE_RECORDS=1; pair with LLAMA_LSA_BLOCK=128 so records die whole)
                                    // LLAMA_LSA_SJ_SKIP=2: also skip dead 16-cell strips inside live records (GGML_SJKVARN_LIVE_STRIPS=1)
    bool        recall    = false;  // LLAMA_LSA_RECALL: plain-K probe, T <= 64: per layer, the share of the true attention
                                    // probability (all q heads) that falls on the kept cells, logged at WARN every 256 steps
    bool        stats     = false;  // LLAMA_LSA_STATS: log the mean kept share per layer (WARN, every 256 steps; syncs)
    bool        spec_recall = false; // LLAMA_LSA_SPEC_RECALL: diagnostic only; queries one to three layers ahead
    // LLAMA_LSA_J3=1 (opt-in, needs LLAMA_LSA_RESIDENT record pages): look-ahead prefetch into the slots of later LSA
    // layers on a side copy stream. Never changes what attention reads (a record without a slot comes from the master).
    // Sources, each its own switch: _A3=<pct> stale query three mapped layers ahead (pct of the target's pick budget,
    // 100/110/125/150, default 150, 0 off); _A1=<pct> one layer ahead, top-up of what is not already resident (default
    // 150, 0 off); _PREV=0/1 the target's previous-step pick (default 1, also keeps those slots from prefetch eviction).
    // _FIRST=N: the first N mapped layers keep the whole body on the card (default 3; no earlier layer to predict them).
    bool        j3        = false;
    int         j3_a3     = 150;
    int         j3_a1     = 150;
    bool        j3_prev   = true;
    int         j3_first  = 3;
    std::string layers;             // LLAMA_LSA_LAYERS: comma list of layer ids to sparsify (default: all mapped)
    std::string keep_frac_layers;   // LLAMA_LSA_KEEP_FRAC_LAYERS: "il:frac,il:frac" per-layer override of keep_frac
                                    // (frac of n_kv; 1 or more = that layer stays dense)
    std::string keep_sched;         // LLAMA_LSA_KEEP_SCHED="64K:0.5,128K:0.4,192K:0.3,260K:0.25" (opt-in): keep fraction
                                    // by context length (n_kv of the graph, K = 1024). Below the first point every layer
                                    // stays dense (no pick); at or past a point its fraction holds until the next point.
                                    // Replaces LLAMA_LSA_KEEP_FRAC / LLAMA_LSA_KEEP; per-layer KEEP_FRAC_LAYERS still win.
                                    // n_kv is part of every graph's shape, so a step never reuses a stale graph.
    double      resident_mib = 0.0; // LLAMA_LSA_RESIDENT_MIB=<MiB> (opt-in): fixed on-card slot budget for all
                                    // LLAMA_LSA_RESIDENT layers together, split evenly; wins over the <pct> size.
                                    // Setting it alone turns residency on.
    bool        resident_auto = false; // LLAMA_LSA_RESIDENT=auto (opt-in): size the slots from the VRAM left after the
                                    // model, the rest of the KV cache and LLAMA_LSA_RESIDENT_RESERVE_MIB (compute + margin),
                                    // aiming for >= LLAMA_LSA_RESIDENT_TARGET % of records on the card; residency stays off
                                    // when the whole body fits or the pinned master would leave < 5 GiB host RAM.
                                    // One log line gives the choice and the reason (llama_lsa_resident_fit, llama-lsa-fit.h).
                                    // LLAMA_LSA_RESIDENT=<pct> or LLAMA_LSA_RESIDENT_MIB win over auto.
    double      resident_reserve_mib = 1536.0; // LLAMA_LSA_RESIDENT_RESERVE_MIB (auto): compute buffers (~860 MiB at 100K) + margin
    bool        resident_reserve_set = false; // the env set LLAMA_LSA_RESIDENT_RESERVE_MIB / _COMPUTE_MIB: a measured compute
    bool        resident_compute_set = false; // buffer (llama_lsa_note_compute) does not replace that value
    double      resident_compute_mib = 1024.0; // LLAMA_LSA_RESIDENT_COMPUTE_MIB (auto): compute buffers alone, no margin; the
                                    // non-resident fall-back is only taken if the bodies fit next to this
    double      resident_margin_mib = -1.0; // LLAMA_LSA_RESIDENT_MARGIN_MIB (auto): free VRAM left after slots + compute;
                                    // < 0 = the default (reserve - compute, 512 MiB). Explicit --fit-target sets it (common.cpp)
                                    // when the env is unset, so auto keeps the same free margin as the fit (7,680 MiB cap on
                                    // an 8 GB card = --fit-target 512). LLAMA_LSA_RESIDENT_RESERVE_MIB still wins over both.
    std::string resident_margin_from; // LLAMA_LSA_RESIDENT_MARGIN_FROM: where the margin came from, for the auto log line
    double      resident_target = 80.0; // LLAMA_LSA_RESIDENT_TARGET (auto): wanted % of records on the card
    bool        resident_keep_fit = false; // LLAMA_LSA_RESIDENT_LOWER_KEEP=1 (old name _KEEP_FIT; auto, opt-in, lossy): if the auto share is
                                    // below one row's read set, cap the keep fraction so the read set fits the slots
    double      resident_keep_min = 0.25; // LLAMA_LSA_RESIDENT_KEEP_MIN: the cap never goes under this fraction
    bool        resident_stage_auto = true; // LLAMA_LSA_RESIDENT_AUTO_STAGE (auto, default 1): when auto keeps a slot pool
                                    // below one step's read set, size the record-page stage for the excess; an explicit
                                    // GGML_CUDA_LSA_RESIDENT_STAGE wins (auto stage then off)
                                    // LLAMA_LSA_RESIDENT_FREE_MIB (auto, test hook): use this as the device's free VRAM
    int64_t     slice           = 0;   // LLAMA_LSA_SLICE=<cells> (opt-in, lossy vs the global pick): every select node
                                    // (cells / block / summary, both refine stages) cuts the units into slices of this many cells; each
                                    // slice keeps its own share of keep (largest remainder), no global top-k. A fused
                                    // decode kernel could then pick per slice in one launch. 0 = global pick (unchanged)
    double      resident_sticky = 0.0; // LLAMA_LSA_RESIDENT_STICKY=<b> (opt-in, record pages only): the pick adds b (log
                                    // scale, so e^b times the mass) to a unit whose 128-cell record is already in a card
                                    // slot, so near-ties go to resident records and fewer are fetched. Output then depends
                                    // on slot history: a restore is exact vs a run with the same slots, not vs any run.

    bool   mapless_probe() const {
        return map_path.empty() && (mode == ORACLE || mode == MASS);
    }
    bool   layer_enabled(int il) const;
    double keep_frac_for(int il) const; // per-layer override, else keep_frac
    double keep_frac_for(int il, double base) const; // per-layer override, else base
    // LLAMA_LSA_KEEP_SCHED: fraction at this n_kv; 0 = below the first point (dense), < 0 = no schedule
    double keep_frac_sched(int64_t n_kv) const;
};

// parsed once from the environment
const llama_lsa_config & llama_lsa_cfg();

// Build the logging probe's coarse gate and fine pick with the same cell-to-unit map.
ggml_tensor * llama_lsa_build_recall_pick(ggml_context * ctx, ggml_tensor * scores, ggml_tensor * mask,
        ggml_tensor * cell_pos, ggml_tensor * q_pos, int block, int refine, float tau,
        int window, int sinks, int coarse_keep, int fine_keep, int budget_pct = 100);

// LLAMA_LSA_GRAPH_REUSE=1 and no setting whose graph depends on more than (n_tokens, n_kv, settled caches): pick reuse,
// min_pos, dumps/score logs and the resident paths keep rebuilding per ubatch
bool llama_lsa_graph_reuse_on();
// bumped whenever a built LSA graph may have gone stale: a cache entry was created, invalidated, forgotten or restored
uint64_t llama_lsa_graph_gen();

// LLAMA_LSA_RESIDENT_LOWER_KEEP: cap on the keep fraction (frac) or cells (absolute LLAMA_LSA_KEEP) set once by the KV
// cache's resident sizing before any graph is built; 0 = no cap (default, every pick unchanged)
void   llama_lsa_set_keep_cap(double frac, int64_t cells);
double llama_lsa_keep_cap_frac();
int64_t llama_lsa_keep_cap_cells();

// LLAMA_LSA_RESIDENT=auto: compute buffer bytes a memory-check pass (no_alloc context) measured for a device (largest
// seen per device); the real context's resident sizing uses it instead of the default reserve (llama_lsa_resident_reserve).
// 0 = no measurement.
void    llama_lsa_note_compute(const void * dev, int64_t bytes);
int64_t llama_lsa_measured_compute(const void * dev);

// Per-layer map tensors on the given buffer type. Returns false if there is no map for this layer
// (or, for mode=oracle, if the layer is not listed in the map), and logs once if the file cannot be read.
// the layer is LSA-enabled and the map has its index rows (host check, no device copy)
bool llama_lsa_map_has_layer(int il);
bool llama_lsa_get_map(int il, ggml_backend_buffer_type_t buft, ggml_tensor ** wq, ggml_tensor ** wk);

// M1a index-key side cache (LLAMA_LSA_IDX_CACHE=1): one F16 (q8_0 with LLAMA_LSA_IDX_Q8) [d_idx, kv_size, n_head]
// tensor per KV cache layer tensor; n_head = 1 for the shared key, n_head_kv for the per-KV-head key (head-major, so
// each head's keys are one contiguous [d_idx, kv_size] matrix the score matmul reads in place)
// (keyed by the K cache tensor), holding rms-normalized Wk^T k for every cell. Cells are written by the graph as
// K is stored. Anything that changes K outside the graph (state load = prompt-cache restore, K shift, stream
// copies) calls llama_lsa_idx_invalidate(); the next executed graph then rebuilds the whole layer from the K
// cache once. So prompt caching keeps working with no new state in the saved file. Validity is committed only
// when a graph that did the rebuild is about to execute (set_input), never at build time (reserve graphs).
struct llama_lsa_idx_slot {
    ggml_tensor * t     = nullptr;
    ggml_tensor * smax  = nullptr; // LLAMA_LSA_BLOCK_SUM: F16 [d_idx, kv_size/block] channel max / min of each block's
    ggml_tensor * smin  = nullptr; // index keys (same validity as t; a state restore recomputes them on the host)
    bool          valid = false; // false: this graph must rebuild [0, n_kv) from the K cache
    uint64_t      epoch = 0;     // epoch to commit with llama_lsa_idx_commit after the rebuild graph is set up
};
// kv_size = rows of the index cache (0: kcache->ne[1]; SJ-KVaRN passes the cell count, its K tensor holds only the ring)
bool llama_lsa_idx_get(const ggml_tensor * kcache, int64_t d_idx, ggml_backend_buffer_type_t buft, llama_lsa_idx_slot * out,
        int64_t kv_size = 0, int64_t n_head = 1);
void llama_lsa_idx_commit(const ggml_tensor * kcache, uint64_t epoch);
// kcaches = the K cache tensors whose cells changed outside the graph (each KV cache passes only its own layers, so the
// MTP drafter's restore or shift no longer marks the main model's layers stale); nullptr = every layer of every context
void llama_lsa_idx_invalidate(const ggml_tensor * const * kcaches, size_t n, const char * why = nullptr);
// the K caches are being destroyed: drop their index/mean entries and free the buffers
void llama_lsa_idx_forget(const ggml_tensor * const * kcaches, size_t n);
bool llama_lsa_idx_trace(); // LLAMA_LSA_IDX_TRACE=1: invalidate/save/restore/stale events logged at WARN
// one WARN per process (no IDX_TRACE needed) when a prompt-cache restore leaves LSA layers dense: the saved state
// had no LSA index suffix (saved with LSA off or by an older build) or some of its layers did not match this context
void llama_lsa_warn_restore_dense(const char * why, uint32_t n_ok, uint32_t n);
// LLAMA_LSA_COMP=2: F16 block-mean cache of K [dk, n_blk] and V [dv, n_blk] for this K cache tensor (dk = head dim x KV
// heads), same epoch rule as the index cache (*valid false: rebuild every whole block this graph, then commit)
// n_ring > 0 (SJ-KVaRN): also an F16 [dk|dv, n_ring] staging ring of the newest K/V rows (row = position mod n_ring),
// because the sealed records cannot be read back as F16; the block means are taken from it (kr/vr)
// rot: the caller feeds rows in the rotated SJ-KVaRN basis (LLAMA_LSA_VK); a cache built in the other basis is refused
bool llama_lsa_mean_get(const ggml_tensor * kcache, int64_t dk, int64_t dv, int64_t n_blk, ggml_backend_buffer_type_t buft,
        ggml_tensor ** km, ggml_tensor ** vm, bool * valid, uint64_t * epoch,
        int64_t n_ring = 0, ggml_tensor ** kr = nullptr, ggml_tensor ** vr = nullptr, bool rot = false);
void llama_lsa_mean_commit(const ggml_tensor * kcache, uint64_t epoch);

// LLAMA_LSA_PICK_EVERY: per K cache tensor, an F32 [n_cap] 0/1 block selection that later graphs reuse. get() decides
// for this graph (pos0 = first row, edge = window-edge block of the last row, nbs = block count now): *reuse true =
// read *sel[0, nbs) instead of picking; false = pick and write all n_cap entries. The decision is committed in
// set_input (reserve graphs never run it), with the gen get() returned.
struct llama_lsa_pick_slot {
    ggml_tensor * sel   = nullptr;
    bool          reuse = false;
    uint64_t      gen   = 0;
    int64_t       age   = 0; // reuse: graphs since the pick graph (1 = the graph right after it)
};
// seq = the ubatch's only sequence id: a pick is reused only by the sequence that made it (callers skip the pick
// cache for ubatches with more than one sequence, whose rows must not share row 0's pick).
bool llama_lsa_pick_get(const ggml_tensor * kcache, int64_t n_cap, ggml_backend_buffer_type_t buft, int64_t pos0,
        int64_t edge, int64_t nbs, int32_t seq, llama_lsa_pick_slot * out);
// expire = true: drop the stored pick (any gen) so the next graph re-picks; the other fields are ignored
void llama_lsa_pick_commit(const ggml_tensor * kcache, uint64_t gen, bool reuse, int64_t pos0, int64_t edge, int64_t nbs,
        int32_t seq, bool expire = false);

// SJ-KVaRN + index cache: an invalid layer cannot be rebuilt from its rotated, sealed K, so it is only made valid
// again by a ubatch that starts the sequence (position 0); until then the layer stays dense. Fills the I64 query-row
// cell tensor (cell == position on SJ-KVaRN); no-op for nullptr or an unallocated tensor.
struct llama_ubatch;
void llama_lsa_set_q_cell(ggml_tensor * q_cell, const llama_ubatch * ubatch);
// LLAMA_LSA_BLOCK_SUM: I32 [n_tokens] block of each query row's index-cache row (cells / block, clamped to the last
// whole block), from the I64 row tensor the index store uses (q_cell on SJ-KVaRN, the K store rows otherwise)
void llama_lsa_set_q_blk(ggml_tensor * q_blk, const ggml_tensor * rows_i64, int64_t n_blocks);
// LLAMA_LSA_COMP=2 block-mean refresh inputs: u_blk I32 [n_u] = the distinct blocks of rows_i64 (clamped to
// n_blocks - 1, padded with the last one), u_rblk I32 [n_u] = their staging-ring blocks (block mod n_ring/B; SJ-KVaRN),
// q_ring I64 [n_tokens] = each row's staging-ring row (row mod n_ring; SJ-KVaRN). Null tensors are skipped.
// ring_rows (carry ring, LLAMA_LSA_RING_ROWS): u_rblk I32 [n_u*B] = for each row of each block, its row in the graph's
// [carry ring ; this ubatch's rows] concat (n_ring + ubatch row, else row mod n_ring), q_ring I64 [m] = carry-ring rows
// of the ubatch's last m = min(n_tokens, n_ring) rows.
void llama_lsa_set_mean_inputs(ggml_tensor * u_blk, ggml_tensor * u_rblk, ggml_tensor * q_ring, const ggml_tensor * rows_i64,
        int64_t n_blocks, int64_t n_ring, bool ring_rows = false);

// SJ-KVaRN + index cache: the prompt cache carries the index cache, because an invalid layer cannot be rebuilt from
// the sealed K. Written by llama_memory_hybrid as an optional suffix behind the recurrent state, only when
// LLAMA_LSA_IDX_CACHE=1 and at least one layer is valid, so states saved with LSA off are unchanged. Rows [0, n_rows)
// (cells == positions). The read marks each restored layer valid for the current epoch (the K restore just bumped
// it); a layer that is unknown here (or LSA off) is skipped and stays dense as before. With LLAMA_LSA_COMP=2 on
// Version 4 checks the loaded projection-map SHA256 before restoring any index layer. Legacy unbound suffixes
// are rejected with LSA on; rebuild the prompt cache. A flag says whether block means and staging rings follow.
class llama_io_write_i;
class llama_io_read_i;
void llama_lsa_idx_state_write(llama_io_write_i & io, int64_t n_rows);
void llama_lsa_idx_state_read(llama_io_read_i & io);

// one-line summary for the log
std::string llama_lsa_describe();

// One-shot LSA log lines ("... active", "... stays off"): the first graphs are built inside the server's memory-fit
// dry run, whose logger drops WARN, so a plain static flag spent its line there and server.log never showed it (the
// e2e guards then read a working arm as off). llama_context bumps the epoch once per context; a line fires once per
// epoch (a few repeats at most: dry run, main, MTP draft context).
void llama_lsa_log_epoch_bump();
bool llama_lsa_log_once(uint32_t & seen);
