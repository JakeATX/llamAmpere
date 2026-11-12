// LLAMA_LSA_RESIDENT=auto sizing (src/llama-lsa-fit.h): 8 / 12 / 24 GB dense 27B cases, host-RAM floor, memory
// projection, monotonic share, never over the room. Sizes: SJ-KVaRN 4/4 0.0166 MiB/token, 3/2t 0.01086 MiB/token
// (all KV layers); bodies are taken as 95% of that, per-record tables 2% of the body.

#include "../src/llama-lsa-fit.h"
#include "../src/llama-lsa.h"
#include "../ggml/src/ggml-lsa-recall.h"
#include "ggml-cpu.h"

#include <cstdio>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static constexpr int64_t MiB = 1ll << 20;

static llama_lsa_resident_fit_in card(double free_mib, double reserve_mib, double mib_per_token, double tokens, double keep) {
    llama_lsa_resident_fit_in in;
    const double kv = mib_per_token*tokens;
    in.free_bytes    = (int64_t) (free_mib*MiB);
    in.reserve_bytes = (int64_t) (reserve_mib*MiB);
    in.compute_bytes = (int64_t) (std::min(1024.0, reserve_mib)*MiB);
    in.body_bytes    = (int64_t) (0.95*kv*MiB);
    in.other_bytes   = (int64_t) (0.05*kv*MiB);
    in.slots_full    = in.body_bytes;
    in.fixed_bytes   = in.body_bytes/50;
    in.need_share    = std::min(1.0, 1.25*(keep*tokens + 8192 + 128)/tokens);
    in.host_avail    = 48ll << 30;
    return in;
}

static void show(const char * name, const llama_lsa_resident_fit_in & in, const llama_lsa_resident_fit_out & r) {
    printf("%-28s on=%d share=%5.1f%% card=%6.0f MiB | %s\n", name, r.on ? 1 : 0, 100.0*r.share, (double) r.card_bytes/MiB, r.why.c_str());
    if (r.on && r.share > 0.0 && !in.dry_run) {
        // when on, the chosen card bytes never exceed the room (the dry run reports the target and reads no free memory)
        CHECK(r.card_bytes <= in.free_bytes - in.reserve_bytes - in.other_bytes);
        CHECK(r.share <= 1.0);
    }
    if (!r.on && r.why.find("falling back to the non-resident path") != std::string::npos) {
        // the fall-back is only taken when the full bodies fit next to the compute buffers on a real card
        CHECK(in.free_bytes - in.other_bytes - in.compute_bytes >= in.body_bytes);
    }
    if (r.thrash) {
        CHECK(r.on && r.why.find("does not fit either") != std::string::npos);
    }
}

int main() {
    // Use the real predictor builder; check all horizons, four budgets, and quota saturation.
    for (int refine : { 128, 16 }) {
        for (int rows : { 1, 8 }) {
            for (bool saturated : { false, true }) {
                ggml_context * ctx = ggml_init({ 4*1024*1024, nullptr, false });
                CHECK(ctx != nullptr);
                if (!ctx) {
                    return 1;
                }
                const int cells = 384, units = cells/refine, per_block = 128/refine;
                const int keep = saturated ? units : refine == 16 ? 7 : 1;
                ggml_tensor * cp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cells);
                ggml_tensor * qp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rows);
                ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cells, rows);
                ggml_set_zero(mask);
                for (int c = 0; c < cells; ++c) { ((float *) cp->data)[c] = (float) c; }
                for (int r = 0; r < rows; ++r) { ((float *) qp->data)[r] = (float) (cells + r); }
                ggml_tensor * scores[3];
                for (int ahead = 0; ahead < 3; ++ahead) {
                    scores[ahead] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cells, rows);
                    for (int r = 0; r < rows; ++r) {
                        for (int c = 0; c < cells; ++c) {
                            const int rank = (c/128 - (ahead + r)%3 + 3)%3;
                            ((float *) scores[ahead]->data)[r*cells + c] = 32.0f*(3 - rank) + (float) ((c%128)/16);
                        }
                    }
                }
                const int percents[] = { 100, 110, 125, 150 };
                CHECK(GGML_LSA_RECALL_BUDGETS == 4 && GGML_LSA_RECALL_HORIZONS == 3);
                ggml_tensor * both = nullptr;
                for (int b = 0; b < GGML_LSA_RECALL_BUDGETS; ++b) {
                    CHECK(ggml_lsa_recall_budget_pct(b) == percents[b]);
                    ggml_tensor * picks[3];
                    for (int ahead = 0; ahead < 3; ++ahead) {
                        picks[ahead] = llama_lsa_build_recall_pick(ctx, scores[ahead], mask, cp, qp, 128, refine,
                                1.0f, 0, 0, saturated ? 3 : 1, keep, ggml_lsa_recall_budget_pct(b));
                        CHECK(picks[ahead]->src[4]->ne[0] == cells/128);
                        CHECK(picks[ahead]->op_params[7] == per_block);
                        CHECK(picks[ahead]->op_params[3] == std::min(units, (keep*percents[b] + 99)/100));
                        CHECK(picks[ahead]->src[4]->op_params[3] == (saturated ? 3 : b == 0 ? 1 : 2));
                    }
                    ggml_tensor * budget = ggml_concat(ctx, ggml_concat(ctx, picks[0], picks[1], 2), picks[2], 2);
                    both = both ? ggml_concat(ctx, both, budget, 3) : budget;
                }
                both = ggml_cont(ctx, both);
                CHECK(both->ne[0] == units && both->ne[1] == rows && both->ne[2] == 3 && both->ne[3] == 4);
                ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, false);
                ggml_build_forward_expand(graph, both);
                CHECK(ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS);
                for (int b = 0; b < GGML_LSA_RECALL_BUDGETS; ++b) {
                    const int quota = std::min(units, (keep*percents[b] + 99)/100);
                    for (int ahead = 0; ahead < 3; ++ahead) {
                        for (int r = 0; r < rows; ++r) {
                            for (int u = 0; u < units; ++u) {
                                const int rank = (u/per_block - (ahead + r)%3 + 3)%3;
                                const bool want = rank*per_block + per_block - 1 - u%per_block < quota;
                                CHECK(((float *) both->data)[((b*3 + ahead)*rows + r)*units + u] == (want ? 1.0f : 0.0f));
                            }
                        }
                    }
                }
                ggml_free(ctx);
            }
        }
    }
    // Missing-page recall: hits must not inflate it; an unavailable newly sealed page still costs a miss.
    // These are physical pages after row union, with a true positive, false positive, false negative and new seal.
    {
        ggml_lsa_recall_counts c{};
        c.observe(true,  true,  true,  true);  // resident hit, ignored
        c.observe(false, true,  true,  true);  // true positive
        c.observe(false, false, true,  true);  // false positive
        c.observe(false, true,  false, true);  // false negative
        c.observe(false, true,  true,  false); // new seal: prediction cannot use future data
        c.observe(false, false, false, true);  // irrelevant missing page
        CHECK(c.actual_new == 3 && c.predicted_new == 2 && c.hit == 1 && c.newly_sealed == 1);
        CHECK(c.predicted_new - c.hit == 1); // false-byte count, multiplied by physical record stride
        ggml_lsa_recall_counts empty{};
        empty.observe(true, true, true, true);
        CHECK(empty.actual_new == 0 && empty.hit == 0 && empty.predicted_new == 0);
        c.add(empty);
        CHECK(c.actual_new == 3 && c.hit == 1);
    }
    // A page can remain live while its strips turn over. Never count old card hits as page prediction recall.
    {
        ggml_lsa_workset_counts w{};
        w.observe(0x05, 0x03); // one shared strip, one new, same page
        w.observe(0x00, 0x80); // page and strip dropped
        w.observe(0x80, 0x00); // page and strip newly touched
        CHECK(w.pages == 2 && w.strips == 3 && w.prev_pages == 2 && w.prev_strips == 3);
        CHECK(w.new_pages == 1 && w.new_strips == 2);
        ggml_lsa_workset_counts partial{};
        partial.observe(0x01, 0); // only the last partial strip exists; padded cells are absent
        w.add(partial);
        CHECK(w.pages == 3 && w.strips == 4 && w.new_pages == 2 && w.new_strips == 3);
        for (uint32_t n : { 0u, 128u, 99968u, 100000u, 100001u, 100095u, 100096u }) {
            const uint32_t sealed = n - n%128;
            CHECK(!ggml_lsa_workset_partial_page(sealed));
            CHECK(ggml_lsa_workset_partial_page(n) == (n != sealed));
        }
        CHECK(ggml_lsa_history_contiguous(100, 3, 100, 101, 1, 4));
        CHECK(!ggml_lsa_history_contiguous(100, 0, 100, 101, 1, 1)); // cold/reset
        CHECK(!ggml_lsa_history_contiguous(100, 3, 90, 91, 1, 4)); // restore/replay
        CHECK(!ggml_lsa_history_contiguous(100, 3, 100, 101, 1, 5)); // missing resolve
        CHECK(!ggml_lsa_history_contiguous(100, 3, 100, 103, 1, 4)); // non-append geometry
    }
    // The diagnostic source must never turn on the block-list, virtual-key, or LSE attention paths.
    {
        ggml_init_params params{64*1024, nullptr, true};
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * fa = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        fa->op = GGML_OP_FLASH_ATTN_EXT;
        fa->src[0] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 2, 8, 1);
        fa->src[6] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        CHECK(ggml_flash_attn_ext_get_lsa_spec_recall(fa) == nullptr);
        ggml_tensor * pick = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 2, 3, 4);
        ggml_flash_attn_ext_set_lsa_spec_recall(fa, pick);
        CHECK(ggml_flash_attn_ext_get_lsa_spec_recall(fa) == pick);
        CHECK(ggml_flash_attn_ext_get_lsa_block(fa) == 0);
        CHECK(!ggml_flash_attn_ext_get_lsa_vk(fa));
        CHECK(ggml_flash_attn_ext_get_lse(fa) == nullptr);
        ggml_free(ctx);
    }
    // Only the exact-score diagnostics can omit a map; a supplied map keeps the old layer and width checks.
    for (auto mode : { llama_lsa_config::OFF, llama_lsa_config::KEY, llama_lsa_config::POS,
                       llama_lsa_config::ORACLE, llama_lsa_config::MASS, llama_lsa_config::QUEST }) {
        llama_lsa_config c;
        c.mode = mode;
        CHECK(c.mapless_probe() == (mode == llama_lsa_config::ORACLE || mode == llama_lsa_config::MASS));
        c.map_path = "map.gguf";
        CHECK(!c.mapless_probe());
    }
    // 24 GB 3090: ~8,000 MiB free after the 27B weights, 4/4 KV: the body fits at 200K and 256K -> residency off
    for (double t : { 200e3, 256e3 }) {
        auto in = card(8000, 1536, 0.0166, t, 0.25);
        auto r = llama_lsa_resident_fit(in);
        show(t > 210e3 ? "24GB 4/4 256K" : "24GB 4/4 200K", in, r);
        CHECK(!r.on);
        CHECK(r.card_bytes == in.body_bytes);
    }
    // 12 GB: ~3,300 MiB free, 3/2t. 128K fits (off); 260K at 25% keep: on, below the 80% target but above the read set
    {
        auto in = card(3300, 1536, 0.01086, 128e3, 0.25);
        auto r = llama_lsa_resident_fit(in);
        show("12GB 3/2t 128K", in, r);
        CHECK(!r.on);
    }
    {
        auto in = card(3300, 1536, 0.01086, 260e3, 0.25);
        auto r = llama_lsa_resident_fit(in);
        show("12GB 3/2t 260K keep25", in, r);
        CHECK(r.on);
        CHECK(r.share > in.need_share && r.share < 0.80);
        CHECK(r.why.find("below the 80% target") != std::string::npos);
    }
    // 12 GB 3/2t 192K: just short of fitting -> on and >= 80%
    {
        auto in = card(3300, 1536, 0.01086, 192e3, 0.25);
        auto r = llama_lsa_resident_fit(in);
        show("12GB 3/2t 192K keep25", in, r);
        CHECK(r.on);
        CHECK(r.share >= 0.80 && r.share < 1.0);
        CHECK(r.why.find("meets the 80% target") != std::string::npos);
    }
    // 8 GB (7,680 MiB cap): ~1,700 MiB free, 1,024 MiB reserve. 64K: >= 80%; 260K: below the read set, and the 2.7 GB
    // non-resident body can't fit an 8 GB card either -> slot pool kept (same output, slow), keep untouched
    {
        auto in = card(1700, 1024, 0.01086, 64e3, 0.25);
        auto r = llama_lsa_resident_fit(in);
        show("8GB 3/2t 64K keep25", in, r);
        CHECK(r.on && r.share >= 0.80);
    }
    {
        auto in = card(1700, 1024, 0.01086, 260e3, 0.25);
        auto r = llama_lsa_resident_fit(in);
        show("8GB 3/2t 260K keep25", in, r);
        CHECK(r.on && r.thrash && r.keep_cap == 0.0 && r.share < in.need_share);
        CHECK(r.why.find("LLAMA_LSA_RESIDENT_LOWER_KEEP=1") != std::string::npos);
    }
    // 8 GB with almost nothing left: the body can't fit either -> slot pool (no room line), keep untouched
    {
        auto in = card(1100, 1024, 0.01086, 260e3, 0.25);
        auto r = llama_lsa_resident_fit(in);
        show("8GB no room", in, r);
        CHECK(r.on && r.thrash && r.keep_cap == 0.0);
    }
    // 50% keep, knob off: 8 GB and 12 GB 260K: the non-resident body (2.7 GB) does not fit next to the compute buffers
    // on either card -> slot pool kept, keep untouched (falling back would fail to allocate on a real card)
    for (double f : { 1700.0, 3300.0 }) {
        auto in = card(f, f < 2000 ? 1024 : 1536, 0.01086, 260e3, 0.5);
        auto r = llama_lsa_resident_fit(in);
        show(f < 2000 ? "8GB 260K keep50 pool" : "12GB 260K keep50 pool", in, r);
        CHECK(r.on && r.thrash && r.keep_cap == 0.0 && r.share < in.need_share);
    }
    // af1/kf1 fmid case (98K, 3/2t, emulated free 2,486 MiB): 62% share < 73% read set, but the 1,156 MiB body fits
    // next to 1,024 MiB compute (2486 - 235 - 1024 = 1227) -> non-resident fall-back, bodies on the card
    // flo (free 2,236): 977 MiB < 1,156 -> slot pool kept (same output, slow)
    for (double f : { 2486.0, 2236.0 }) {
        llama_lsa_resident_fit_in in;
        in.free_bytes = (int64_t) (f*MiB); in.reserve_bytes = 1536*MiB; in.compute_bytes = 1024*MiB;
        in.other_bytes = 235*MiB; in.body_bytes = 1156*MiB; in.slots_full = in.body_bytes; in.fixed_bytes = 0;
        in.need_share = 0.73; in.host_avail = 48ll << 30;
        auto r = llama_lsa_resident_fit(in);
        show(f > 2400 ? "fmid 98K fallback" : "flo 98K pool", in, r);
        if (f > 2400) {
            CHECK(!r.on && r.card_bytes == in.body_bytes && r.why.find("falling back to the non-resident path") != std::string::npos);
        } else {
            CHECK(r.on && r.thrash);
        }
    }
    // RAM fit rule: the pinned master must leave 5 GiB of host RAM, else bodies stay on the card
    {
        auto in = card(3300, 1536, 0.01086, 260e3, 0.25);
        in.host_avail = 6ll << 30;
        auto r = llama_lsa_resident_fit(in);
        show("12GB 260K host 6 GiB", in, r);
        CHECK(!r.on);
        CHECK(r.why.find("host RAM") != std::string::npos);
    }
    // memory projection (no_alloc fit pass): reports the target share, no free-memory read
    {
        auto in = card(0, 1536, 0.01086, 260e3, 0.25);
        in.dry_run = true;
        auto r = llama_lsa_resident_fit(in);
        show("dry run", in, r);
        CHECK(r.on && r.share == 0.80);
        CHECK(r.card_bytes == in.fixed_bytes + (int64_t) std::ceil(0.80*in.slots_full));
    }
    // no eligible layer
    {
        llama_lsa_resident_fit_in in;
        in.free_bytes = 1ll << 34;
        auto r = llama_lsa_resident_fit(in);
        CHECK(!r.on);
    }
    // as free VRAM grows: slot pool below the read set -> resident at/above it (share never drops) -> off once the body
    // fits; never back. A non-resident fall-back may only appear where its bodies fit (checked in show)
    for (double keep : { 0.25, 0.5 }) {
        double last = -1.0;
        int phase = 0, seen_on = 0;
        for (double f = 1000; f <= 6000; f += 50) {
            auto in = card(f, 1536, 0.01086, 260e3, keep);
            auto r = llama_lsa_resident_fit(in);
            show("sweep", in, r);
            const int p = r.thrash ? 0 : r.on ? 1 : 2;
            CHECK(p >= phase);
            phase = p;
            if (r.on) {
                CHECK(r.share >= last);
                CHECK(r.thrash || r.share >= in.need_share);
                last = r.share;
                seen_on++;
            }
        }
        CHECK(phase == 2 && seen_on > 0);
    }
    // LLAMA_LSA_RESIDENT_LOWER_KEEP (opt-in): below the read set, keep is capped so 1.25*(keep + sinks/window) fits the share
    auto fit = [](llama_lsa_resident_fit_in in, double keep, double tokens) {
        in.keep_fit = true; in.keep_now = keep; in.keep_min = 0.25; in.fixed_share = (8192 + 128)/tokens;
        return in;
    };
    {   // 12 GB 3/2t 260K at 50% keep: share ~58% < read set ~66% -> keep capped between 25% and 50%, then it fits
        auto in = fit(card(3300, 1536, 0.01086, 260e3, 0.5), 0.5, 260e3);
        auto r = llama_lsa_resident_fit(in);
        show("12GB 3/2t 260K keep50 fit", in, r);
        CHECK(r.on && r.share < in.need_share);
        CHECK(r.keep_cap > 0.25 && r.keep_cap < 0.5);
        CHECK(1.25*(r.keep_cap + in.fixed_share) <= r.share + 1e-9);
        CHECK(r.why.find("keep lowered") != std::string::npos && r.why.find("thrash") == std::string::npos);
        auto off = in; off.keep_fit = false;  // knob off: no cap; the body can't fit either -> slot pool, keep untouched
        auto r0 = llama_lsa_resident_fit(off);
        CHECK(r0.on && r0.thrash && r0.keep_cap == 0.0);
    }
    {   // 8 GB 3/2t 260K at 50% keep: even the 25% floor does not fit and neither does the body -> knob on: pool at the
        // 25% floor (the user opted into lossy keep)
        auto in = fit(card(1700, 1024, 0.01086, 260e3, 0.5), 0.5, 260e3);
        auto r = llama_lsa_resident_fit(in);
        show("8GB 3/2t 260K keep50 fit", in, r);
        CHECK(r.on && r.thrash && r.keep_cap == 0.25);
        CHECK(r.why.find("(floor)") != std::string::npos);
    }
    {   // fmid-like with the knob on but the floor still short and the body fits: non-resident, cap dropped
        llama_lsa_resident_fit_in in;
        in.free_bytes = 2486*MiB; in.reserve_bytes = 1536*MiB; in.compute_bytes = 1024*MiB; in.other_bytes = 235*MiB;
        in.body_bytes = 1156*MiB; in.slots_full = in.body_bytes; in.need_share = 0.95; in.host_avail = 48ll << 30;
        in.keep_fit = true; in.keep_now = 0.7; in.keep_min = 0.5; in.fixed_share = 0.06; // cap 0.5 still reads 70% > 62%
        auto r = llama_lsa_resident_fit(in);
        show("fallback after keep floor", in, r);
        CHECK(!r.on && r.keep_cap == 0.0 && r.why.find("even after lowering keep") != std::string::npos);
    }
    {   // share already above the read set: no cap
        auto in = fit(card(3300, 1536, 0.01086, 192e3, 0.25), 0.25, 192e3);
        auto r = llama_lsa_resident_fit(in);
        CHECK(r.on && r.keep_cap == 0.0);
    }
    {   // keep already at the floor: nothing to lower, body can't fit -> slot pool, no cap
        auto in = fit(card(1700, 1024, 0.01086, 260e3, 0.25), 0.25, 260e3);
        auto r = llama_lsa_resident_fit(in);
        CHECK(r.on && r.thrash && r.keep_cap == 0.0);
    }
    {   // dry run: no cap (no free memory is read)
        auto in = fit(card(0, 1536, 0.01086, 260e3, 0.5), 0.5, 260e3);
        in.dry_run = true;
        CHECK(llama_lsa_resident_fit(in).keep_cap == 0.0);
    }
    {   // once on, the cap never drops as free VRAM grows (0 = no cap counts as 50%); fall-back rows carry no cap
        double last = 0.0;
        for (double f = 1000; f <= 6000; f += 50) {
            auto r = llama_lsa_resident_fit(fit(card(f, 1536, 0.01086, 260e3, 0.5), 0.5, 260e3));
            if (!r.on) {
                CHECK(r.keep_cap == 0.0);
                continue;
            }
            const double k = r.keep_cap > 0.0 ? r.keep_cap : 0.5;
            CHECK(k >= last - 1e-12 && k <= 0.5 && k >= 0.25);
            last = k;
        }
    }
    {   // measured compute (memory-check pass) replaces the default reserve unless the env set it
        const int64_t R = 1536*MiB, C = 1024*MiB;
        auto d = llama_lsa_resident_reserve(R, C, false, false, 0);              // no measurement: defaults
        CHECK(!d.measured && d.reserve_bytes == R && d.compute_bytes == C);
        auto m = llama_lsa_resident_reserve(R, C, false, false, 860*MiB);        // 860 + 64 slack, + 512 margin
        CHECK(m.measured && m.compute_bytes == 924*MiB && m.reserve_bytes == 1436*MiB);
        auto big = llama_lsa_resident_reserve(R, C, false, false, 2000*MiB);     // 5% slack above 1280 MiB
        CHECK(big.compute_bytes == 2100*MiB && big.reserve_bytes == 2612*MiB);
        auto e = llama_lsa_resident_reserve(R, C, true, true, 860*MiB);          // both env: untouched
        CHECK(!e.measured && e.reserve_bytes == R && e.compute_bytes == C);
        auto ec = llama_lsa_resident_reserve(R, C, false, true, 860*MiB);        // compute env only: reserve measured
        CHECK(ec.measured && ec.compute_bytes == C && ec.reserve_bytes == 1436*MiB);
        auto er = llama_lsa_resident_reserve(700*MiB, C, true, false, 860*MiB);  // reserve env below compute: compute capped
        CHECK(er.reserve_bytes == 700*MiB && er.compute_bytes == 700*MiB);
        // free margin override (LLAMA_LSA_RESIDENT_MARGIN_MIB / explicit --fit-target): replaces the 512 MiB default
        auto g0 = llama_lsa_resident_reserve(R, C, false, false, 0, 256*MiB);       // no measurement: 1024 + 256
        CHECK(g0.margin_set && g0.compute_bytes == C && g0.reserve_bytes == 1280*MiB && g0.margin_bytes == 256*MiB);
        auto g1 = llama_lsa_resident_reserve(R, C, false, false, 860*MiB, 0);       // measured, zero margin: 924 + 0
        CHECK(g1.margin_set && g1.measured && g1.reserve_bytes == 924*MiB && g1.margin_bytes == 0);
        auto g2 = llama_lsa_resident_reserve(R, C, false, false, 860*MiB, 1024*MiB); // measured, 1 GiB margin
        CHECK(g2.reserve_bytes == 1948*MiB && g2.compute_bytes == 924*MiB);
        auto g3 = llama_lsa_resident_reserve(R, C, true, false, 860*MiB, 256*MiB);  // explicit reserve wins over margin
        CHECK(!g3.margin_set && g3.reserve_bytes == R);
        auto g4 = llama_lsa_resident_reserve(R, C, false, false, 860*MiB, -1);      // unset == old call
        CHECK(!g4.margin_set && g4.reserve_bytes == m.reserve_bytes && g4.compute_bytes == m.compute_bytes && g4.margin_bytes == 512*MiB);
        // 8 GB card, 7,680 MiB process cap = --fit-target 512 on 8,192 MiB: same reserve as the default 512 margin; a
        // tighter --fit-target 256 frees 256 MiB more for slots, a looser 1024 takes 512 MiB from them. No measurement
        // (compute 1024), so the fall-back does not fit (977 < 1156 MiB body) and the slot pool is what moves.
        auto c8 = [&](int64_t margin) {
            llama_lsa_resident_fit_in in = card(2236, 1536, 0.01086, 98e3, 0.422);
            in.other_bytes = 235*MiB; in.body_bytes = 1156*MiB; in.slots_full = in.body_bytes; in.fixed_bytes = in.body_bytes/50;
            auto rv = llama_lsa_resident_reserve(R, C, false, false, 0, margin);
            in.reserve_bytes = rv.reserve_bytes; in.compute_bytes = rv.compute_bytes;
            return llama_lsa_resident_fit(in);
        };
        const auto r512 = c8(512*MiB), rdef = c8(-1), r256 = c8(256*MiB), r1024 = c8(1024*MiB);
        CHECK(r512.on == rdef.on && std::fabs(r512.share - rdef.share) < 1e-12 && r512.card_bytes == rdef.card_bytes);
        CHECK(r512.on && r256.on && r256.card_bytes > r512.card_bytes);
        CHECK(!r1024.on || r1024.card_bytes <= r512.card_bytes);
        // af1 flo card (free 2236, other 235, body 1156): default compute 1024 -> 977 MiB < body -> slot pool; a measured
        // 700 MiB compute buffer (-ub 256) -> 764 -> 1237 >= body -> non-resident fall-back (output == residency off)
        auto flo = [&](const llama_lsa_resident_reserve_out & rv) {
            llama_lsa_resident_fit_in in = card(2236, 1536, 0.01086, 98e3, 0.422);
            in.other_bytes = 235*MiB; in.body_bytes = 1156*MiB; in.slots_full = in.body_bytes; in.fixed_bytes = in.body_bytes/50;
            in.reserve_bytes = rv.reserve_bytes; in.compute_bytes = rv.compute_bytes;
            auto r = llama_lsa_resident_fit(in);
            show(rv.measured ? "af1 flo measured 700" : "af1 flo defaults", in, r);
            return r;
        };
        auto r0 = flo(d);
        CHECK(r0.on && r0.thrash);
        auto r1 = flo(llama_lsa_resident_reserve(R, C, false, false, 700*MiB));
        CHECK(!r1.on && r1.why.find("falling back to the non-resident path") != std::string::npos);
    }
    // auto stage (record pages): kf3 pool_lo card (free 2236, other 235, body 1156 = 16 layers x 831 records, read set 73%):
    // the slot pool is kept (thrash); the stage takes the read-set excess, all 16 layers share it, so the slots shrink by
    // only ~1/16 of the staged share; off (or no thrash) = exactly the old answer and no stage
    {
        auto base = [&](double f, bool st) {
            llama_lsa_resident_fit_in in;
            in.free_bytes = (int64_t) (f*MiB); in.reserve_bytes = 1536*MiB; in.compute_bytes = 1024*MiB;
            in.other_bytes = 235*MiB; in.body_bytes = 1156*MiB; in.slots_full = in.body_bytes; in.fixed_bytes = 0;
            in.need_share = 0.73; in.host_avail = 48ll << 30;
            in.stage_auto = st; in.records = 831; in.stage_rec_bytes = in.body_bytes/(16*831);
            return in;
        };
        auto i0 = base(2236, false), i1 = base(2236, true);
        auto r0 = llama_lsa_resident_fit(i0), r1 = llama_lsa_resident_fit(i1);
        show("pool_lo no stage", i0, r0);
        show("pool_lo auto stage", i1, r1);
        CHECK(r0.thrash && r0.stage_records == 0);
        CHECK(r1.thrash && r1.stage_records > 0);
        CHECK(r1.stage_records >= (int64_t) std::floor((i1.need_share - r1.share)*831));     // covers the excess
        CHECK(r1.stage_records <= (int64_t) std::ceil((i1.need_share - r1.share)*831) + 1);
        CHECK(r1.share < r0.share && r0.share - r1.share < 0.05);                          // slots shrink by ~1/16 only
        CHECK(r1.card_bytes <= i1.free_bytes - i1.reserve_bytes - i1.other_bytes);
        CHECK(r1.why.find("AUTO_STAGE=0") != std::string::npos);
        auto i2 = base(2486, true);                                                         // fall-back case: no stage
        auto r2 = llama_lsa_resident_fit(i2);
        show("fmid auto stage", i2, r2);
        CHECK(!r2.on && r2.stage_records == 0);
        auto i3 = base(3600, true);                                                         // pool meets the read set: no stage
        auto r3 = llama_lsa_resident_fit(i3);
        show("roomy auto stage", i3, r3);
        CHECK(!r3.thrash && r3.stage_records == 0);
        auto i4 = base(2236, true); i4.stage_max = 64;                                      // cap respected
        auto r4 = llama_lsa_resident_fit(i4);
        CHECK(r4.thrash && r4.stage_records == 64);
        auto i5 = base(1800, true);                                                         // almost no room: stage never overdraws
        auto r5 = llama_lsa_resident_fit(i5);
        show("tiny auto stage", i5, r5);
        CHECK(r5.card_bytes <= std::max<int64_t>(0, i5.free_bytes - i5.reserve_bytes - i5.other_bytes) || r5.share == 0.0);

        auto all = base(2236, true);
        all.stage_all = true;
        all.stage_max = 1; // the legacy cap must not leave a full-stage decode reading host KV
        auto full = llama_lsa_resident_fit(all);
        show("full read-set stage", all, full);
        CHECK(full.on && full.thrash && full.stage_records == all.records);
        CHECK(full.keep_cap == 0.0 && full.share < r1.share);
        const int64_t stage_bytes = all.records*all.stage_rec_bytes;
        CHECK(full.card_bytes == all.fixed_bytes + (int64_t) std::floor(full.share*all.slots_full) + stage_bytes);
        CHECK(full.card_bytes <= all.free_bytes - all.reserve_bytes - all.other_bytes);
        for (int64_t records : { 1ll, 831ll, 2048ll }) {
            all.records = records;
            all.stage_rec_bytes = 4096;
            const auto exact = llama_lsa_resident_fit(all);
            if (records == 1) {
                CHECK(!exact.on && exact.stage_records == 0); // one slot is the whole body
            } else {
                CHECK(exact.on && exact.stage_records == records && exact.keep_cap == 0.0);
                CHECK(exact.card_bytes <= all.free_bytes - all.reserve_bytes - all.other_bytes);
            }
        }
        all = base(2486, true); all.stage_all = true;
        const auto fallback = llama_lsa_resident_fit(all);
        CHECK(!fallback.on && fallback.stage_records == 0 && fallback.keep_cap == 0.0);
        all = base(2236, true); all.stage_all = true; all.dry_run = true;
        const auto projection = llama_lsa_resident_fit(all);
        CHECK(projection.on && projection.stage_records == all.records);
        CHECK(projection.card_bytes == all.fixed_bytes + (int64_t) std::ceil(all.target_share*all.slots_full) + stage_bytes);
        all = base(2236, true); all.stage_all = true; all.stage_rec_bytes = INT64_MAX;
        CHECK(!llama_lsa_resident_fit(all).on); // reject the product before signed overflow
        all.stage_rec_bytes = MiB;
        CHECK(!llama_lsa_resident_fit(all).on); // full stage leaves no room for slots

        all = base(2236, true); all.stage_all = true; all.fixed_bytes = 17*1024;
        const int64_t minimum = all.slots_full/all.records + (all.slots_full % all.records != 0);
        const int64_t fixed_stage = all.fixed_bytes + all.records*all.stage_rec_bytes;
        for (int64_t delta : { -1ll, 0ll, 1ll }) {
            all.free_bytes = all.reserve_bytes + all.other_bytes + fixed_stage + minimum + delta;
            const auto edge = llama_lsa_resident_fit(all);
            CHECK(edge.on == (delta >= 0));
            CHECK(edge.keep_cap == 0.0);
            if (edge.on) {
                CHECK(edge.card_bytes >= fixed_stage + minimum);
                CHECK(edge.card_bytes <= all.free_bytes - all.reserve_bytes - all.other_bytes);
            }
        }
        all.dry_run = true; all.target_share = 0.0;
        CHECK(llama_lsa_resident_fit(all).card_bytes == fixed_stage + minimum);
    }
    printf("%s (%d failures)\n", n_fail ? "FAIL" : "OK", n_fail);
    return n_fail ? 1 : 0;
}
