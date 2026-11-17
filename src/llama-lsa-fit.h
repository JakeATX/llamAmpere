#pragma once

// LLAMA_LSA_RESIDENT=auto (opt-in, LSA only): pick how much of the LSA layers' sealed KV body stays on the card from
// the VRAM that is left after the model, the rest of the KV cache and the compute buffers still to come. Pure function
// (no ggml, no I/O) so tests/test-lsa-resident-fit.cpp can check it for 8/12/24 GB cards.
//
// Card cost of residency is linear in the share s of records that get a slot: fixed + s*slots_full (fixed = the per-
// record tables every resident layer keeps on the card; slots_full = the slot bytes if every record had one).
// Measured cost of a smaller share (rs3, 3/2t 128-token union pick at 50% keep, 98K, PCIe 3.0 x16): 80% on card about
// +0.5 ms/token, 65% +2.3 ms, 55% +8 ms; so the target is >= 80% and anything below is logged as a slow fit.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

struct llama_lsa_resident_fit_in {
    int64_t free_bytes   = 0;   // device free now (weights already loaded)
    int64_t reserve_bytes = 0;  // compute buffers + VRAM margin that still have to fit after the KV cache
    int64_t compute_bytes = -1; // hard floor for compute buffers alone (no margin): the non-resident fall-back is only
                                // taken when free - other - compute >= body; -1 = reserve_bytes (fall-back never fits)
    int64_t other_bytes  = 0;   // KV the card holds anyway (tail rings, other layers, edge tier, metadata)
    int64_t body_bytes   = 0;   // all residency-eligible sealed bodies together, on-card size when residency is off
    int64_t fixed_bytes  = 0;   // card bytes residency adds regardless of share (per-record tables)
    int64_t slots_full   = 0;   // card slot bytes at share 1.0
    double  need_share   = 0.0; // share one decode row reads at full context (sinks + window + keep, x1.25); below = thrash
    double  target_share = 0.80;
    int64_t host_avail   = -1;  // MemAvailable in bytes, -1 = unknown (no host check)
    int64_t host_floor   = 5ll << 30; // RAM fit rule: leave 5 GiB free after the pinned master
    bool    dry_run      = false; // memory projection without allocation (fit pass): report the target share
    // Below need_share the default is the non-resident path (bodies on the card, output == residency off): auto never
    // trades quality for speed on its own (af1: 40% on card thrashes, 43.7 vs 14.4 ms/token non-resident).
    // LLAMA_LSA_RESIDENT_LOWER_KEEP=1 (opt-in, lossy; old name LLAMA_LSA_RESIDENT_KEEP_FIT): lower the keep fraction until
    // one row's read set fits the slots, never under keep_min; if even keep_min does not fit, non-resident as above
    bool    keep_fit     = false;
    double  keep_now     = 0.0;   // keep fraction of the full context the pick uses now (absolute keep: keep/kv)
    double  keep_min     = 0.25;
    double  fixed_share  = 0.0;   // (sinks + window)/kv: cells every row reads regardless of keep
    // Auto stage (record pages, LLAMA_LSA_RESIDENT_AUTO_STAGE, default on under auto): when the slot pool is kept below
    // one step's read set (thrash), size one on-card stage per buffer for the records past the slots. Its records are
    // copied in each resolve at page-copy speed instead of read over the bus by the attention kernel (kf3 pool_lo: ~18
    // MiB/layer/step past the slots at ~3.4 GB/s). All layers of a buffer share it, so a stage record costs 1/layers of a
    // slot record. Same bytes, so output never depends on it.
    bool    stage_auto   = false;
    bool    stage_all    = false; // one shared stage holds the worst-case read set (all records); never changes keep
    int64_t records      = 0;     // records per layer (stage sizing unit)
    int64_t stage_rec_bytes = 0;  // card bytes of one stage record (0 = no stage possible)
    int64_t stage_max    = 4096;  // cap on stage records
};

struct llama_lsa_resident_fit_out {
    bool        on    = false; // false = bodies stay on the card (byte-identical to residency off)
    double      share = 0.0;   // share of records with a card slot when on
    int64_t     card_bytes = 0; // fixed + share*slots_full when on, body_bytes when off
    double      keep_cap = 0.0; // keep_fit: cap on the keep fraction (0 = no cap); never above keep_now
    bool        thrash = false; // on, below the read set, because the non-resident fall-back does not fit the card
    int64_t     stage_records = 0; // full stage, or stage_auto + thrash; its bytes are inside card_bytes
    std::string why;
};

static inline llama_lsa_resident_fit_out llama_lsa_resident_fit(const llama_lsa_resident_fit_in & in) {
    constexpr double MiB = 1024.0*1024.0;
    llama_lsa_resident_fit_out out;
    char buf[640];
    auto off = [&](std::string reason) {
        out.on = false; out.share = 0.0; out.card_bytes = in.body_bytes; out.stage_records = 0; out.why = std::move(reason);
        return out;
    };
    if (in.body_bytes <= 0 || in.slots_full <= 0) {
        return off("no residency-eligible LSA layer: bodies stay on the card");
    }
    const double target = std::min(1.0, std::max(0.0, in.target_share));
    int64_t stage_bytes = 0;
    int64_t slots_min = 0;
    if (in.stage_all) {
        if (in.records <= 0 || in.stage_rec_bytes <= 0 || in.records > INT64_MAX/in.stage_rec_bytes) {
            return off("invalid full-stage size: bodies stay on the card");
        }
        stage_bytes = in.records*in.stage_rec_bytes;
        slots_min = in.slots_full/in.records + (in.slots_full % in.records != 0);
    }
    if (in.dry_run) {
        out.on = target < 1.0; out.share = target;
        out.card_bytes = out.on ? in.fixed_bytes + std::max(slots_min, (int64_t) std::ceil(target*in.slots_full)) + stage_bytes : in.body_bytes;
        out.stage_records = out.on && in.stage_all ? in.records : 0;
        snprintf(buf, sizeof(buf), "memory projection: target share %.0f%% (%.0f MiB on the card)", 100.0*target, out.card_bytes/MiB);
        out.why = buf;
        return out;
    }
    const int64_t room_full = in.free_bytes - in.reserve_bytes - in.other_bytes;
    if (room_full >= in.body_bytes) {
        snprintf(buf, sizeof(buf), "whole body fits: %.0f MiB free - %.0f reserve - %.0f other KV = %.0f MiB >= %.0f MiB body; bodies stay on the card",
            in.free_bytes/MiB, in.reserve_bytes/MiB, in.other_bytes/MiB, room_full/MiB, in.body_bytes/MiB);
        return off(buf);
    }
    if (in.host_avail >= 0 && in.host_avail - in.body_bytes < in.host_floor) {
        snprintf(buf, sizeof(buf), "pinned host master needs %.0f MiB but only %.0f MiB host RAM is available (keeps %.0f MiB free); bodies stay on the card and may not fit",
            in.body_bytes/MiB, in.host_avail/MiB, in.host_floor/MiB);
        return off(buf);
    }
    if (in.stage_all && (room_full <= in.fixed_bytes || stage_bytes > room_full - in.fixed_bytes ||
            room_full - in.fixed_bytes - stage_bytes < slots_min)) {
        return off("full stage leaves less than one CLOCK slot per layer: bodies stay on the card and may not fit");
    }
    const int64_t room = room_full - stage_bytes;
    double s = (double) (room - in.fixed_bytes)/(double) in.slots_full;
    s = std::min(s, 1.0);
    const bool none = s <= 0.0;
    s = std::max(s, 0.0);
    out.on = true;
    out.share = s;
    out.card_bytes = in.fixed_bytes + std::max(slots_min, (int64_t) std::floor(s*in.slots_full)) + stage_bytes;
    out.stage_records = in.stage_all ? in.records : 0;
    // keep_fit: one row reads 1.25*(keep + fixed_share) of the context; solve for the keep that fits the share
    char fitbuf[160] = "";
    double need = in.need_share;
    if (in.keep_fit && s < need && in.keep_now > 0.0) {
        const double kmin = std::min(in.keep_now, std::max(0.0, in.keep_min));
        const double kfit = std::max(kmin, std::min(in.keep_now, s/1.25 - in.fixed_share));
        if (kfit < in.keep_now) {
            out.keep_cap = kfit;
            need = kfit > kmin ? s : std::min(1.0, 1.25*(kfit + in.fixed_share)); // solved: read set == share exactly
            snprintf(fitbuf, sizeof(fitbuf), "; keep lowered %.1f%% -> %.1f%% of the context%s (LLAMA_LSA_RESIDENT_LOWER_KEEP, lossy)",
                100.0*in.keep_now, 100.0*kfit, kfit <= kmin && s < need ? " (floor)" : " so one step's reads fit");
        }
    }
    const int64_t cfloor = in.compute_bytes >= 0 ? in.compute_bytes : in.reserve_bytes;
    const int64_t fb_room = in.free_bytes - in.other_bytes - cfloor;
    if (s < need && fb_room < in.body_bytes) {
        // the non-resident bodies do not fit next to the compute buffers either: falling back would fail to allocate on
        // a real card, so keep the slot pool (same output as residency off, only slower) and say so
        snprintf(buf, sizeof(buf), "%.1f%% of records on the card (%.0f MiB room vs %.0f MiB body), below the %.1f%% one step reads at keep %.1f%%; "
            "the non-resident fall-back does not fit either (%.0f MiB free - %.0f other KV - %.0f compute = %.0f MiB < %.0f MiB body): "
            "keeping the slot pool, output unchanged but SLOW (af1: 40%% on card = 3x); lower the context%s",
            100.0*s, room/MiB, in.body_bytes/MiB, 100.0*need, 100.0*(out.keep_cap > 0.0 ? out.keep_cap : in.keep_now),
            in.free_bytes/MiB, in.other_bytes/MiB, cfloor/MiB, fb_room/MiB, in.body_bytes/MiB,
            in.keep_fit ? "" : " or set LLAMA_LSA_RESIDENT_LOWER_KEEP=1 (lossy)");
        out.why = std::string(buf) + fitbuf;
        out.thrash = true;
        if (in.stage_auto && !in.stage_all && in.records > 0 && in.stage_rec_bytes > 0) {
            // stage = the read-set excess over the slots; its bytes come out of the slot room, so solve twice
            double ss = s;
            int64_t st = 0;
            for (int it = 0; it < 3; ++it) {
                st = std::min<int64_t>(in.stage_max, (int64_t) std::ceil(std::max(0.0, need - ss)*(double) in.records));
                ss = std::max(0.0, std::min(1.0, (double) (room - in.fixed_bytes - st*in.stage_rec_bytes)/(double) in.slots_full));
            }
            if (st > 0 && room - in.fixed_bytes - st*in.stage_rec_bytes > 0) {
                out.share = ss;
                out.stage_records = st;
                out.card_bytes = in.fixed_bytes + (int64_t) std::floor(ss*in.slots_full) + st*in.stage_rec_bytes;
                snprintf(buf, sizeof(buf), "; stage %lld records (%.1f MiB, shared by the layers) for the reads past the slots, slots now %.1f%% "
                    "(LLAMA_LSA_RESIDENT_AUTO_STAGE=0 turns it off)", (long long) st, st*in.stage_rec_bytes/MiB, 100.0*ss);
                out.why += buf;
            }
        }
        return out;
    }
    if (s < need) {
        // auto does not lower keep by itself and does not run a thrashing slot pool: non-resident path, same output
        snprintf(buf, sizeof(buf), "%.1f%% of records would fit on the card (%.0f MiB free - %.0f reserve - %.0f other KV = %.0f MiB room vs %.0f MiB body), "
            "below the %.1f%% one step reads at keep %.1f%%%s: falling back to the non-resident path (bodies on the card: %.0f MiB, %.0f MiB free after other KV + compute)",
            100.0*s, in.free_bytes/MiB, in.reserve_bytes/MiB, in.other_bytes/MiB, room/MiB, in.body_bytes/MiB,
            100.0*need, 100.0*(out.keep_cap > 0.0 ? out.keep_cap : in.keep_now),
            out.keep_cap > 0.0 ? " even after lowering keep (LLAMA_LSA_RESIDENT_LOWER_KEEP floor)" :
            in.keep_fit ? "" : "; LLAMA_LSA_RESIDENT_LOWER_KEEP=1 lowers keep instead (lossy)", in.body_bytes/MiB, fb_room/MiB);
        out.keep_cap = 0.0;
        return off(buf);
    }
    const char * cls = none            ? "NO ROOM for slots: every record is read from host RAM (very slow); lower the context" :
                       s < target        ? "below the 80% target: slower decode (rs3 3/2t: 65% ~ +2.3 ms/token)" :
                                           "meets the 80% target (rs3 3/2t: 80% ~ +0.5 ms/token)";
    snprintf(buf, sizeof(buf), "%.1f%% of records on the card: %.0f MiB free - %.0f reserve - %.0f other KV = %.0f MiB room vs %.0f MiB body; %s",
        100.0*s, in.free_bytes/MiB, in.reserve_bytes/MiB, in.other_bytes/MiB, room/MiB, in.body_bytes/MiB, cls);
    out.why = std::string(buf) + fitbuf;
    return out;
}

// Compute reserve for the sizing above. The defaults (reserve 1536 MiB = compute + 512 margin, compute floor 1024 MiB)
// are guesses; a memory-check pass (no_alloc context, e.g. the --fit sizing run before the real context) measures the
// device's compute buffer. With a measurement (> 0) and no explicit env value, compute = measured + max(64 MiB, 5%),
// reserve = that + the default margin (reserve - compute). Explicit LLAMA_LSA_RESIDENT_RESERVE_MIB / _COMPUTE_MIB win.
struct llama_lsa_resident_reserve_out {
    int64_t reserve_bytes = 0;
    int64_t compute_bytes = 0;
    bool    measured      = false; // at least one of the two came from the measurement
    bool    margin_set    = false; // margin_bytes came from margin_override (env or --fit-target)
    int64_t margin_bytes  = 0;     // reserve - compute
};

// margin_override >= 0 replaces the default margin (def_reserve - def_compute) in the reserve, with or without a
// measurement; an explicit reserve (reserve_set) still wins over it.
inline llama_lsa_resident_reserve_out llama_lsa_resident_reserve(int64_t def_reserve, int64_t def_compute,
        bool reserve_set, bool compute_set, int64_t measured, int64_t margin_override = -1) {
    llama_lsa_resident_reserve_out out;
    out.reserve_bytes = def_reserve;
    out.compute_bytes = std::min(def_compute, def_reserve);
    const int64_t margin = margin_override >= 0 ? margin_override : std::max<int64_t>(0, def_reserve - def_compute);
    out.margin_set = margin_override >= 0 && !reserve_set;
    if (measured <= 0) {
        if (out.margin_set) {
            out.reserve_bytes = out.compute_bytes + margin;
        }
        out.margin_bytes = out.reserve_bytes - out.compute_bytes;
        return out;
    }
    const int64_t c      = measured + std::max<int64_t>(64ll << 20, measured/20);
    if (!compute_set) {
        out.compute_bytes = c;
        out.measured      = true;
    }
    if (!reserve_set) {
        out.reserve_bytes = std::max(out.compute_bytes, c + margin);
        out.measured      = true;
    }
    out.compute_bytes = std::min(out.compute_bytes, out.reserve_bytes);
    out.margin_bytes  = out.reserve_bytes - out.compute_bytes;
    return out;
}
