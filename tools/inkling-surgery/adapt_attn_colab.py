#!/usr/bin/env python3
"""Clean-room frozen-expert attention surgery for Inkling 2 Mini (Pattern A, Option 1B).

Modifies ONLY the 8 physical attention blocks on slots r=2 (SWA-4K with extended
(16, 4096) relative bias) and r=4 (LSA with dynamic keep clamp(max(2048, 4%), 2048, 16384)
and learnable far-distance bias), keeping all 128 NVFP4 MoE experts, embeddings,
output head, loop controllers, and the 16 untouched attention blocks strictly frozen.

Supports:
  1. --verify-logits: exact stock 5:1 forward check against llama-server / GGUF.
  2. Online self-teacher distillation + long-range retrieval CE + DSpark tap MSE.
  3. Zero-secret public HF download (thinkingmachines/Inkling-2-Mini) or local --model-dir.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open


MODE_GLOBAL = 0
MODE_SWA = 1
MODE_SWA_MID = 2
MODE_LSA = 3

E2M1_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (xf * inv).to(x.dtype)
    if weight is not None:
        y = y * weight.to(x.dtype)
    return y


def l2_clip(x: torch.Tensor, scale: Optional[torch.Tensor], threshold: float) -> torch.Tensor:
    if scale is not None:
        x = x * scale.to(x.dtype)
    norm = x.float().norm(dim=-1, keepdim=True) + 1e-6
    factor = torch.clamp(threshold / norm, max=1.0).to(x.dtype)
    return x * factor


def causal_static_sconv(x: torch.Tensor, weight_c1k: torch.Tensor) -> torch.Tensor:
    """x: [T, C], weight_c1k: [C, 1, K] -> [T, C] with residual and 1/sqrt(K+1)."""
    k = weight_c1k.shape[-1]
    c = x.shape[-1]
    xt = x.transpose(0, 1).unsqueeze(0)  # [1, C, T]
    x_pad = F.pad(xt, (k - 1, 0))
    conv_out = F.conv1d(x_pad, weight_c1k.to(x.dtype), groups=c)[0].transpose(0, 1)
    return (x + conv_out) * (1.0 / math.sqrt(k + 1))


def causal_dynamic_sconv(
    x: torch.Tensor,
    static_w_kc: torch.Tensor,
    dyn_proj_w: torch.Tensor,
    inp_for_dyn: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    """x: [T, C], static_w_kc: [K, C] (lag 0..K-1), dyn_proj_w: [H*K, D_in], inp_for_dyn: [T, D_in]."""
    t_len, c = x.shape
    k = static_w_kc.shape[0]
    head_dim = c // n_heads
    # static_w[lag, ch] has lag 0 = current t, lag K-1 = t-(K-1); flip for conv1d causal order
    w_causal = static_w_kc.flip(0).transpose(0, 1).contiguous().unsqueeze(1)  # [C, 1, K]
    xt = x.transpose(0, 1).unsqueeze(0)
    x_pad = F.pad(xt, (k - 1, 0))
    out = F.conv1d(x_pad, w_causal.to(x.dtype), groups=c)[0].transpose(0, 1)

    dyn = F.linear(inp_for_dyn, dyn_proj_w.to(inp_for_dyn.dtype)).view(t_len, n_heads, k)
    x_h = x.view(t_len, n_heads, head_dim)
    for lag in range(k):
        tap = dyn[:, :, lag : lag + 1]  # [T, H, 1]
        if lag == 0:
            x_lag = x_h
        elif lag >= t_len:
            x_lag = torch.zeros_like(x_h)
        else:
            x_lag = F.pad(x_h[:-lag], (0, 0, 0, 0, lag, 0))
        out = out + (x_lag * tap).reshape(t_len, c)
    return (x + out) * (1.0 / math.sqrt(2 * k + 1))


def depth_mix(blocks: List[torch.Tensor], cur: torch.Tensor, query: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Attention-over-depth residual mixer. Returns [T, n_q, H]."""
    states = [rms_norm(b, None, eps).float() for b in blocks] + [rms_norm(cur, None, eps).float()]
    stacked = torch.stack(states, dim=1)  # [T, M, H]
    q = query.float()
    if q.ndim == 1:
        q = q.unsqueeze(0)  # [1, H]
    scale = 1.0 / math.sqrt(cur.shape[-1])
    scores = torch.einsum("tmh,qh->tqm", stacked, q) * scale
    w = F.softmax(scores, dim=-1)
    out = torch.einsum("tqm,tmh->tqh", w, stacked)
    return out.to(cur.dtype)


class FrozenNVFP4Experts(nn.Module):
    """Stores 128 NVFP4 or MXFP8 experts compactly and dequantizes chunks on the fly."""

    def __init__(
        self,
        w13_w: torch.Tensor,
        w13_scale: torch.Tensor,
        w13_outer: Optional[torch.Tensor],
        w2_w: torch.Tensor,
        w2_scale: torch.Tensor,
        w2_outer: Optional[torch.Tensor],
        situ_beta: float = 4.0,
        situ_linear_beta: float = 25.0,
        chunk_experts: int = 16,
    ):
        super().__init__()
        self.register_buffer("w13_w", w13_w, persistent=False)
        self.register_buffer("w13_scale", w13_scale, persistent=False)
        self.register_buffer("w13_outer", w13_outer, persistent=False)
        self.register_buffer("w2_w", w2_w, persistent=False)
        self.register_buffer("w2_scale", w2_scale, persistent=False)
        self.register_buffer("w2_outer", w2_outer, persistent=False)
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.chunk_experts = chunk_experts

    @staticmethod
    @torch.no_grad()
    def _dequant_chunk(
        packed: torch.Tensor,
        scale: torch.Tensor,
        outer_f32: Optional[torch.Tensor],
        lut: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if outer_f32 is not None:
            # NVFP4: packed [E, R, C/2] uint8, scale [E, R, C/16] float8_e4m3fn, outer_f32 [E]
            e, r, c2 = packed.shape
            lo = lut[(packed & 0x0F).to(torch.int32)]
            hi = lut[(packed >> 4).to(torch.int32)]
            vals = torch.stack((lo, hi), dim=-1).reshape(e, r, c2 * 2)
            s = scale.to(torch.float32) * outer_f32.view(e, 1, 1)
            vals = vals.view(e, r, -1, 16) * s.unsqueeze(-1)
            return vals.reshape(e, r, c2 * 2).to(dtype)
        # MXFP8: packed [E, R, C] float8_e4m3fn, scale [E, R, C/32] uint8 (UE8M0: 2^(s - 127))
        e, r, c = packed.shape
        vals = packed.to(torch.float32)
        s = torch.ldexp(torch.ones(scale.shape, dtype=torch.float32, device=scale.device), scale.to(torch.int32) - 127)
        vals = vals.view(e, r, -1, 32) * s.unsqueeze(-1)
        return vals.reshape(e, r, c).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        top_idx: torch.Tensor,
        top_w: torch.Tensor,
        lut: torch.Tensor,
    ) -> torch.Tensor:
        t_len, hidden = x.shape
        k_top = top_idx.shape[-1]
        n_exp = self.w13_w.shape[0]

        flat_exp = top_idx.reshape(-1)
        flat_tok = torch.arange(t_len, device=x.device).unsqueeze(1).expand(t_len, k_top).reshape(-1)
        flat_w = top_w.reshape(-1).to(x.dtype)

        order = torch.argsort(flat_exp)
        sorted_exp = flat_exp[order]
        sorted_tok = flat_tok[order]
        sorted_w = flat_w[order]

        counts = torch.bincount(sorted_exp, minlength=n_exp)
        counts_cpu = counts.tolist()
        offsets = [0] * (n_exp + 1)
        acc = 0
        for i, cnt in enumerate(counts_cpu):
            acc += cnt
            offsets[i + 1] = acc

        out = torch.zeros_like(x)
        beta = self.situ_beta
        lbeta = self.situ_linear_beta

        for e0 in range(0, n_exp, self.chunk_experts):
            e1 = min(e0 + self.chunk_experts, n_exp)
            active = [e for e in range(e0, e1) if counts_cpu[e] > 0]
            if not active:
                continue
            idx_t = torch.tensor(active, device=x.device, dtype=torch.long)
            w13_c = self._dequant_chunk(
                self.w13_w[idx_t],
                self.w13_scale[idx_t],
                self.w13_outer[idx_t] if self.w13_outer is not None else None,
                lut,
                x.dtype,
            )
            w2_c = self._dequant_chunk(
                self.w2_w[idx_t],
                self.w2_scale[idx_t],
                self.w2_outer[idx_t] if self.w2_outer is not None else None,
                lut,
                x.dtype,
            )
            for local_i, e in enumerate(active):
                s0, s1 = offsets[e], offsets[e + 1]
                tok_ids = sorted_tok[s0:s1]
                xe = x.index_select(0, tok_ids)
                gu = F.linear(xe, w13_c[local_i])
                g = gu[:, 0::2]
                u = gu[:, 1::2]
                act = (beta * torch.tanh(g / beta) * torch.sigmoid(g)) * (lbeta * torch.tanh(u / lbeta))
                ye = F.linear(act, w2_c[local_i]) * sorted_w[s0:s1].unsqueeze(-1)
                out.index_add_(0, tok_ids, ye)
        return out


class InklingAttentionBlock(nn.Module):
    """Single physical Inkling 2 Mini attention block (trainable on modified layers)."""

    def __init__(self, layer_idx: int, shared_norm: bool, d_model: int = 2304, n_head: int = 24, n_head_kv: int = 8, head_dim: int = 128, d_rel: int = 16, rel_extent: int = 1024, sconv_k: int = 4):
        super().__init__()
        self.layer_idx = layer_idx
        self.shared_norm = shared_norm
        self.d_model = d_model
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.d_rel = d_rel
        self.rel_extent = rel_extent

        if shared_norm:
            self.attn_norm = nn.Parameter(torch.empty(d_model, dtype=torch.bfloat16))
        else:
            self.q_input_norm = nn.Parameter(torch.empty(d_model, dtype=torch.bfloat16))
            self.k_input_norm = nn.Parameter(torch.empty(d_model, dtype=torch.bfloat16))
            self.v_input_norm = nn.Parameter(torch.empty(d_model, dtype=torch.bfloat16))

        self.wq = nn.Parameter(torch.empty(n_head * head_dim, d_model, dtype=torch.bfloat16))
        self.wk = nn.Parameter(torch.empty(n_head_kv * head_dim, d_model, dtype=torch.bfloat16))
        self.wv = nn.Parameter(torch.empty(n_head_kv * head_dim, d_model, dtype=torch.bfloat16))
        self.wo = nn.Parameter(torch.empty(d_model, n_head * head_dim, dtype=torch.bfloat16))
        self.w_gate = nn.Parameter(torch.empty(n_head * head_dim, d_model, dtype=torch.bfloat16))
        self.w_rel = nn.Parameter(torch.empty(n_head * d_rel, d_model, dtype=torch.bfloat16))
        self.w_sink = nn.Parameter(torch.empty(n_head, d_model, dtype=torch.bfloat16))

        self.q_clip = nn.Parameter(torch.empty(n_head, head_dim, dtype=torch.bfloat16))
        self.rel_clip = nn.Parameter(torch.empty(n_head, d_rel, dtype=torch.bfloat16))
        self.rel_bias = nn.Parameter(torch.empty(d_rel, rel_extent, dtype=torch.bfloat16))

        if shared_norm:
            self.k_sconv = nn.Parameter(torch.empty(n_head_kv * head_dim, 1, sconv_k, dtype=torch.bfloat16))
            self.v_sconv = nn.Parameter(torch.empty(n_head_kv * head_dim, 1, sconv_k, dtype=torch.bfloat16))
        else:
            self.k_static_w = nn.Parameter(torch.empty(sconv_k, n_head_kv * head_dim, dtype=torch.bfloat16))
            self.k_weight_proj = nn.Parameter(torch.empty(n_head_kv * sconv_k, d_model, dtype=torch.bfloat16))
            self.v_static_w = nn.Parameter(torch.empty(sconv_k, n_head_kv * head_dim, dtype=torch.bfloat16))
            self.v_weight_proj = nn.Parameter(torch.empty(n_head_kv * sconv_k, d_model, dtype=torch.bfloat16))

        self.attn_sconv = nn.Parameter(torch.empty(d_model, 1, sconv_k, dtype=torch.bfloat16))
        # Scalar far-distance bias applied to d >= rel_extent on LSA layers (0.0 for stock/SWA_MID)
        self.lsa_far_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)

    def extend_rel_bias(self, new_extent: int, init_mode: str = "tail", init_scale: float = 1.0, far_offset: float = 0.0) -> None:
        old = self.rel_bias.detach().float()  # [16, old_E]
        old_e = old.shape[1]
        if new_extent == old_e:
            return
        new_b = torch.zeros((old.shape[0], new_extent), dtype=torch.float32, device=old.device)
        new_b[:, :old_e] = old
        if init_mode == "tail":
            # Smooth continuation from edge-of-window d=1023 vector
            tail = old[:, old_e - 1 : old_e] * init_scale
            new_b[:, old_e:] = tail
        elif init_mode == "zero":
            new_b[:, old_e:] = 0.0
        else:
            raise ValueError(f"Unknown init_mode={init_mode}")
        self.rel_bias = nn.Parameter(new_b.to(self.rel_bias.dtype))
        self.rel_extent = new_extent
        if far_offset != 0.0:
            self.lsa_far_bias = nn.Parameter(torch.tensor([far_offset], dtype=torch.float32, device=old.device))

    def snapshot_cpu(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().to("cpu", copy=True) for k, v in self.named_parameters()}

    def forward_attn(
        self,
        inputs: List[torch.Tensor],
        mode: int,
        tau: torch.Tensor,
        q_blk: int = 256,
        swa_window: int = 1024,
        swa_mid_window: int = 4096,
        lsa_keep_min: int = 2048,
        lsa_keep_frac: float = 0.04,
        lsa_keep_max: int = 16384,
        lsa_block: int = 64,
        lsa_sinks: int = 128,
        qk_clip_thr: float = 22.627416997969522,
        rel_clip_thr: float = 8.0,
        has_messages: bool = False,
        stats_out: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        if self.shared_norm and has_messages:
            x0 = rms_norm(inputs[0], self.attn_norm)
            x1 = rms_norm(inputs[1], self.attn_norm)
            x2 = rms_norm(inputs[2], self.attn_norm)
        elif self.shared_norm:
            x0 = rms_norm(inputs[0], self.attn_norm)
            x1 = x2 = x0
        else:
            x0 = rms_norm(inputs[0], self.q_input_norm)
            x1 = rms_norm(inputs[1], self.k_input_norm)
            x2 = rms_norm(inputs[2], self.v_input_norm)

        t_len = x0.shape[0]
        dtype = x0.dtype

        q = F.linear(x0, self.wq.to(dtype))
        r = F.linear(x0, self.w_rel.to(dtype))
        sinks = F.linear(x0, self.w_sink.to(dtype))  # [T, H]
        gate = F.linear(x0, self.w_gate.to(dtype))
        k = F.linear(x1, self.wk.to(dtype))
        v = F.linear(x2, self.wv.to(dtype))

        if self.shared_norm:
            k = causal_static_sconv(k, self.k_sconv)
            v = causal_static_sconv(v, self.v_sconv)
        else:
            k = causal_dynamic_sconv(k, self.k_static_w, self.k_weight_proj, x1, self.n_head_kv)
            v = causal_dynamic_sconv(v, self.v_static_w, self.v_weight_proj, x2, self.n_head_kv)

        q = l2_clip(q.view(t_len, self.n_head, self.head_dim), self.q_clip, qk_clip_thr)
        k = l2_clip(k.view(t_len, self.n_head_kv, self.head_dim), None, qk_clip_thr)
        v = rms_norm(v.view(t_len, self.n_head_kv, self.head_dim), None, 1e-6)
        r = l2_clip(r.view(t_len, self.n_head, self.d_rel), self.rel_clip, rel_clip_thr)

        is_swa = mode in (MODE_SWA, MODE_SWA_MID)
        if not is_swa:
            q = q * tau.to(q.dtype)
            r = r * tau.to(r.dtype)

        # Build [T, H, E+1] relative bias table in FP32 for numerical stability
        b_pad = F.pad(self.rel_bias.float(), (0, 1), value=0.0)  # [16, E+1]
        # table[t, h, d] = dot(r[t, h], b_pad[:, d]) - sinks[t, h]
        rel_table = torch.einsum("thd,de->the", r.float(), b_pad) - sinks.float().unsqueeze(-1)
        if mode == MODE_LSA or (mode == MODE_SWA_MID and self.lsa_far_bias.item() != 0.0):
            # Apply scalar far-bias on the d >= E entry (matching inkling2.cpp:602)
            fb = self.lsa_far_bias.float()
            rel_table = torch.cat([rel_table[:, :, :-1], rel_table[:, :, -1:] + fb], dim=-1)

        if stats_out is not None and "_diag_rel1023" in stats_out:
            # Record mean dot(R, B[:, 1023]) - sink for calibration diagnostics
            stats_out["_diag_rel1023"] = float(rel_table[:, :, min(1023, rel_table.shape[-1] - 2)].mean().item())

        # Prepare K/V in [H, T, D] (repeat_interleave for GQA)
        rep = self.n_head // self.n_head_kv
        k_h = k.permute(1, 0, 2).repeat_interleave(rep, dim=0)  # [H, T, D]
        v_h = v.permute(1, 0, 2).repeat_interleave(rep, dim=0)  # [H, T, D]
        q_h = q.permute(1, 0, 2)  # [H, T, D]

        # Precompute LSA block keys if needed
        k_blk_h = None
        nk_far = 0
        if mode == MODE_LSA:
            n_keep = max(lsa_keep_min, int(lsa_keep_frac * t_len))
            n_keep = min(max(n_keep, lsa_keep_min), lsa_keep_max)
            nk_far = max(0, (n_keep - lsa_sinks - swa_window) // lsa_block)
            n_blk = t_len // lsa_block
            if n_blk > 0:
                k_trim = k[: n_blk * lsa_block].view(n_blk, lsa_block, self.n_head_kv, self.head_dim)
                k_blk_kv = k_trim.float().mean(dim=1)  # [n_blk, H_kv, D]
                k_blk_h = k_blk_kv.permute(1, 0, 2).repeat_interleave(rep, dim=0)  # [H, n_blk, D]

        scale = 1.0 / math.sqrt(self.head_dim)
        extent = self.rel_extent
        out_chunks = []
        far_mass_sum = 0.0
        mid_mass_sum = 0.0
        far4k_mass_sum = 0.0
        far_q_count = 0

        for q0 in range(0, t_len, q_blk):
            q1 = min(q0 + q_blk, t_len)
            qt = q_h[:, q0:q1, :]  # [H, Bq, D]
            rt = rel_table[q0:q1].permute(1, 0, 2)  # [H, Bq, E+1]
            q_pos = torch.arange(q0, q1, device=x0.device, dtype=torch.long)

            if mode == MODE_SWA:
                k0 = max(0, q0 - (swa_window - 1))
                k_pos = torch.arange(k0, q1, device=x0.device, dtype=torch.long)
                kt = k_h[:, k0:q1, :]
                vt = v_h[:, k0:q1, :]
                dist = q_pos[:, None] - k_pos[None, :]  # [Bq, Bk]
                valid = (dist >= 0) & (dist < swa_window)
            elif mode == MODE_SWA_MID:
                k0 = max(0, q0 - (swa_mid_window - 1))
                k_pos = torch.arange(k0, q1, device=x0.device, dtype=torch.long)
                kt = k_h[:, k0:q1, :]
                vt = v_h[:, k0:q1, :]
                dist = q_pos[:, None] - k_pos[None, :]
                valid = (dist >= 0) & (dist < swa_mid_window)
            elif mode == MODE_GLOBAL:
                k_pos = torch.arange(0, q1, device=x0.device, dtype=torch.long)
                kt = k_h[:, :q1, :]
                vt = v_h[:, :q1, :]
                dist = q_pos[:, None] - k_pos[None, :]
                valid = dist >= 0
            elif mode == MODE_LSA:
                # Exact INKLSA-SJ block voting: sinks [0..sinks) U topk far blocks U near [q0-W+1..q1)
                near_start = max(0, q0 - (swa_window - 1))
                first_elig_blk = lsa_sinks // lsa_block
                last_elig_blk = near_start // lsa_block
                n_elig = max(0, last_elig_blk - first_elig_blk)
                if nk_far <= 0 or n_elig <= nk_far or k_blk_h is None:
                    k_pos = torch.arange(0, q1, device=x0.device, dtype=torch.long)
                    kt = k_h[:, :q1, :]
                    vt = v_h[:, :q1, :]
                else:
                    with torch.no_grad():
                        kb_elig = k_blk_h[:, first_elig_blk:last_elig_blk, :]  # [H, n_elig, D]
                        sc_blk = torch.bmm(qt.float(), kb_elig.transpose(1, 2)) * scale  # [H, Bq, n_elig]
                        votes = F.softmax(sc_blk, dim=-1).sum(dim=(0, 1))  # [n_elig]
                        top_b = torch.topk(votes, k=nk_far, largest=True, sorted=False).indices + first_elig_blk
                        top_b = torch.sort(top_b).values
                        far_pos = (
                            top_b.unsqueeze(1) * lsa_block
                            + torch.arange(lsa_block, device=x0.device, dtype=torch.long).unsqueeze(0)
                        ).reshape(-1)
                        sink_pos = torch.arange(0, min(lsa_sinks, near_start), device=x0.device, dtype=torch.long)
                        near_pos = torch.arange(last_elig_blk * lsa_block, q1, device=x0.device, dtype=torch.long)
                        k_pos = torch.cat([sink_pos, far_pos, near_pos], dim=0)
                    kt = k_h.index_select(1, k_pos)
                    vt = v_h.index_select(1, k_pos)
                dist = q_pos[:, None] - k_pos[None, :]
                valid = dist >= 0
            else:
                raise ValueError(f"Unknown mode {mode}")

            scores = torch.bmm(qt.float(), kt.float().transpose(1, 2)) * scale  # [H, Bq, Bk]
            d_idx = torch.clamp(dist, min=0, max=extent)  # [Bq, Bk]
            # Gather relative bias: rt is [H, Bq, E+1], d_idx is [Bq, Bk]
            d_exp = d_idx.unsqueeze(0).expand(self.n_head, -1, -1)
            bias = torch.gather(rt, 2, d_exp)
            scores = scores + bias
            scores = scores.masked_fill(~valid.unsqueeze(0), -1e9)

            # Softmax with zero-logit sink: denom = exp(0) + sum_k exp(scores_k)
            m = torch.maximum(scores.max(dim=-1, keepdim=True).values, torch.zeros_like(scores[..., :1]))
            exp_s = torch.exp(scores - m) * valid.unsqueeze(0).to(scores.dtype)
            denom = exp_s.sum(dim=-1, keepdim=True) + torch.exp(-m)
            probs = exp_s / denom  # [H, Bq, Bk]

            if stats_out is not None and mode in (MODE_SWA_MID, MODE_LSA):
                with torch.no_grad():
                    eligible_q = q_pos >= 1024
                    n_eq = int(eligible_q.sum().item())
                    if n_eq > 0:
                        far_mask = valid & (dist >= 1024)
                        mid_mask = valid & (dist >= 1024) & (dist < 4096)
                        f4k_mask = valid & (dist >= 4096)
                        # Sum over eligible query rows & keys, average over heads
                        far_mass_sum += float((probs[:, eligible_q, :] * far_mask[eligible_q].unsqueeze(0)).sum().item()) / self.n_head
                        mid_mass_sum += float((probs[:, eligible_q, :] * mid_mask[eligible_q].unsqueeze(0)).sum().item()) / self.n_head
                        far4k_mass_sum += float((probs[:, eligible_q, :] * f4k_mask[eligible_q].unsqueeze(0)).sum().item()) / self.n_head
                        far_q_count += n_eq

            out_q = torch.bmm(probs.to(vt.dtype), vt)  # [H, Bq, D]
            out_chunks.append(out_q.permute(1, 0, 2).reshape(q1 - q0, self.n_head * self.head_dim))

        if stats_out is not None and far_q_count > 0 and mode in (MODE_SWA_MID, MODE_LSA):
            key_prefix = "swa_mid" if mode == MODE_SWA_MID else "lsa"
            stats_out[f"{key_prefix}_far_sum"] = stats_out.get(f"{key_prefix}_far_sum", 0.0) + far_mass_sum
            stats_out[f"{key_prefix}_mid_sum"] = stats_out.get(f"{key_prefix}_mid_sum", 0.0) + mid_mass_sum
            stats_out[f"{key_prefix}_f4k_sum"] = stats_out.get(f"{key_prefix}_f4k_sum", 0.0) + far4k_mass_sum
            stats_out[f"{key_prefix}_q_cnt"] = stats_out.get(f"{key_prefix}_q_cnt", 0.0) + far_q_count

        fa_out = torch.cat(out_chunks, dim=0)  # [T, H*D]
        gated = fa_out * torch.sigmoid(gate.float()).to(fa_out.dtype)
        proj = F.linear(gated, self.wo.to(gated.dtype))
        return causal_static_sconv(proj, self.attn_sconv)


class InklingLayer(nn.Module):
    def __init__(self, ip: int, shared_norm: bool):
        super().__init__()
        self.ip = ip
        self.attn = InklingAttentionBlock(ip, shared_norm=shared_norm)
        # Non-attention weights stored as frozen buffers
        self.experts: Optional[FrozenNVFP4Experts] = None

    def set_frozen(self, name: str, tensor: torch.Tensor) -> None:
        self.register_buffer(name, tensor, persistent=False)

    def forward_visit(
        self,
        cur: torch.Tensor,
        blocks: List[torch.Tensor],
        mode: int,
        tau: torch.Tensor,
        lut: torch.Tensor,
        messages: Optional[torch.Tensor],
        q_blk: int,
        stats_out: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        inp_sa = cur
        if self.ip > 0:
            mixed = depth_mix(blocks, cur, self.attn_res_query)
            if self.attn.shared_norm:
                inputs = [mixed[:, 0], mixed[:, 0], mixed[:, 0]]
            else:
                inputs = [mixed[:, 0], mixed[:, 1], mixed[:, 2]]
        else:
            inputs = [cur, cur, cur]

        if messages is not None:
            inputs = [
                self._control(inputs[0], messages[:, 0], self.ctrl_q_down, self.ctrl_q_gain, self.ctrl_q_forget),
                self._control(inputs[1], messages[:, 1], self.ctrl_k_down, self.ctrl_k_gain, self.ctrl_k_forget),
                self._control(inputs[2], messages[:, 2], self.ctrl_v_down, self.ctrl_v_gain, self.ctrl_v_forget),
            ]

        attn_out = self.attn.forward_attn(
            inputs=inputs,
            mode=mode,
            tau=tau,
            q_blk=q_blk,
            has_messages=(messages is not None),
            stats_out=stats_out,
        )

        pushed_block = None
        if self.ip % 6 == 0:
            pushed_block = inp_sa
            cur = attn_out
            cur_blocks = blocks + [inp_sa]
        else:
            cur = inp_sa + attn_out
            cur_blocks = blocks

        ffn_inp = cur
        h = depth_mix(cur_blocks, cur, self.ffn_res_query)[:, 0]
        if messages is not None:
            h = self._control(h, messages[:, 3], self.ctrl_m_down, self.ctrl_m_gain, self.ctrl_m_forget)
        h = rms_norm(h, self.mlp_norm)

        probs = torch.sigmoid(F.linear(h.float(), self.gate_weight.float()))
        sel_scores = probs + self.gate_bias.float()
        top_idx = torch.topk(sel_scores, k=8, dim=-1).indices
        top_w = probs.gather(-1, top_idx)
        top_w = (top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-20)) * 8.0

        ffn_out = self.experts(h, top_idx, top_w, lut)
        ffn_out = causal_static_sconv(ffn_out, self.mlp_sconv)
        cur = ffn_inp + ffn_out
        return cur, pushed_block

    @staticmethod
    def _control(cur: torch.Tensor, msg: torch.Tensor, down_w: torch.Tensor, gain_w: torch.Tensor, forget_w: torch.Tensor) -> torch.Tensor:
        g_in = torch.cat([rms_norm(cur, None), rms_norm(msg, None)], dim=-1)
        gate = F.linear(g_in, down_w.to(g_in.dtype))
        gain = torch.tanh(F.linear(gate, gain_w.to(gate.dtype)))
        forget = torch.tanh(F.linear(gate, forget_w.to(gate.dtype)))
        return cur - forget * cur + gain * msg


class Inkling2MiniSurgeryModel(nn.Module):
    def __init__(self, model_dir: Path, device: torch.device, chunk_experts: int = 16):
        super().__init__()
        with open(model_dir / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        tc = cfg["text_config"]
        self.hidden_size = tc["hidden_size"]
        self.n_phys = tc["num_hidden_layers"]
        self.unpadded_vocab = tc["unpadded_vocab_size"]
        self.softcap = float(tc["final_logit_softcapping"])
        lc = tc["loop_config"]
        self.core_start = lc["core_start"]
        self.core_end = lc["core_end"]
        self.n_passes = lc["num_passes"]
        local_set = set(tc["local_layer_ids"])

        self.register_buffer("e2m1_lut", torch.tensor(E2M1_TABLE, dtype=torch.float32, device=device), persistent=False)
        self.layers = nn.ModuleList([InklingLayer(ip, shared_norm=(ip in local_set)) for ip in range(self.n_phys)])

        st_path = model_dir / "model-00001-of-00001.safetensors"
        t0 = time.time()
        with safe_open(str(st_path), framework="pt", device="cpu") as sf:
            self.register_buffer("embed_w", sf.get_tensor("model.llm.embed.weight").to(device), persistent=False)
            self.register_buffer("embed_norm_w", sf.get_tensor("model.llm.embed_norm.weight").to(device), persistent=False)
            self.register_buffer("out_norm_w", sf.get_tensor("model.llm.norm.weight").to(device), persistent=False)
            self.register_buffer("out_res_q", sf.get_tensor("model.llm.attn_res.output_query").to(device), persistent=False)
            # Slice unpadded rows immediately to save memory
            self.register_buffer("unembed_w", sf.get_tensor("model.llm.unembed.weight")[: self.unpadded_vocab].contiguous().to(device), persistent=False)

            cp_q = sf.get_tensor("model.llm.middle_cycle_cross_pass_queries").to(device)
            cp_kv = sf.get_tensor("model.llm.middle_cycle_cross_pass_kv_queries").to(device)
            n_core = self.core_end - self.core_start
            msg_q = []
            for i in range(n_core):
                msg_q.append(torch.stack([cp_q[2 * i], cp_kv[2 * i], cp_kv[2 * i + 1], cp_q[2 * i + 1]], dim=0))
            self.register_buffer("msg_queries", torch.stack(msg_q, dim=0), persistent=False)  # [12, 4, H]

            for ip in range(self.n_phys):
                p = f"model.llm.layers.{ip}"
                layer = self.layers[ip]
                a = layer.attn
                if a.shared_norm:
                    a.attn_norm.data.copy_(sf.get_tensor(f"{p}.attn_norm.weight"))
                    a.k_sconv.data.copy_(sf.get_tensor(f"{p}.attn.k_sconv.weight"))
                    a.v_sconv.data.copy_(sf.get_tensor(f"{p}.attn.v_sconv.weight"))
                else:
                    a.q_input_norm.data.copy_(sf.get_tensor(f"{p}.q_input_norm.weight"))
                    a.k_input_norm.data.copy_(sf.get_tensor(f"{p}.k_input_norm.weight"))
                    a.v_input_norm.data.copy_(sf.get_tensor(f"{p}.v_input_norm.weight"))
                    a.k_static_w.data.copy_(sf.get_tensor(f"{p}.attn.k_sconv.static_w"))
                    a.k_weight_proj.data.copy_(sf.get_tensor(f"{p}.attn.k_sconv.weight_proj"))
                    a.v_static_w.data.copy_(sf.get_tensor(f"{p}.attn.v_sconv.static_w"))
                    a.v_weight_proj.data.copy_(sf.get_tensor(f"{p}.attn.v_sconv.weight_proj"))

                a.wq.data.copy_(sf.get_tensor(f"{p}.attn.wq_du.weight"))
                a.wk.data.copy_(sf.get_tensor(f"{p}.attn.wk_dv.weight"))
                a.wv.data.copy_(sf.get_tensor(f"{p}.attn.wv_dv.weight"))
                a.wo.data.copy_(sf.get_tensor(f"{p}.attn.wo_ud.weight"))
                a.w_gate.data.copy_(sf.get_tensor(f"{p}.attn.gate_attn.weight"))
                a.w_rel.data.copy_(sf.get_tensor(f"{p}.attn.wr_du.weight"))
                a.w_sink.data.copy_(sf.get_tensor(f"{p}.attn.attn_sink_proj.weight"))
                a.q_clip.data.copy_(sf.get_tensor(f"{p}.attn.q_l2_clip.scale"))
                a.rel_clip.data.copy_(sf.get_tensor(f"{p}.attn.rel_l2_clip.scale"))
                a.rel_bias.data.copy_(sf.get_tensor(f"{p}.attn.rel_logits_proj.proj"))
                a.attn_sconv.data.copy_(sf.get_tensor(f"{p}.attn_sconv.weight"))
                a.to(device)

                if ip > 0:
                    layer.set_frozen("attn_res_query", sf.get_tensor(f"model.llm.attn_res.attn_queries.{ip - 1}").to(device))
                layer.set_frozen("ffn_res_query", sf.get_tensor(f"model.llm.attn_res.mlp_queries.{ip}").to(device))
                layer.set_frozen("mlp_norm", sf.get_tensor(f"{p}.mlp_norm.weight").to(device))
                layer.set_frozen("mlp_sconv", sf.get_tensor(f"{p}.mlp_sconv.weight").to(device))
                layer.set_frozen("gate_weight", sf.get_tensor(f"{p}.mlp.gate.weight").to(device))
                layer.set_frozen("gate_bias", sf.get_tensor(f"{p}.mlp.gate.bias").to(device))

                if self.core_start <= ip < self.core_end:
                    for c_short, c_hf in (("q", "attn"), ("k", "k"), ("v", "v"), ("m", "mlp")):
                        layer.set_frozen(f"ctrl_{c_short}_down", sf.get_tensor(f"{p}.controllers.{c_hf}.down_weight").to(device))
                        layer.set_frozen(f"ctrl_{c_short}_gain", sf.get_tensor(f"{p}.controllers.{c_hf}.gain_weight").to(device))
                        layer.set_frozen(f"ctrl_{c_short}_forget", sf.get_tensor(f"{p}.controllers.{c_hf}.forget_weight").to(device))

                sf_keys = set(sf.keys())
                amax13_k = f"{p}.mlp.experts.w13_weight_amax"
                if amax13_k in sf_keys:
                    w13_amax = sf.get_tensor(amax13_k).float()
                    w13_den = sf.get_tensor(f"{p}.mlp.experts.w13_weight_outer_scale_denominator").float().item()
                    w2_amax = sf.get_tensor(f"{p}.mlp.experts.w2_weight_amax").float()
                    w2_den = sf.get_tensor(f"{p}.mlp.experts.w2_weight_outer_scale_denominator").float().item()
                    w13_outer = (w13_amax / w13_den).to(device)
                    w2_outer = (w2_amax / w2_den).to(device)
                else:
                    w13_outer = None
                    w2_outer = None
                layer.experts = FrozenNVFP4Experts(
                    w13_w=sf.get_tensor(f"{p}.mlp.experts.w13_weight").to(device),
                    w13_scale=sf.get_tensor(f"{p}.mlp.experts.w13_weight_scale").to(device),
                    w13_outer=w13_outer,
                    w2_w=sf.get_tensor(f"{p}.mlp.experts.w2_weight").to(device),
                    w2_scale=sf.get_tensor(f"{p}.mlp.experts.w2_weight_scale").to(device),
                    w2_outer=w2_outer,
                    chunk_experts=chunk_experts,
                )

        # Freeze everything by default
        for param in self.parameters():
            param.requires_grad_(False)
        vram_str = (
            f"{torch.cuda.memory_allocated(device) / (1024**3):.2f} GiB"
            if device.type == "cuda"
            else "cpu"
        )
        print(
            f"[load] Inkling 2 Mini loaded in {time.time() - t0:.1f}s | VRAM allocated: {vram_str}",
            flush=True,
        )

    def forward_sequence(
        self,
        input_ids: torch.Tensor,
        layer_modes: List[int],
        use_checkpoint: bool = False,
        q_blk: int = 256,
        tap_visits: Tuple[int, ...] = (5, 23, 41),
        stats_out: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Run full 48-visit forward pass. Returns (logits [T, V], {visit_idx: hidden [T, H]})."""
        t_len = input_ids.shape[0]
        device = input_ids.device
        pos = torch.arange(t_len, device=device, dtype=torch.float32)
        tau = (1.0 + 0.1 * torch.log(torch.clamp((pos + 1.0) / 8192.0, min=1.0))).view(t_len, 1, 1)

        cur = rms_norm(F.embedding(input_ids, self.embed_w), self.embed_norm_w)
        blocks: List[torch.Tensor] = []
        taps: Dict[int, torch.Tensor] = {}
        visit_idx = 0

        def run_visit(ip: int, c_in: torch.Tensor, blks: List[torch.Tensor], msgs: Optional[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            nonlocal visit_idx
            vid = visit_idx
            if vid in tap_visits:
                taps[vid] = c_in
            visit_idx += 1
            layer = self.layers[ip]
            mode = layer_modes[ip]

            if use_checkpoint:
                n_b = len(blks)
                has_m = msgs is not None

                def _fn(c_arg, *rest):
                    b_list = list(rest[:n_b])
                    m_arg = rest[n_b] if has_m else None
                    s_arg = None
                    if stats_out is not None:
                        seen = stats_out.setdefault("_seen", set())
                        if vid not in seen:
                            seen.add(vid)
                            s_arg = stats_out
                    c_out, pushed = layer.forward_visit(c_arg, b_list, mode, tau, self.e2m1_lut, m_arg, q_blk, s_arg)
                    return (c_out, pushed) if pushed is not None else (c_out,)

                args = [c_in] + list(blks) + ([msgs] if has_m else [])
                res = checkpoint(_fn, *args, use_reentrant=False)
                c_out = res[0]
                new_blks = list(blks) + ([res[1]] if len(res) > 1 else [])
                return c_out, new_blks
            else:
                c_out, pushed = layer.forward_visit(c_in, blks, mode, tau, self.e2m1_lut, msgs, q_blk, stats_out)
                new_blks = list(blks) + ([pushed] if pushed is not None else [])
                return c_out, new_blks

        # Prelude: layers 0..5
        for ip in range(self.core_start):
            cur, blocks = run_visit(ip, cur, blocks, None)

        prelude = cur
        prelude_blocks = list(blocks)

        # Looped core: [6..17] x 3 passes
        for pass_idx in range(self.n_passes):
            bank: List[torch.Tensor] = []
            bank_out: Optional[torch.Tensor] = None
            if pass_idx > 0:
                bank = blocks[len(prelude_blocks) + 1 :]
                bank_out = cur
                cur = prelude
                blocks = list(prelude_blocks)
            for ip in range(self.core_start, self.core_end):
                msgs = None
                if pass_idx > 0:
                    assert bank_out is not None
                    msgs = depth_mix(bank, bank_out, self.msg_queries[ip - self.core_start])
                cur, blocks = run_visit(ip, cur, blocks, msgs)

        # Epilogue: layers 18..23
        for ip in range(self.core_end, self.n_phys):
            cur, blocks = run_visit(ip, cur, blocks, None)

        cur = depth_mix(blocks, cur, self.out_res_q)[:, 0]
        cur = rms_norm(cur, self.out_norm_w)
        # Compute logits in FP32 for exact softcapping & KL stability
        logits = F.linear(cur.float(), self.unembed_w.float())
        if self.softcap != 0.0:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits, taps


def pattern_stock_modes(n_phys: int = 24) -> List[int]:
    return [MODE_GLOBAL if (i % 6 == 5) else MODE_SWA for i in range(n_phys)]


def pattern_a_modes(n_phys: int = 24, mid_slot: int = 2, lsa_slot: int = 4) -> List[int]:
    modes = []
    for i in range(n_phys):
        r = i % 6
        if r == 5:
            modes.append(MODE_GLOBAL)
        elif r == mid_slot:
            modes.append(MODE_SWA_MID)
        elif r == lsa_slot:
            modes.append(MODE_LSA)
        else:
            modes.append(MODE_SWA)
    return modes


def resolve_model_dir(model_dir: Optional[str], hf_repo: str) -> Path:
    if model_dir:
        p = Path(model_dir)
        if (p / "model-00001-of-00001.safetensors").is_file():
            return p
        raise FileNotFoundError(f"Missing model-00001-of-00001.safetensors in {p}")
    from huggingface_hub import hf_hub_download

    cache_dir = Path("/content/ink_hf")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("config.json", "model-00001-of-00001.safetensors"):
        print(f"[hf] downloading public {hf_repo}/{fname} ...", flush=True)
        hf_hub_download(repo_id=hf_repo, filename=fname, local_dir=str(cache_dir), token=False)
    return cache_dir


@torch.no_grad()
def run_verify_logits(model: Inkling2MiniSurgeryModel, token_ids: List[int], out_json: Optional[str]) -> None:
    device = next(model.buffers()).device
    ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    modes = pattern_stock_modes(model.n_phys)
    diag = {"_diag_rel1023": 0.0}
    logits, taps = model.forward_sequence(ids, modes, use_checkpoint=False, q_blk=256, stats_out=diag)
    last = logits[-1].float()
    probs = F.softmax(last, dim=-1)
    topv, topi = torch.topk(last, k=10)
    topp = probs[topi]
    res = {
        "n_tokens": len(token_ids),
        "token_ids": token_ids,
        "top10_ids": topi.tolist(),
        "top10_logits": [round(x, 5) for x in topv.tolist()],
        "top10_probs": [round(x, 6) for x in topp.tolist()],
        "tap_norms": {str(k): round(float(v[-1].float().norm().item()), 4) for k, v in taps.items()},
        "diag_rel1023_mean": round(diag["_diag_rel1023"], 4),
    }
    print(json.dumps(res, indent=2), flush=True)
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=2) + "\n")


def train_surgery(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model_dir = resolve_model_dir(args.model_dir, args.hf_repo)
    model = Inkling2MiniSurgeryModel(model_dir, device=device, chunk_experts=args.chunk_experts)

    if args.verify_logits:
        toks = [int(x) for x in args.verify_tokens.split(",") if x.strip()]
        run_verify_logits(model, toks, args.verify_out)
        return

    pack = torch.load(args.train_pack, map_location="cpu", weights_only=True)
    kl_seqs: List[torch.Tensor] = pack["kl_seqs"]
    ce_items: List[Dict[str, object]] = pack["ce_items"]
    print(f"[data] loaded {len(kl_seqs)} KL seqs and {len(ce_items)} CE long-retrieval items", flush=True)

    stock_modes = pattern_stock_modes(model.n_phys)
    pat_a_modes = pattern_a_modes(model.n_phys, mid_slot=args.mid_slot, lsa_slot=args.lsa_slot)
    mod_layers = [i for i in range(model.n_phys) if pat_a_modes[i] in (MODE_SWA_MID, MODE_LSA)]
    print(f"[surgery] modified physical layers: {mod_layers}", flush=True)

    # 1. Snapshot ONLY the original 8 attention blocks (~432 MB on CPU) for exact online self-teacher
    teacher_snaps: Dict[int, Dict[str, torch.Tensor]] = {}
    for il in mod_layers:
        teacher_snaps[il] = model.layers[il].attn.snapshot_cpu()
    print(f"[teacher] snapshotted {len(teacher_snaps)} original attention blocks to CPU RAM", flush=True)

    # 2. Extend r=2 SWA_MID relative bias to (16, 4096) and enable learnable lsa_far_bias on r=4 LSA
    for il in mod_layers:
        attn = model.layers[il].attn
        if pat_a_modes[il] == MODE_SWA_MID:
            attn.extend_rel_bias(4096, init_mode=args.mid_bias_init, init_scale=args.mid_bias_scale, far_offset=0.0)
        elif pat_a_modes[il] == MODE_LSA:
            attn.lsa_far_bias = nn.Parameter(
                torch.tensor([args.lsa_far_bias_init], dtype=torch.float32, device=device), requires_grad=True
            )

    # Unfreeze ONLY the 8 modified attention blocks (FP32 master params for stable BF16/FP32 updates)
    param_groups = []
    bias_params = []
    weight_params = []
    for il in mod_layers:
        attn = model.layers[il].attn
        for name, p in attn.named_parameters():
            if name == "lsa_far_bias" and pat_a_modes[il] != MODE_LSA:
                p.requires_grad_(False)
                continue
            p.data = p.data.float()
            p.requires_grad_(True)
            if name in ("rel_bias", "lsa_far_bias"):
                bias_params.append(p)
            else:
                weight_params.append(p)

    n_trainable = sum(p.numel() for p in weight_params) + sum(p.numel() for p in bias_params)
    print(
        f"[params] trainable={n_trainable/1e6:.2f}M (weights={sum(p.numel() for p in weight_params)/1e6:.2f}M, "
        f"rel_bias+far_bias={sum(p.numel() for p in bias_params)/1e6:.3f}M) | "
        f"VRAM={torch.cuda.memory_allocated(device)/(1024**3):.2f} GiB",
        flush=True,
    )

    param_groups = [
        {"params": weight_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": bias_params, "lr": args.bias_lr, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        prog = float(step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = []
    t_start = time.time()

    def swap_to_teacher() -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, nn.Parameter]]:
        student_cur: Dict[int, Dict[str, torch.Tensor]] = {}
        saved_rel_params: Dict[int, nn.Parameter] = {}
        for il in mod_layers:
            attn = model.layers[il].attn
            student_cur[il] = {k: v.detach().to("cpu", copy=True) for k, v in attn.named_parameters()}
            # Temporarily swap in teacher weights (1024 rel_bias, lsa_far_bias=0)
            for k, v in teacher_snaps[il].items():
                p = getattr(attn, k)
                if k == "rel_bias" and p.shape != v.shape:
                    saved_rel_params[il] = p
                    attn._parameters["rel_bias"] = nn.Parameter(v.to(device=device, dtype=torch.bfloat16), requires_grad=False)
                    attn.rel_extent = v.shape[1]
                else:
                    p.data.copy_(v.to(device=device, dtype=p.dtype))
        return student_cur, saved_rel_params

    def restore_student(student_cur: Dict[int, Dict[str, torch.Tensor]], saved_rel_params: Dict[int, nn.Parameter]) -> None:
        for il in mod_layers:
            attn = model.layers[il].attn
            if il in saved_rel_params:
                saved = saved_rel_params[il]
                attn._parameters["rel_bias"] = saved
                attn.rel_extent = saved.shape[1]
            for k, v in student_cur[il].items():
                p = getattr(attn, k)
                p.data.copy_(v.to(device=device, dtype=p.dtype))

    for step in range(1, args.steps + 1):
        t_step = time.time()
        optimizer.zero_grad(set_to_none=True)
        collect_stats = (step == 1) or (step % args.log_every == 0) or (step == args.steps)
        step_stats: Optional[Dict[str, float]] = {} if collect_stats else None

        # --- Sub-batch 1: KL + DSpark tap MSE on coding/general (4K..8K) ---
        kl_seq = kl_seqs[(step - 1) % len(kl_seqs)][: args.kl_len].to(device=device, dtype=torch.long)
        student_backup, saved_rel = swap_to_teacher()
        try:
            with torch.no_grad():
                t_logits, t_taps = model.forward_sequence(kl_seq, stock_modes, use_checkpoint=False, q_blk=args.q_blk)
                t_log_probs = F.log_softmax(t_logits, dim=-1)
                t_probs = t_log_probs.exp()
                t_taps_det = {k: v.detach().float() for k, v in t_taps.items()}
                del t_logits
        finally:
            restore_student(student_backup, saved_rel)
            del student_backup, saved_rel

        s_logits, s_taps = model.forward_sequence(
            kl_seq, pat_a_modes, use_checkpoint=True, q_blk=args.q_blk, stats_out=None
        )
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        # Forward KL(teacher || student) over positions >= 128
        kl_pos = slice(128, None) if kl_seq.shape[0] > 256 else slice(None)
        loss_kl = F.kl_div(s_log_probs[kl_pos], t_probs[kl_pos], reduction="batchmean", log_target=False)
        loss_hid = torch.zeros((), device=device, dtype=torch.float32)
        for tid in (5, 23, 41):
            # Relative MSE normalized by teacher variance so scale is O(1)
            diff = s_taps[tid][kl_pos].float() - t_taps_det[tid][kl_pos]
            denom = t_taps_det[tid][kl_pos].pow(2).mean().clamp_min(1e-4)
            loss_hid = loss_hid + (diff.pow(2).mean() / denom) / 3.0

        loss_1 = args.lambda_kl * loss_kl + args.lambda_hid * loss_hid
        loss_1.backward()
        kl_val = float(loss_kl.item())
        hid_val = float(loss_hid.item())
        del s_logits, s_log_probs, s_taps, t_log_probs, t_probs, t_taps_det, loss_kl, loss_hid, loss_1

        # --- Sub-batch 2: Long-context multi-needle / cross-file retrieval CE (8K..12K) ---
        ce_item = ce_items[(step - 1) % len(ce_items)]
        ce_ids = ce_item["ids"][: args.ce_len].to(device=device, dtype=torch.long)
        ce_mask = ce_item["loss_mask"][: args.ce_len].to(device=device, dtype=torch.bool)
        s_ce_logits, _ = model.forward_sequence(
            ce_ids, pat_a_modes, use_checkpoint=True, q_blk=args.q_blk, tap_visits=(), stats_out=step_stats
        )
        shift_logits = s_ce_logits[:-1]
        shift_labels = ce_ids[1:]
        shift_mask = ce_mask[1:]
        if int(shift_mask.sum().item()) > 0:
            loss_ce = F.cross_entropy(shift_logits[shift_mask], shift_labels[shift_mask])
        else:
            loss_ce = F.cross_entropy(shift_logits[-256:], shift_labels[-256:])
        loss_2 = args.lambda_ce * loss_ce
        loss_2.backward()
        ce_val = float(loss_ce.item())
        del s_ce_logits, shift_logits, loss_ce, loss_2

        grad_norm = float(torch.nn.utils.clip_grad_norm_(weight_params + bias_params, args.grad_clip).item())
        optimizer.step()
        scheduler.step()

        total_val = args.lambda_kl * kl_val + args.lambda_hid * hid_val + args.lambda_ce * ce_val
        if not math.isfinite(total_val):
            raise RuntimeError(f"Non-finite loss at step {step}: total={total_val} kl={kl_val} ce={ce_val} hid={hid_val}")

        if collect_stats:
            fb_vals = {
                str(il): round(float(model.layers[il].attn.lsa_far_bias.item()), 4)
                for il in mod_layers
                if pat_a_modes[il] == MODE_LSA
            }
            swa_mid_q = step_stats.get("swa_mid_q_cnt", 0.0) if step_stats else 0.0
            lsa_q = step_stats.get("lsa_q_cnt", 0.0) if step_stats else 0.0
            swa_mid_far = (step_stats.get("swa_mid_far_sum", 0.0) / swa_mid_q) if swa_mid_q > 0 else 0.0
            lsa_far = (step_stats.get("lsa_far_sum", 0.0) / lsa_q) if lsa_q > 0 else 0.0
            lsa_mid = (step_stats.get("lsa_mid_sum", 0.0) / lsa_q) if lsa_q > 0 else 0.0
            lsa_f4k = (step_stats.get("lsa_f4k_sum", 0.0) / lsa_q) if lsa_q > 0 else 0.0
            row = {
                "step": step,
                "loss": round(total_val, 5),
                "kl": round(kl_val, 5),
                "ce_long": round(ce_val, 5),
                "hid_mse": round(hid_val, 5),
                "grad_norm": round(grad_norm, 3),
                "lr": round(scheduler.get_last_lr()[0], 7),
                "bias_lr": round(scheduler.get_last_lr()[1], 7),
                "swa_mid_far_mass_gt1k": round(swa_mid_far, 5),
                "lsa_far_mass_gt1k": round(lsa_far, 5),
                "lsa_mass_1k_4k": round(lsa_mid, 5),
                "lsa_mass_ge4k": round(lsa_f4k, 5),
                "lsa_far_bias": fb_vals,
                "step_sec": round(time.time() - t_step, 2),
                "vram_peak_gib": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2),
            }
            metrics_log.append(row)
            print(
                f"step {step:4d}/{args.steps} | loss={row['loss']:.4f} kl={row['kl']:.4f} "
                f"ce={row['ce_long']:.4f} hid={row['hid_mse']:.4f} gnorm={row['grad_norm']:.2f} | "
                f"mass>1K: swa4k={row['swa_mid_far_mass_gt1k']:.4f} lsa={row['lsa_far_mass_gt1k']:.4f} "
                f"(1-4K={row['lsa_mass_1k_4k']:.4f}, >=4K={row['lsa_mass_ge4k']:.4f}) | "
                f"fb={fb_vals} | {row['step_sec']:.1f}s peak={row['vram_peak_gib']:.1f}G",
                flush=True,
            )
            (out_dir / "train_metrics.json").write_text(json.dumps(metrics_log, indent=2) + "\n")

        if args.save_every > 0 and step % args.save_every == 0 and step < args.steps:
            save_overlay(model, mod_layers, pat_a_modes, out_dir / f"adapted_attn_step{step}.pt", args)

    save_overlay(model, mod_layers, pat_a_modes, out_dir / "adapted_attn.pt", args)
    summary = {
        "steps": args.steps,
        "elapsed_sec": round(time.time() - t_start, 1),
        "mod_layers": mod_layers,
        "layer_modes": pat_a_modes,
        "final": metrics_log[-1] if metrics_log else {},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] saved overlay to {out_dir / 'adapted_attn.pt'} in {summary['elapsed_sec']}s", flush=True)


def save_overlay(
    model: Inkling2MiniSurgeryModel,
    mod_layers: List[int],
    layer_modes: List[int],
    path: Path,
    args: argparse.Namespace,
) -> None:
    layers_dict: Dict[int, Dict[str, torch.Tensor]] = {}
    lsa_far_bias = [0.0] * model.n_phys
    for il in mod_layers:
        attn = model.layers[il].attn
        d: Dict[str, torch.Tensor] = {}
        for k, p in attn.named_parameters():
            if k == "lsa_far_bias":
                if layer_modes[il] == MODE_LSA:
                    lsa_far_bias[il] = float(p.detach().item())
                continue
            # Save rel_bias/norms/clips/convs in F32, 2D weights in BF16 (compact ~432 MB)
            if k in ("attn_norm", "q_clip", "rel_clip", "rel_bias", "k_sconv", "v_sconv", "attn_sconv"):
                d[k] = p.detach().to("cpu", dtype=torch.float32)
            else:
                d[k] = p.detach().to("cpu", dtype=torch.bfloat16)
        layers_dict[il] = d

    payload = {
        "format": "inkling2_attn_surgery_v1",
        "n_phys": model.n_phys,
        "layer_modes": layer_modes,
        "swa_mid": 4096,
        "relative_extent_mid": 4096,
        "lsa_keep_min": 2048,
        "lsa_keep_frac": 0.04,
        "lsa_keep_max": 16384,
        "lsa_block": 64,
        "lsa_sinks": 128,
        "lsa_far_bias": lsa_far_bias,
        "mod_layers": mod_layers,
        "layers": layers_dict,
        "args": vars(args),
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default=None, help="Local dir with config.json + model-00001-of-00001.safetensors")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--hf-repo", type=str, default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--verify-logits", action="store_true")
    ap.add_argument("--verify-tokens", type=str, default="9906,11,1268,527,499,30")
    ap.add_argument("--verify-out", type=str, default=None)
    ap.add_argument("--train-pack", type=str, default="/content/train_pack.pt")
    ap.add_argument("--out-dir", type=str, default="/content/out_p13a")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--bias-lr", type=float, default=2.0e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--lambda-ce", type=float, default=0.5)
    ap.add_argument("--lambda-hid", type=float, default=0.25)
    ap.add_argument("--kl-len", type=int, default=4096)
    ap.add_argument("--ce-len", type=int, default=8192)
    ap.add_argument("--q-blk", type=int, default=256)
    ap.add_argument("--chunk-experts", type=int, default=16)
    ap.add_argument("--mid-slot", type=int, default=2)
    ap.add_argument("--lsa-slot", type=int, default=4)
    ap.add_argument("--mid-bias-init", type=str, default="tail", choices=["tail", "zero"])
    ap.add_argument("--mid-bias-scale", type=float, default=1.0)
    ap.add_argument("--lsa-far-bias-init", type=float, default=-4.0)
    ap.add_argument("--seed", type=int, default=20261124)
    return ap.parse_args()


if __name__ == "__main__":
    train_surgery(parse_args())
