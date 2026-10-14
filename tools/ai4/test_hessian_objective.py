#!/usr/bin/env python3
"""CPU algebra checks for the full-K scoring and feedback audit."""

import numpy as np


def hadamard(n):
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def score(w, q, h):
    e = w - q
    return np.einsum("ri,ij,rj->", e, h, e) / np.einsum("ri,ij,rj->", w, h, w)


def main():
    rng = np.random.default_rng(193)
    n, rb = 512, 256
    sign_words = [0x9E3779B9, 0x7F4A7C15, 0xF39CC060, 0x5CED2A1B,
                  0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A]
    signs = np.array([-1 if (sign_words[i // 32] >> (i % 32)) & 1 else 1 for i in range(rb)])
    one = hadamard(rb) * signs[None, :]
    r = np.zeros((n, n))
    r[:rb, :rb] = one
    r[rb:, rb:] = one
    x = rng.normal(size=(n, 12))
    h = x @ x.T / 12 + np.diag(rng.uniform(0.2, 1.4, n))
    w = rng.normal(size=(3, n))
    q = w + 0.01 * x[:, 0][None, :] + rng.normal(scale=0.002, size=w.shape)
    wr, qr, hr = w @ r.T, q @ r.T, r @ h @ r.T
    full = score(w, q, h)
    rotated = score(wr, qr, hr)
    assert np.isclose(full, rotated, rtol=1e-12)
    block = hr.copy()
    block[:rb, rb:] = 0
    block[rb:, :rb] = 0
    diag = r @ np.diag(np.diag(h)) @ r.T
    print(f"full original-basis error {full:.12e}; full rotated-basis {rotated:.12e}")
    for name, fit_h in [("full", hr), ("blockdiag", block), ("diag", diag)]:
        before = score(wr, qr, fit_h)
        after = score(wr, qr, hr)
        assert np.isclose(after, full, rtol=1e-12)
        print(f"{name}: old reported {before:.12e}; corrected {after:.12e}; old/full {before/full:.6f}")
    # U is the transpose of lower Cholesky(H^-1), as in the cuBLAS row-major view.
    u = np.linalg.cholesky(np.linalg.inv(h)).T
    b = 8
    feedback = np.linalg.solve(u[:b, :b], u[:b, b:])
    optimal = -np.linalg.solve(h[b:, b:], h[b:, :b]).T
    maximum = np.max(np.abs(feedback - optimal))
    assert maximum < 1e-12
    print(f"block LDLQ feedback vs exact conditional minimizer max abs {maximum:.3e}")
    assert not np.isclose(score(wr, qr, diag), full, rtol=0.01)

    # A closed-form row refit minimizes float scales, but those are not the emitted fp16 scales.
    levels = np.array([11, -6, 3, -9], dtype=np.float64)
    group = np.array([0, 0, 1, 1])
    d = np.array([0.125, 1.0], dtype=np.float16).astype(np.float64)
    q0 = d[group] * levels
    h4 = np.eye(4) + 0.25 * np.ones((4, 4))
    noise = np.array([0.03, -0.01, 0.02, 0.005])
    noise -= q0 * ((noise @ h4 @ q0) / (q0 @ h4 @ q0))
    target = q0 * 1.0003 + noise
    c = (target @ h4 @ q0) / (q0 @ h4 @ q0)
    ideal = q0 * c
    emitted_scales = (d * c).astype(np.float16).astype(np.float64)
    emitted = emitted_scales[group] * levels
    old = score(target[None, :], ideal[None, :], h4)
    actual = score(target[None, :], emitted[None, :], h4)
    assert abs(actual - old) / actual > 1e-5
    print(f"refit: ideal-scale score {old:.12e}; emitted-fp16 score {actual:.12e}; old/actual {old/actual:.6f}")
    assert np.array_equal(emitted, emitted_scales[group] * levels)
    print("PASS: refit scoring must use the reconstructed emitted fp16 scales")
    print("PASS: full-H scoring invariance and feedback orientation")


if __name__ == "__main__":
    main()
