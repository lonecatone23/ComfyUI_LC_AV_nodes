"""
Loudness helpers for the LC AV audio nodes: integrated loudness (LUFS, ITU-R BS.1770 K-weighting with gating) and a
look-ahead peak limiter. Arrays are float32 (channels, samples).

Adapted from Score Studio's dsp.py (ComfyUI-ScoreStudio, MIT): RBJ approximations of the BS.1770 filters, the gated
block measure, and the limiter's sliding-maximum + one-pole release.
"""

from __future__ import annotations

import math

import numpy as np


def _biquad(kind, sr, f0, gain_db, q):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * min(f0, sr * 0.49) / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    if kind == "highshelf":
        sa = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    else:  # highpass
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def _k_weight(x, sr):
    from scipy import signal

    b1, a1 = _biquad("highshelf", sr, 1681.97, 3.99984, 0.7071752)
    b2, a2 = _biquad("highpass", sr, 38.1355, 0.0, 0.5003270)
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=-1), axis=-1)


def lufs(x, sr) -> float:
    """Gated integrated loudness in LUFS; -inf for silence."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None]
    y = _k_weight(x, sr)
    block, hop = int(round(0.4 * sr)), int(round(0.1 * sr))
    if y.shape[-1] < block:
        y = np.pad(y, ((0, 0), (0, block - y.shape[-1])))
    starts = np.arange(0, y.shape[-1] - block + 1, hop)
    csum = np.concatenate([np.zeros((y.shape[0], 1)), np.cumsum(y ** 2, axis=-1)], axis=-1)
    power = (csum[:, starts + block] - csum[:, starts]).sum(0) / block  # channel weights 1.0 (mono / stereo)
    lk = -0.691 + 10 * np.log10(np.maximum(power, 1e-12))
    keep = lk > -70.0
    if not keep.any():
        return float("-inf")
    rel = -0.691 + 10 * np.log10(power[keep].mean()) - 10.0
    keep &= lk > rel
    if not keep.any():
        return float("-inf")
    return float(-0.691 + 10 * np.log10(power[keep].mean()))


def limit(x, sr, ceiling_db=-1.0, release_ms=80.0, lookahead_ms=3.0):
    """Look-ahead peak limiter: only the loud moments are turned down, the rest keeps its level."""
    from scipy import signal
    from scipy.ndimage import maximum_filter1d

    ceiling = 10 ** (ceiling_db / 20.0)
    peak = np.max(np.abs(x), axis=0)
    la = int(sr * lookahead_ms / 1000.0)
    if la > 0:
        w = la + 1
        peak = maximum_filter1d(np.pad(peak, (0, la), mode="edge"), size=w, origin=-(w // 2), mode="nearest")[:x.shape[-1]]
    need = 1.0 / np.maximum(peak / ceiling, 1.0)
    coef = math.exp(-1.0 / (sr * release_ms / 1000.0))
    b, a = [1.0 - coef], [1.0, -coef]
    smooth, _ = signal.lfilter(b, a, need, zi=signal.lfilter_zi(b, a) * 1.0)
    gain = np.minimum(need, smooth)
    return np.clip(x * gain[None, :], -ceiling, ceiling).astype(np.float32), float(-20 * np.log10(max(gain.min(), 1e-9)))


def compress(x, sr, threshold_db=-18.0, ratio=3.0, attack_ms=10.0, release_ms=120.0, knee_db=6.0):
    """Feed-forward compressor (soft knee). The level follows the louder channel in 1 ms steps, the gain is smoothed
    with separate attack / release and spread back over the samples. Returns (audio, most gain reduction in dB)."""
    hop = max(1, int(sr / 1000))
    n = x.shape[-1]
    frames = -(-n // hop)
    pk = np.abs(np.pad(x, ((0, 0), (0, frames * hop - n)))).max(0).reshape(frames, hop).max(1)
    level = 20 * np.log10(np.maximum(pk, 1e-9))
    over = level - threshold_db
    half = knee_db / 2.0
    red = np.where(over <= -half, 0.0,
                   np.where(over >= half, over * (1 - 1 / ratio),
                            (1 - 1 / ratio) * (over + half) ** 2 / (2 * max(knee_db, 1e-6))))
    a_c = math.exp(-1.0 / max(attack_ms, 0.1))  # per 1 ms frame
    r_c = math.exp(-1.0 / max(release_ms, 0.1))
    g = np.empty(frames)
    cur = 0.0
    for i in range(frames):  # ~1000 steps a second: fine in plain Python
        target = red[i]
        c = a_c if target > cur else r_c
        cur = c * cur + (1 - c) * target
        g[i] = cur
    gain = 10 ** (-np.interp(np.arange(n), np.arange(frames) * hop + hop / 2, g) / 20)
    return (x * gain[None, :]).astype(np.float32), float(g.max() if frames else 0.0)


def gain_to_lufs(x, sr, target, max_db=24.0):
    """Gain that brings x to `target` LUFS (clamped to +-max_db); 0 dB for silence."""
    cur = lufs(x, sr)
    if not np.isfinite(cur):
        return 0.0
    return float(np.clip(target - cur, -max_db, max_db))
