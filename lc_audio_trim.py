"""
LC Audio Trim
-------------
Cuts the silence off a clip. "start + end" trims only the edges; "every gap" also shortens every pause longer than
max_gap_s down to max_gap_s. Silence is measured against the clip's own loudest moment (threshold_db below the peak),
so it works on quiet and hot recordings alike. Cuts get a short fade so they never click.
"""

from __future__ import annotations

import numpy as np

from .lc_av_media import audio_parts, make_audio

_MODES = ["start + end", "every gap"]


def _voiced_regions(x, sr, threshold_db):
    """Sample ranges [(a, b)] louder than (peak + threshold_db). 20 ms windows, 10 ms hop."""
    hop = max(1, int(sr * 0.010))
    win = max(hop, int(sr * 0.020))
    power = np.mean(x.astype(np.float64) ** 2, axis=0)
    n = power.shape[0]
    csum = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, max(n - 1, 0) + 1, hop)
    ends = np.minimum(starts + win, n)
    level = (csum[ends] - csum[starts]) / np.maximum(ends - starts, 1)
    db = 10 * np.log10(np.maximum(level, 1e-14))
    loud = db > (db.max() + threshold_db)
    regions = []
    i = 0
    while i < len(loud):
        if loud[i]:
            j = i
            while j + 1 < len(loud) and loud[j + 1]:
                j += 1
            a = int(starts[i])
            b = int(ends[j])
            if regions and a <= regions[-1][1]:
                regions[-1] = (regions[-1][0], b)
            else:
                regions.append((a, b))
            i = j + 1
        else:
            i += 1
    return regions


def _fade(seg, n_fade, fade_in, fade_out):
    n_fade = min(n_fade, seg.shape[-1] // 2)
    if n_fade <= 1:
        return seg
    seg = seg.copy()
    ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    if fade_in:
        seg[:, :n_fade] *= ramp
    if fade_out:
        seg[:, -n_fade:] *= ramp[::-1]
    return seg


class LCAudioTrim:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (_MODES, {"default": "start + end",
                                  "tooltip": "start + end: trim the edges only. every gap: also shorten long pauses "
                                             "inside the clip."}),
                "threshold_db": ("FLOAT", {"default": -45.0, "min": -90.0, "max": -10.0, "step": 1.0,
                                           "tooltip": "How far below the loudest moment counts as silence. "
                                                      "-45 suits clean voices; -30 for noisy rooms."}),
                "pad_ms": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 1000.0, "step": 5.0,
                                     "tooltip": "Silence kept before the first sound and after the last one, "
                                                "so breaths and word tails survive."}),
                "max_gap_s": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 10.0, "step": 0.05,
                                        "tooltip": "every gap only. Longest pause allowed inside the clip."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "trim"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = "Trims silence from the start and end, or shortens every long pause. Cuts are faded, no clicks."

    def trim(self, audio, mode="start + end", threshold_db=-45.0, pad_ms=40.0, max_gap_s=0.4):
        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0 or wave.size == 0 or float(np.abs(wave).max()) < 1e-6:
            return (audio, "no sound")
        x = np.asarray(wave, dtype=np.float32)
        n = x.shape[-1]
        regions = _voiced_regions(x, sr, float(threshold_db))
        if not regions:
            return (audio, "no sound")
        pad = int(sr * float(pad_ms) / 1000.0)
        first = max(0, regions[0][0] - pad)
        last = min(n, regions[-1][1] + pad)
        keep = [(first, last)]
        shortened = 0
        if mode == "every gap":
            max_gap = int(sr * float(max_gap_s))
            half = max_gap // 2
            keep = []
            a = first
            for (_, end), (start, _) in zip(regions[:-1], regions[1:]):
                if start - end > max_gap:
                    keep.append((a, end + half))
                    a = start - (max_gap - half)
                    shortened += 1
            keep.append((a, last))
        fade = max(1, int(sr * 0.005))
        parts = []
        for i, (a, b) in enumerate(keep):
            seg = x[:, a:b]
            parts.append(_fade(seg, fade, fade_in=(i > 0 or a > 0), fade_out=(i < len(keep) - 1 or b < n)))
        out = np.concatenate(parts, axis=-1).astype(np.float32)
        removed = (n - out.shape[-1]) / sr
        info = (f"removed {removed:.2f} s ({n / sr:.2f} s -> {out.shape[-1] / sr:.2f} s) | "
                f"start {first / sr:.2f} s, end {(n - last) / sr:.2f} s")
        if mode == "every gap":
            info += f" | {shortened} gaps shortened to {float(max_gap_s):.2f} s"
        return (make_audio(out, sr), info)


NODE_CLASS_MAPPINGS = {"LCAudioTrim": LCAudioTrim}
NODE_DISPLAY_NAME_MAPPINGS = {"LCAudioTrim": "LC Audio Trim ✂️"}
