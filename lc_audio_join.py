"""
LC Audio Join
-------------
Puts up to six clips end to end. A positive gap adds silence between them; a negative gap overlaps them by that many
seconds with an equal-power crossfade. Every clip is converted to the first clip's sample rate and channel count.
The timings output says where each clip starts and ends in the joined audio (seconds, JSON).
"""

from __future__ import annotations

import json
import math

import numpy as np

from .lc_av_media import audio_parts, make_audio


def _resample(x, sr_from, sr_to):
    if sr_from == sr_to or x.shape[-1] == 0:
        return x
    from scipy.signal import resample_poly

    g = math.gcd(int(sr_from), int(sr_to))
    return resample_poly(x, sr_to // g, sr_from // g, axis=-1).astype(np.float32)


def _channels(x, c):
    if x.shape[0] == c:
        return x
    if x.shape[0] == 1:
        return np.repeat(x, c, axis=0)
    if c == 1:
        return x.mean(axis=0, keepdims=True)
    reps = -(-c // x.shape[0])
    return np.tile(x, (reps, 1))[:c]


class LCAudioJoin:
    @classmethod
    def INPUT_TYPES(cls):
        opt = {f"audio_{i}": ("AUDIO", {"tooltip": f"Clip {i}. Leave empty to skip."}) for i in range(2, 7)}
        return {
            "required": {
                "audio_1": ("AUDIO", {"tooltip": "First clip. Sets the sample rate and channels of the result."}),
                "gap_s": ("FLOAT", {"default": 0.3, "min": -5.0, "max": 10.0, "step": 0.05,
                                    "tooltip": "Silence between clips in seconds. Negative = overlap the clips by "
                                               "that long with a smooth crossfade."}),
            },
            "optional": opt,
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "timings", "info")
    FUNCTION = "join"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = "Joins up to six clips with a gap or a crossfade. Outputs where each clip lands (JSON)."

    def join(self, audio_1, gap_s=0.3, audio_2=None, audio_3=None, audio_4=None, audio_5=None, audio_6=None):
        clips = []
        for i, a in enumerate((audio_1, audio_2, audio_3, audio_4, audio_5, audio_6), start=1):
            w, sr = audio_parts(a)
            if w is not None and sr > 0 and w.size > 0:
                clips.append((i, np.asarray(w, dtype=np.float32), sr))
        if not clips:
            return (audio_1, "[]", "no sound")
        sr0 = clips[0][2]
        ch0 = clips[0][1].shape[0]
        gap = float(gap_s)
        out = np.zeros((ch0, 0), dtype=np.float32)
        timings = []
        for k, (idx, w, sr) in enumerate(clips):
            w = _channels(_resample(w, sr, sr0), ch0)
            if k == 0:
                start = 0
                out = w.copy()
            elif gap >= 0:
                start = out.shape[-1] + int(round(gap * sr0))
                out = np.concatenate([out, np.zeros((ch0, start - out.shape[-1]), np.float32), w], axis=-1)
            else:
                ov = min(int(round(-gap * sr0)), out.shape[-1], w.shape[-1])
                start = out.shape[-1] - ov
                if ov > 0:
                    t = np.linspace(0.0, 1.0, ov, dtype=np.float32)
                    fade_out = np.cos(t * np.pi / 2)
                    fade_in = np.sin(t * np.pi / 2)
                    mixed = out[:, start:] * fade_out + w[:, :ov] * fade_in
                    out = np.concatenate([out[:, :start], mixed, w[:, ov:]], axis=-1)
                else:
                    out = np.concatenate([out, w], axis=-1)
            timings.append({"clip": idx, "start": round(start / sr0, 3), "end": round((start + w.shape[-1]) / sr0, 3)})
        out = np.clip(out, -1.0, 1.0).astype(np.float32)
        kind = "gap" if gap >= 0 else "crossfade"
        info = (f"joined {len(clips)} clips -> {out.shape[-1] / sr0:.2f} s | {kind} {abs(gap):.2f} s | "
                f"{sr0} Hz, {ch0} ch")
        return (make_audio(out, sr0), json.dumps(timings), info)


NODE_CLASS_MAPPINGS = {"LCAudioJoin": LCAudioJoin}
NODE_DISPLAY_NAME_MAPPINGS = {"LCAudioJoin": "LC Audio Join 🔗"}
