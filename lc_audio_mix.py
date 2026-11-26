"""
LC Audio Mix
------------
Merges speech, sound effects, music and two extra tracks into one soundtrack: a level (dB) and a start time per track,
optional music ducking under speech, and a peak limiter so the sum never clips. Pairs with LC Audio Split: split an H3
clip, swap or re-level a part, mix it back. Pure numpy, no model.
"""

from __future__ import annotations

import math

import numpy as np

from .lc_av_media import audio_parts, make_audio

_TRACKS = ("speech", "sfx", "music", "track_4", "track_5")


def _resample(wave, sr, target):
    from .lc_av_media import resample

    return resample(wave, sr, target)


def _envelope(mono, sr, attack, release):
    """Smoothed level of a signal (fast attack, slow release), one value per sample."""
    hop = max(1, sr // 100)  # 10 ms blocks
    n = int(np.ceil(len(mono) / hop))
    pad = np.zeros(n * hop, dtype=np.float32)
    pad[:len(mono)] = np.abs(mono)
    blocks = pad.reshape(n, hop).max(1)
    a, r = np.exp(-1.0 / max(1, attack / 0.01)), np.exp(-1.0 / max(1, release / 0.01))
    env, out = 0.0, np.empty(n, dtype=np.float32)
    for i, v in enumerate(blocks):
        env = a * env + (1 - a) * v if v > env else r * env + (1 - r) * v
        out[i] = env
    return np.repeat(out, hop)[:len(mono)]


class LCAudioMix:
    @classmethod
    def INPUT_TYPES(cls):
        req = {}
        for t in _TRACKS:
            req[f"{t}_db"] = ("FLOAT", {"default": 0.0, "min": -60.0, "max": 12.0, "step": 0.5,
                                        "tooltip": f"Level of {t} in dB. 0 = as it is, -6 = half as loud, -60 = silent."})
            req[f"{t}_start"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.05,
                                           "tooltip": f"Seconds into the mix where {t} begins."})
        req["duck_music"] = ("FLOAT", {"default": 0.0, "min": -30.0, "max": 0.0, "step": 0.5,
                                       "tooltip": "Lowers the music while someone speaks (dB). 0 = off. -6 to -12 keeps "
                                                  "lines clear over a score. Needs a speech track."})
        req["length"] = (["longest track", "speech", "sfx", "music", "seconds"],
                         {"default": "longest track", "tooltip": "How long the mix is. 'seconds' uses the box below."})
        req["seconds"] = ("FLOAT", {"default": 10.0, "min": 0.1, "max": 3600.0, "step": 0.1,
                                    "tooltip": "Length when length = seconds."})
        req["match_loudness"] = ("BOOLEAN", {"default": False,
                                             "tooltip": "Brings every track to the same loudness (LUFS) before its dB "
                                                        "level, so 0 dB everywhere = equally loud. Use it when voices "
                                                        "from different engines or Foley come in at different levels."})
        req["target_lufs"] = ("FLOAT", {"default": 0.0, "min": -40.0, "max": 0.0, "step": 0.5,
                                        "tooltip": "Loudness of the finished mix. 0 = off. -14 YouTube and streaming, "
                                                   "-16 web and podcasts, -23 broadcast."})
        req["peak_db"] = ("FLOAT", {"default": -1.0, "min": -12.0, "max": 0.0, "step": 0.5,
                                    "tooltip": "Loudest the mix may peak. Off target: turns the whole mix down when "
                                               "the sum would go over. With a target: a limiter catches only the "
                                               "loud moments so the target holds."})
        return {"required": req, "optional": {t: ("AUDIO", {"tooltip": f"{t} (any length, rate or channels)"})
                                              for t in _TRACKS}}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "mix"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = ("Merges speech, sound effects, music and two extra tracks: a level and a start time each, optional "
                   "music ducking under speech, and a limiter so it never clips. Pairs with LC Audio Split.")

    def mix(self, length="longest track", seconds=10.0, duck_music=0.0, peak_db=-1.0, match_loudness=False,
            target_lufs=0.0, **kw):
        from .lc_loudness import limit, lufs

        tracks = {}
        for t in _TRACKS:
            wave, sr = audio_parts(kw.get(t))
            if wave is None or sr <= 0 or wave.size == 0:
                continue
            wave = np.asarray(wave, dtype=np.float32)
            if wave.ndim == 1:
                wave = wave[None]
            tracks[t] = (wave, sr)
        if not tracks:
            raise ValueError("LC Audio Mix: connect at least one track.")
        sr = max(s for _, s in tracks.values())
        chans = 2 if any(w.shape[0] > 1 for w, _ in tracks.values()) else 1
        placed, levels = {}, []
        for t, (w, s) in tracks.items():
            w = _resample(w, s, sr)
            w = np.repeat(w[:1], 2, 0) if (chans == 2 and w.shape[0] == 1) else w[:chans]
            start = int(round(float(kw.get(f"{t}_start", 0.0)) * sr))
            db = float(kw.get(f"{t}_db", 0.0))
            if match_loudness:
                cur = lufs(w, sr)
                if math.isfinite(cur):
                    db += max(-24.0, min(24.0, -23.0 - cur))  # every track to -23 LUFS, then its own level
                    levels.append(f"{t} {cur:.1f}")
            placed[t] = (start, w * 10 ** (db / 20))
        ends = {t: s + w.shape[-1] for t, (s, w) in placed.items()}
        if length == "seconds":
            n = int(round(float(seconds) * sr))
        elif length in ends:
            n = ends[length]
        else:
            n = max(ends.values())
        lay = {}
        for t, (s, w) in placed.items():
            buf = np.zeros((chans, n), dtype=np.float32)
            if s < n:
                seg = w[:, :max(0, n - s)]
                buf[:, s:s + seg.shape[-1]] = seg
            lay[t] = buf
        note = ""
        if duck_music < 0 and "music" in lay and "speech" in lay:
            env = _envelope(lay["speech"].mean(0), sr, attack=0.05, release=0.4)
            on = np.clip(env / max(1e-6, 0.1 * float(env.max())), 0, 1)  # full duck once speech is within 20 dB of its peak
            lay["music"] *= 10 ** ((float(duck_music) * on) / 20)
            note = f" Music ducked up to {duck_music:g} dB under speech."
        out = sum(lay.values())
        if levels:
            note += f" Matched loudness (LUFS in: {', '.join(levels)})."
        if target_lufs < 0:
            cur = lufs(out, sr)
            if math.isfinite(cur):
                out = out * 10 ** (max(-40.0, min(40.0, float(target_lufs) - cur)) / 20)
                out, gr = limit(out, sr, ceiling_db=float(peak_db))
                note += f" Loudness {cur:.1f} -> {lufs(out, sr):.1f} LUFS"
                note += f", limiter caught peaks by up to {gr:.1f} dB." if gr > 0.05 else "."
        else:
            peak = float(np.abs(out).max()) if out.size else 0.0
            ceil = 10 ** (float(peak_db) / 20)
            if peak > ceil:
                out *= ceil / peak
                note += f" Turned down {20 * np.log10(peak / ceil):.1f} dB to stay under {peak_db:g} dBFS."
        info = (f"Mixed {', '.join(lay)} at {sr} Hz, {'stereo' if chans == 2 else 'mono'}, {n / sr:.2f} s.{note}")
        return (make_audio(out, sr), info)


NODE_CLASS_MAPPINGS = {"LCAudioMix": LCAudioMix}
NODE_DISPLAY_NAME_MAPPINGS = {"LCAudioMix": "LC Audio Mix 🎛️"}
