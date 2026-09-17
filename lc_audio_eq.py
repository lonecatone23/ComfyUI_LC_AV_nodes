"""
LC Audio Equalizer (3-band)
---------------------------
Low shelf, peaking mid, high shelf. NumPy biquads — no Numba / SoX.
"""

from __future__ import annotations

import numpy as np

from .lc_av_media import audio_parts, make_audio


def _biquad(wave, b0, b1, b2, a0, a1, a2):
    """Direct-form I per channel. wave [C, T]."""
    b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
    a1, a2 = a1 / a0, a2 / a0
    out = np.zeros_like(wave)
    for c in range(wave.shape[0]):
        x1 = x2 = y1 = y2 = 0.0
        src = wave[c]
        dst = out[c]
        for i in range(src.shape[0]):
            x0 = float(src[i])
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            dst[i] = y0
            x2, x1 = x1, x0
            y2, y1 = y1, y0
    return out


def _low_shelf(wave, sr, freq, gain_db):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    cosw, sinw = np.cos(w0), np.sin(w0)
    S = 1.0
    alpha = sinw / 2.0 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    b0 = A * ((A + 1) - (A - 1) * cosw + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
    b2 = A * ((A + 1) - (A - 1) * cosw - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * cosw + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cosw)
    a2 = (A + 1) + (A - 1) * cosw - 2 * np.sqrt(A) * alpha
    return _biquad(wave, b0, b1, b2, a0, a1, a2)


def _high_shelf(wave, sr, freq, gain_db):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    cosw, sinw = np.cos(w0), np.sin(w0)
    S = 1.0
    alpha = sinw / 2.0 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    b0 = A * ((A + 1) + (A - 1) * cosw + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
    b2 = A * ((A + 1) + (A - 1) * cosw - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cosw + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cosw)
    a2 = (A + 1) - (A - 1) * cosw - 2 * np.sqrt(A) * alpha
    return _biquad(wave, b0, b1, b2, a0, a1, a2)


def _peaking(wave, sr, freq, gain_db, q):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    cosw, sinw = np.cos(w0), np.sin(w0)
    alpha = sinw / (2.0 * max(q, 0.05))
    b0 = 1 + alpha * A
    b1 = -2 * cosw
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cosw
    a2 = 1 - alpha / A
    return _biquad(wave, b0, b1, b2, a0, a1, a2)


class LCAudioEqualizer3Band:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "Comfy AUDIO to EQ. Stereo is processed per channel. 0 dB bands are skipped so they do not color the signal.",
                    },
                ),
                "low_gain_dB": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -24.0,
                        "max": 24.0,
                        "step": 0.1,
                        "tooltip": "Low-shelf gain in dB. Positive boosts bass below low_freq; negative cuts it. 0 skips this band.",
                    },
                ),
                "low_freq": (
                    "INT",
                    {
                        "default": 100,
                        "min": 20,
                        "max": 500,
                        "tooltip": "Low-shelf corner in Hz (typical 60–150). Everything below this is boosted or cut by low_gain_dB.",
                    },
                ),
                "mid_gain_dB": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -24.0,
                        "max": 24.0,
                        "step": 0.1,
                        "tooltip": "Peaking-mid gain in dB around mid_freq. Use small cuts to tame boxiness; 0 skips this band.",
                    },
                ),
                "mid_freq": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 200,
                        "max": 4000,
                        "tooltip": "Center of the mid peak in Hz (voice body ~800–1500, presence ~2–4 kHz).",
                    },
                ),
                "mid_q": (
                    "FLOAT",
                    {
                        "default": 0.707,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.001,
                        "tooltip": "Width of the mid peak. 0.7 is broad; higher Q is a narrower notch/boost. Only used when mid_gain_dB is not 0.",
                    },
                ),
                "high_gain_dB": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -24.0,
                        "max": 24.0,
                        "step": 0.1,
                        "tooltip": "High-shelf gain in dB. Positive adds air above high_freq; negative dulls hiss. 0 skips this band.",
                    },
                ),
                "high_freq": (
                    "INT",
                    {
                        "default": 5000,
                        "min": 1000,
                        "max": 15000,
                        "tooltip": "High-shelf corner in Hz (typical 4–8 kHz). Everything above this follows high_gain_dB.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "eq"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = "3-band EQ: low shelf, peaking mid, high shelf. 0 dB bands are skipped."

    def eq(
        self,
        audio,
        low_gain_dB=0.0,
        low_freq=100,
        mid_gain_dB=0.0,
        mid_freq=1000,
        mid_q=0.707,
        high_gain_dB=0.0,
        high_freq=5000,
    ):
        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0:
            return (audio,)
        if wave.ndim == 1:
            wave = wave[None, :]
        out = wave.astype(np.float32, copy=True)
        if low_gain_dB:
            out = _low_shelf(out, sr, float(low_freq), float(low_gain_dB))
        if mid_gain_dB:
            out = _peaking(out, sr, float(mid_freq), float(mid_gain_dB), float(mid_q))
        if high_gain_dB:
            out = _high_shelf(out, sr, float(high_freq), float(high_gain_dB))
        return (make_audio(out, sr),)


NODE_CLASS_MAPPINGS = {
    "LCAudioEqualizer3Band": LCAudioEqualizer3Band,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCAudioEqualizer3Band": "LC Audio EQ 3-Band 🎛️",
}
