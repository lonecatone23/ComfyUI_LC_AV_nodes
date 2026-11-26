"""
LC Audio Speed
--------------
Plays audio faster or slower without changing the pitch (librosa's phase vocoder, ported to numpy in lc_dsp, one channel at a time).
speed 1.25 = 25 % faster and shorter, 0.8 = slower and longer. "speech (smoother)" uses more window overlap, which
sounds less phasey on voices; "music" uses a shorter window that keeps drum hits sharper.
"""

from __future__ import annotations

import numpy as np

from .lc_av_media import audio_parts, make_audio

_QUALITY = {  # n_fft, hop at 44.1 / 48 kHz (scaled for other rates)
    "speech (smoother)": (2048, 256),
    "music": (1024, 256),
}


class LCAudioSpeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01, "display": "slider",
                                    "tooltip": "1 = unchanged. 1.25 = 25% faster (shorter), 0.8 = slower (longer). "
                                               "Pitch stays the same."}),
                "quality": (list(_QUALITY), {"default": "speech (smoother)",
                                             "tooltip": "speech: smoother voices. music: sharper drum hits."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "stretch"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = "Faster or slower audio with the same pitch (0.5x to 2x)."

    def stretch(self, audio, speed=1.0, quality="speech (smoother)"):
        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0 or wave.size == 0 or float(np.abs(wave).max()) < 1e-6:
            return (audio, "no sound")
        speed = float(speed)
        n = wave.shape[-1]
        if abs(speed - 1.0) < 1e-4:
            return (audio, f"unchanged ({n / sr:.2f} s)")
        from . import lc_dsp as librosa_lite

        n_fft, hop = _QUALITY.get(quality, _QUALITY["speech (smoother)"])
        scale = max(0.25, min(4.0, sr / 44100.0))
        n_fft = int(2 ** round(np.log2(n_fft * scale)))
        hop = max(32, int(2 ** round(np.log2(hop * scale))))
        target = int(round(n / speed))
        chans = []
        for ch in np.asarray(wave, dtype=np.float32):
            y = librosa_lite.time_stretch(ch, rate=speed, n_fft=n_fft, hop_length=hop)
            y = librosa_lite.fix_length(y, size=target)
            chans.append(y)
        out = np.stack(chans).astype(np.float32)
        peak_in = float(np.abs(wave).max())
        peak_out = float(np.abs(out).max())
        if peak_out > max(peak_in, 1e-9):  # phase vocoder can nudge peaks up a little
            out *= peak_in / peak_out
        info = f"speed {speed:.2f}x | {n / sr:.2f} s -> {target / sr:.2f} s | {quality}"
        return (make_audio(out, sr), info)


NODE_CLASS_MAPPINGS = {"LCAudioSpeed": LCAudioSpeed}
NODE_DISPLAY_NAME_MAPPINGS = {"LCAudioSpeed": "LC Audio Speed ⏩"}
