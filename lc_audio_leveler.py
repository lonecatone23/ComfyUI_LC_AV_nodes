"""
LC Audio Leveler
----------------
Evens out a voice track: a compressor takes the edge off loud words, then the track goes to a loudness target
(LUFS) and a look-ahead limiter catches the last peaks. The info output says how hard each stage worked, so a limiter
doing 7 dB of work shows up instead of hiding.
"""

from __future__ import annotations

import numpy as np

from .lc_av_media import audio_parts, make_audio
from .lc_loudness import compress, gain_to_lufs, limit, lufs

_PRESETS = {  # threshold dB, ratio, attack ms, release ms
    "Dialogue (gentle)": (-20.0, 2.5, 8.0, 150.0),
    "Dialogue (even)": (-24.0, 4.0, 5.0, 120.0),
    "Narration / podcast": (-22.0, 3.0, 10.0, 200.0),
    "Off (loudness + limiter only)": None,
    "Custom": None,
}


class LCAudioLeveler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "preset": (list(_PRESETS), {"default": "Dialogue (gentle)",
                                            "tooltip": "Compressor settings. Custom uses the sliders below."}),
                "target_lufs": ("FLOAT", {"default": -16.0, "min": -36.0, "max": -6.0, "step": 0.5,
                                          "tooltip": "-16 for video and social posts, -14 for music platforms, "
                                                     "-23 for broadcast."}),
                "peak_db": ("FLOAT", {"default": -1.0, "min": -12.0, "max": 0.0, "step": 0.1,
                                      "tooltip": "The limiter's ceiling."}),
            },
            "optional": {
                "threshold_db": ("FLOAT", {"default": -20.0, "min": -60.0, "max": 0.0, "step": 0.5,
                                           "tooltip": "Custom only. Anything louder gets compressed."}),
                "ratio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1,
                                    "tooltip": "Custom only. 2-4 for voices; 10+ acts like a limiter."}),
                "attack_ms": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 200.0, "step": 0.1,
                                        "tooltip": "Custom only. Shorter catches word onsets."}),
                "release_ms": ("FLOAT", {"default": 150.0, "min": 10.0, "max": 2000.0, "step": 5.0,
                                         "tooltip": "Custom only. Longer sounds smoother."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "level"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = "Compressor + loudness target (LUFS) + peak limiter for voices and finished mixes."

    def level(self, audio, preset="Dialogue (gentle)", target_lufs=-16.0, peak_db=-1.0, threshold_db=-20.0, ratio=3.0,
              attack_ms=8.0, release_ms=150.0):
        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0 or wave.size == 0:
            return (audio, "no sound")
        x = np.asarray(wave, dtype=np.float32)
        before = lufs(x, sr)
        settings = (threshold_db, ratio, attack_ms, release_ms) if preset == "Custom" else _PRESETS[preset]
        comp = 0.0
        if settings and settings[1] > 1.0:
            # compress relative to the track's own loudness so the preset works on quiet and hot takes alike
            pre = gain_to_lufs(x, sr, -20.0)
            x = x * 10 ** (pre / 20)
            x, comp = compress(x, sr, *settings)
        g = gain_to_lufs(x, sr, target_lufs)
        x = x * 10 ** (g / 20)
        x, lim = limit(x, sr, peak_db)
        after = lufs(x, sr)
        info = (f"in {before:.1f} LUFS -> out {after:.1f} LUFS | compressor up to {comp:.1f} dB | "
                f"limiter up to {lim:.1f} dB")
        return (make_audio(x, sr), info)


NODE_CLASS_MAPPINGS = {"LCAudioLeveler": LCAudioLeveler}
NODE_DISPLAY_NAME_MAPPINGS = {"LCAudioLeveler": "LC Audio Leveler 📏"}
