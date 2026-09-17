"""
LC Audio Volume
---------------
Gain slider. 1.0 = unity. Soft clip at ±1 after apply.
"""

from __future__ import annotations

import numpy as np

from .lc_av_media import audio_parts, make_audio


class LCAudioVolume:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "volume": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Linear gain. 0 = mute, 1 = unchanged, 2 = +6 dB.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "apply"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = "Volume slider for AUDIO. Soft-clips after gain."

    def apply(self, audio, volume=1.0):
        wave, sr = audio_parts(audio)
        gain = float(volume)
        if gain == 1.0:
            return (audio,)
        out = np.clip(wave.astype(np.float32) * gain, -1.0, 1.0)
        return (make_audio(out, sr),)


NODE_CLASS_MAPPINGS = {
    "LCAudioVolume": LCAudioVolume,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCAudioVolume": "LC Audio Volume 🔊",
}
