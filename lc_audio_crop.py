"""
LC Audio Crop
-------------
Trim AUDIO by start_time / end_time strings (MM:SS), same as VRGDG_AudioCrop.
"""

from .lc_av_media import audio_parts, make_audio


def _parse_timestamp(text) -> float:
    """'0:00', '1:00', '1:02:03', or a plain number of seconds."""
    s = str(text or "").strip()
    if not s:
        return 0.0
    if ":" not in s:
        try:
            return max(0.0, float(s))
        except ValueError:
            return 0.0
    parts = s.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return 0.0
    if len(nums) == 2:
        return nums[0] * 60.0 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]
    return 0.0


class LCAudioCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_time": (
                    "STRING",
                    {
                        "default": "0:00",
                        "tooltip": "MM:SS (or H:MM:SS). Start of the kept region.",
                    },
                ),
                "end_time": (
                    "STRING",
                    {
                        "default": "1:00",
                        "tooltip": "MM:SS (or H:MM:SS). End of the kept region.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "crop"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = "Crop AUDIO from start_time to end_time (MM:SS), same layout as VRGDG_AudioCrop."

    def crop(self, audio, start_time="0:00", end_time="1:00"):
        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0:
            return (audio,)
        if wave.ndim == 1:
            wave = wave[None, :]
        n = wave.shape[-1]
        a = max(0, int(_parse_timestamp(start_time) * sr))
        b = int(_parse_timestamp(end_time) * sr)
        if b <= 0:
            b = n
        b = min(n, b)
        if b < a:
            a, b = b, a
        return (make_audio(wave[:, a:b], sr),)


NODE_CLASS_MAPPINGS = {
    "LCAudioCrop": LCAudioCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCAudioCrop": "LC Audio Crop ✂️",
}
