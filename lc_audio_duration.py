"""
LC Audio Duration
-----------------
Seconds as INT (floor) and FLOAT, plus sample metadata.
"""

from .lc_av_media import audio_parts


class LCAudioDuration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Comfy AUDIO {waveform, sample_rate}."}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = ("seconds_int", "seconds_float", "minutes_int", "minutes_float")
    FUNCTION = "measure"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = (
        "Duration from AUDIO. Matches Audio Duration: seconds_int / seconds_float "
        "and minutes_int / minutes_float."
    )

    def measure(self, audio):
        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0:
            return (0, 0.0, 0, 0.0)
        samples = int(wave.shape[-1])
        sec = float(samples) / float(sr)
        mins = sec / 60.0
        return (int(sec), sec, int(mins), mins)


NODE_CLASS_MAPPINGS = {
    "LCAudioDuration": LCAudioDuration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCAudioDuration": "LC Audio Duration ⏱️",
}
