"""
LC Audio Channel
----------------
Stereo L/R (or mono duplicate). This is the old channel-split backend,
kept as its own node so LC Audio Separate can match Demucs stems.
"""

from .lc_av_media import audio_parts, make_audio


class LCAudioChannel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input AUDIO."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("left", "right", "note")
    FUNCTION = "split"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = "Split stereo to L/R. Mono is duplicated to both outputs."

    def split(self, audio):
        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0:
            empty = make_audio([[0.0]], max(sr, 1))
            return (empty, empty, "no audio")
        if wave.ndim == 1:
            wave = wave[None, :]
        if wave.shape[0] == 1:
            return (make_audio(wave, sr), make_audio(wave, sr), "mono duplicated")
        return (
            make_audio(wave[0:1], sr),
            make_audio(wave[1:2], sr),
            "stereo L/R",
        )


NODE_CLASS_MAPPINGS = {
    "LCAudioChannel": LCAudioChannel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCAudioChannel": "LC Audio Channel 🎚️",
}
