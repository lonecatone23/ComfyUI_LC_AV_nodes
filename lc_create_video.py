"""
LC Create Video
---------------
Pack an IMAGE batch into an LC_AV_PIPE with fps / duration.
Optional AUDIO is stored on the pipe (mux happens at Save Video).
"""

from .lc_av_media import audio_duration_sec, tensor_to_uint8_frames
from .lc_av_pipe import PIPE_TYPE, _empty, _normalize_pipe
from .lc_av_types import pipe_in


class LCCreateVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Frame batch [B,H,W,C]. Order is playback order.",
                }),
                "frame_rate": ("FLOAT", {
                    "default": 24.0,
                    "min": 0.01,
                    "max": 240.0,
                    "step": 0.01,
                    "tooltip": "Playback rate. duration = frame_count / frame_rate.",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Optional soundtrack. Stored on LC_AV_PIPE; Save Video muxes when possible.",
                }),
                "pipe": (pipe_in, {
                    "tooltip": "LC_AV_PIPE or LC_PIPE. LC_PIPE contributes prompts/seed/steps/size only.",
                }),
            },
        }

    RETURN_TYPES = (PIPE_TYPE, "IMAGE", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("pipe", "images", "frame_rate", "duration", "frame_count")
    FUNCTION = "create"
    CATEGORY = "LC AV/video"
    DESCRIPTION = (
        "Create an LC_AV_PIPE from an image batch. "
        "Does not encode a file — wire LC Save Video for that."
    )

    def create(self, images, frame_rate=24.0, audio=None, pipe=None):
        frames = tensor_to_uint8_frames(images)
        n, h, w, _c = frames.shape
        fps_f = float(frame_rate) if frame_rate else 24.0
        if fps_f <= 0:
            fps_f = 24.0
        duration = float(n) / fps_f
        aud_dur = audio_duration_sec(audio)

        out = _normalize_pipe(pipe) if pipe is not None else _empty()
        out["video"] = images
        out["frame_rate"] = fps_f
        out["duration"] = duration
        out["width"] = int(w)
        out["height"] = int(h)
        out["frame_count"] = int(n)
        if audio is not None:
            out["audio"] = audio
            if isinstance(audio, dict) and audio.get("sample_rate"):
                out["sample_rate"] = int(audio["sample_rate"])
            if aud_dur:
                out["audio_duration"] = aud_dur
        return (out, images, fps_f, duration, int(n))


NODE_CLASS_MAPPINGS = {
    "LCCreateVideo": LCCreateVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCCreateVideo": "LC Create Video 🎬",
}
