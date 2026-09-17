"""
LC AV Pipe Out / Edit
---------------------
Audio/video workflow pipe. Separate type from LC_PIPE (image pack).

Naming follows LC123 pipes:
  - socket / dict keys are snake_case
  - the pipe socket is named pipe (type is still LC_AV_PIPE)
  - Out labels: Title Case for unique media / prompt names
  - Out labels: raw key when the name matches a widget (frame_rate, total_steps)

Pipe In is omitted. Civitai AIR is not a pipe slot.
"""

from .lc_av_types import LC_H3_PIPE, LC_H3_PIPE_V2, LC_PIPE, copy_shared_from_lc_pipe

PIPE_TYPE = "LC_AV_PIPE"

# Comma-joined so LC MiniMax H3 Pipe / V2 can feed these "pipe" sockets too --
# copy_shared_from_lc_pipe() and _normalize_pipe() already handle any dict
# via shared key names, so only the accepted socket type needed widening.
shared_pipe_in = f"{LC_PIPE},{LC_H3_PIPE},{LC_H3_PIPE_V2}"

SLOT_ORDER = [
    ("video", "IMAGE"),
    ("audio", "AUDIO"),
    ("frame_rate", "FLOAT"),
    ("duration", "FLOAT"),
    ("sample_rate", "INT"),
    ("width", "INT"),
    ("height", "INT"),
    ("frame_count", "INT"),
    ("positive_prompt", "STRING"),
    ("negative_prompt", "STRING"),
    ("seed", "INT"),
    ("total_steps", "INT"),
    ("sampler_name", "STRING"),
    ("models", "STRING"),
]

DISPLAY = {
    "video": "Video",
    "audio": "Audio",
    "frame_rate": "frame_rate",
    "duration": "duration",
    "sample_rate": "sample_rate",
    "width": "Width",
    "height": "Height",
    "frame_count": "frame_count",
    "positive_prompt": "Positive prompt",
    "negative_prompt": "Negative prompt",
    "seed": "Seed",
    "total_steps": "total_steps",
    "sampler_name": "sampler_name",
    "models": "Models",
}


def _empty():
    return {"_type": PIPE_TYPE}


def _slot_input(key, kind):
    label = DISPLAY.get(key, key)
    if kind == "STRING":
        return ("STRING", {
            "default": "",
            "multiline": True,
            "tooltip": label,
            "forceInput": True,
        })
    if kind == "INT":
        return ("INT", {
            "default": 0,
            "min": -1,
            "max": 0xFFFFFFFFFFFFFFFF,
            "tooltip": label,
            "forceInput": True,
        })
    if kind == "FLOAT":
        return ("FLOAT", {
            "default": 0.0,
            "min": 0.0,
            "max": 100000.0,
            "step": 0.01,
            "tooltip": label,
            "forceInput": True,
        })
    return (kind, {"tooltip": label})


def _is_provided(val):
    if val is None:
        return False
    if isinstance(val, str) and val == "":
        return False
    return True


def _normalize_pipe(pipe):
    """Accept LC_AV_PIPE, old AV keys, or LC_PIPE / LC_H3_PIPE(_V2) (shared fields only)."""
    if not isinstance(pipe, dict):
        return _empty()
    incoming = pipe.get("_type")
    if incoming in (LC_PIPE, LC_H3_PIPE, LC_H3_PIPE_V2):
        out = _empty()
        out.update(copy_shared_from_lc_pipe(pipe))
    else:
        out = dict(pipe)
    out["_type"] = PIPE_TYPE
    if "frame_rate" not in out and out.get("fps") not in (None, ""):
        out["frame_rate"] = out["fps"]
    if "total_steps" not in out and out.get("steps") not in (None, ""):
        out["total_steps"] = out["steps"]
    if "sampler_name" not in out and out.get("sampler") not in (None, ""):
        out["sampler_name"] = out["sampler"]
    if "audio" not in out and out.get("waveform") is not None:
        out["audio"] = out["waveform"]
    out.pop("civitai_air", None)
    return out


class LCAvPipeOut:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "pipe": (shared_pipe_in, {
                    "tooltip": "LC123 LC_PIPE or LC MiniMax H3 Pipe / V2. Overlays prompts/seed/steps/size. Not output.",
                }),
                "av_pipe": (PIPE_TYPE, {
                    "tooltip": "LC_AV_PIPE to unpack.",
                }),
            },
        }

    RETURN_TYPES = (PIPE_TYPE,) + tuple(kind for _, kind in SLOT_ORDER)
    RETURN_NAMES = ("av_pipe",) + tuple(DISPLAY[k] for k, _ in SLOT_ORDER)
    FUNCTION = "unpack"
    CATEGORY = "LC AV/pipe"
    DESCRIPTION = (
        "Inputs: pipe (LC_PIPE or H3 Pipe / V2), av_pipe. Out: av_pipe plus slots. No LC_PIPE out."
    )

    def unpack(self, pipe=None, av_pipe=None):
        base = _normalize_pipe(av_pipe)
        if isinstance(pipe, dict):
            for key, val in copy_shared_from_lc_pipe(pipe).items():
                if _is_provided(val):
                    base[key] = val
        values = tuple(base.get(key) for key, _ in SLOT_ORDER)
        return (base,) + values


class LCAvPipeEdit:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "pipe": (shared_pipe_in, {
                "tooltip": "LC123 LC_PIPE or LC MiniMax H3 Pipe / V2. Copies prompts/seed/steps/size only. Never written back.",
            }),
            "av_pipe": (PIPE_TYPE, {
                "tooltip": "Existing LC_AV_PIPE to edit inline. Unwired = pack from sockets (in).",
            }),
        }
        optional.update({key: _slot_input(key, kind) for key, kind in SLOT_ORDER})
        return {"required": {}, "optional": optional}

    RETURN_TYPES = (PIPE_TYPE,)
    RETURN_NAMES = ("av_pipe",)
    FUNCTION = "edit"
    CATEGORY = "LC AV/pipe"
    DESCRIPTION = (
        "In/edit for LC_AV_PIPE. Top sockets: pipe (LC_PIPE or H3 Pipe / V2), av_pipe (LC_AV_PIPE), "
        "then video, audio, … Out is av_pipe only."
    )

    def edit(self, pipe=None, av_pipe=None, **kwargs):
        base = _normalize_pipe(av_pipe)
        if isinstance(pipe, dict):
            for key, val in copy_shared_from_lc_pipe(pipe).items():
                if _is_provided(val):
                    base[key] = val
        for key, _kind in SLOT_ORDER:
            val = kwargs.get(key)
            if _is_provided(val):
                base[key] = val
        return (base,)


class LCPipeToAvPipe:
    """One socket: LC_PIPE (or LC MiniMax H3 Pipe / V2) in → LC_AV_PIPE out. No widgets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": (shared_pipe_in, {
                    "tooltip": "LC123 LC_PIPE or LC MiniMax H3 Pipe / V2. Copies prompts, seed, steps, sampler, size.",
                }),
            },
        }

    RETURN_TYPES = (PIPE_TYPE,)
    RETURN_NAMES = ("av_pipe",)
    FUNCTION = "convert"
    CATEGORY = "LC AV/pipe"
    DESCRIPTION = (
        "Convert LC_PIPE or LC MiniMax H3 Pipe / V2 to LC_AV_PIPE. Shared fields only. "
        "Does not copy models, CLIP, VAE, latent, image, or mask."
    )

    def convert(self, pipe):
        base = _empty()
        if isinstance(pipe, dict):
            base.update(copy_shared_from_lc_pipe(pipe))
            if "total_steps" not in base and base.get("steps") not in (None, ""):
                base["total_steps"] = base["steps"]
            if "sampler_name" not in base and base.get("sampler") not in (None, ""):
                base["sampler_name"] = base["sampler"]
        return (base,)


NODE_CLASS_MAPPINGS = {
    "LCAvPipeOut": LCAvPipeOut,
    "LCAvPipeEdit": LCAvPipeEdit,
    "LCPipeToAvPipe": LCPipeToAvPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCAvPipeOut": "LC AV Pipe Out 🎬",
    "LCAvPipeEdit": "LC AV Pipe (in/edit) 🎬",
    "LCPipeToAvPipe": "LC Pipe → AV Pipe 🔌",
}
