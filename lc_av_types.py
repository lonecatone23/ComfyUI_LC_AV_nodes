"""
Type adapters so AV can read LC123 pipes / metadata without rewriting LC_PIPE.

Same pattern as LC MiniMax H3 Pipe: accept LC_PIPE, copy only shared keys,
never mutate the incoming dict, never put video/audio onto LC_PIPE.
"""

LC_PIPE = "LC_PIPE"
LC_AV_PIPE = "LC_AV_PIPE"
LC_SAVE_META = "LC_SAVE_META"
LC_SAVE_VIDEO_META = "LC_SAVE_VIDEO_META"

# Keys that exist on both packs. Copied LC_PIPE → LC_AV_PIPE only.
SHARED_FROM_LC_PIPE = (
    "width",
    "height",
    "positive_prompt",
    "negative_prompt",
    "seed",
    "total_steps",
    "steps",
    "sampler_name",
    "sampler",
    "scheduler",
    "denoise",
    "cfg_1",
    "cfg",
)


class PipeAccept(str):
    """Wire LC_AV_PIPE or LC_PIPE (image pack)."""

    def __ne__(self, other):
        o = str(other) if other is not None else ""
        return o not in {LC_AV_PIPE, LC_PIPE, "*"}


class MetaAccept(str):
    """Wire LC123 Save Metadata or AV Save Video Metadata."""

    def __ne__(self, other):
        o = str(other) if other is not None else ""
        return o not in {LC_SAVE_META, LC_SAVE_VIDEO_META, "*"}


pipe_in = PipeAccept(LC_AV_PIPE)
meta_in = MetaAccept(LC_SAVE_META)


def copy_shared_from_lc_pipe(src):
    """Return a new dict of shared fields only. Does not touch src."""
    if not isinstance(src, dict):
        return {}
    out = {}
    for k in SHARED_FROM_LC_PIPE:
        if k in src and src[k] is not None and src[k] != "":
            out[k] = src[k]
    # IMAGE on LC_PIPE is still an image — do not treat it as video frames here.
    return out
