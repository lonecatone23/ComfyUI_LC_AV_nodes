"""
Type adapters so AV can read LC123 pipes / metadata without rewriting LC_PIPE.

Same pattern as LC MiniMax H3 Pipe: accept LC_PIPE, copy only shared keys,
never mutate the incoming dict, never put video/audio onto LC_PIPE.
"""

LC_PIPE = "LC_PIPE"
LC_AV_PIPE = "LC_AV_PIPE"
LC_SAVE_META = "LC_SAVE_META"
LC_SAVE_VIDEO_META = "LC_SAVE_VIDEO_META"
# LC123's MiniMax H3 pipe types -- string literals rather than an import so
# this pack never takes a hard dependency on ComfyUI_LC123_nodes being
# installed (same reasoning LC_PIPE above is a literal, not an import).
LC_H3_PIPE = "LC_H3_PIPE"
LC_H3_PIPE_V2 = "LC_H3_PIPE_V2"

# Keys that exist on both packs. Copied LC_PIPE / LC_H3_PIPE(_V2) → LC_AV_PIPE
# or metadata only. "prompt" is the H3 pipe's single-prompt field name.
SHARED_FROM_LC_PIPE = (
    "width",
    "height",
    "positive_prompt",
    "negative_prompt",
    "prompt",
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

# Comma-joined "T1,T2,..." is what ComfyUI's socket-type checking actually
# understands at both the frontend drag-connect layer and the backend
# validator -- a str subclass with a __ne__ override (the previous approach
# here) only ever affects the backend check, so it never let these sockets
# actually connect to anything but their own primary type live in the UI.
# Confirmed live this session on LC MiniMax H3 Pipe; same fix applied here.
pipe_in = f"{LC_AV_PIPE},{LC_PIPE},{LC_H3_PIPE},{LC_H3_PIPE_V2}"
meta_in = f"{LC_SAVE_META},{LC_SAVE_VIDEO_META}"


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
