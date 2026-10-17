"""
LC VAE Decode
-------------
Wrapper around core VAE Decode and VAE Decode (Tiled).
Same call as running those two nodes and picking one with a boolean.
"""

from __future__ import annotations

import contextlib


def _core_classes():
    decode_cls = None
    tiled_cls = None
    try:
        import nodes as _nodes

        decode_cls = getattr(_nodes, "VAEDecode", None)
        tiled_cls = getattr(_nodes, "VAEDecodeTiled", None)
    except Exception:
        pass
    return decode_cls, tiled_cls


@contextlib.contextmanager
def _tile_progress():
    """Core's video (3D) tiled decode has no progress bar and can't be cancelled until it ends: for this call, every
    tile ticks a bar (counted on core's own tile grid) and checks Cancel. 2D decodes already bring their own bar."""
    import itertools

    import comfy.model_management
    import comfy.utils

    orig = comfy.utils.tiled_scale_multidim

    def wrapped(samples, function, tile=(64, 64), overlap=8, *args, pbar=None, **kw):
        if pbar is None:
            dims = len(tile)
            ov = overlap if isinstance(overlap, (tuple, list)) else [overlap] * dims
            per = 1
            if not all(samples.shape[d + 2] <= tile[d] for d in range(dims)):
                pos = [range(0, samples.shape[d + 2] - ov[d], tile[d] - ov[d]) if samples.shape[d + 2] > tile[d] else [0]
                       for d in range(dims)]
                per = len(list(itertools.product(*pos)))
            pbar = comfy.utils.ProgressBar(samples.shape[0] * per)

        def tile_fn(x):
            comfy.model_management.throw_exception_if_processing_interrupted()
            return function(x)

        return orig(samples, tile_fn, tile, overlap, *args, pbar=pbar, **kw)

    comfy.utils.tiled_scale_multidim = wrapped
    try:
        yield
    finally:
        comfy.utils.tiled_scale_multidim = orig


@contextlib.contextmanager
def _self_tiling_progress(vae, samples):
    """VAEs that tile themselves (MiniMax H3's video VAE: temporal chunks x spatial tiles, whatever the tiled switch says)
    skip core's tile loop: count their tiles from their own plan, tick a bar per tile and check Cancel."""
    import comfy.model_management
    import comfy.utils

    m = getattr(vae, "first_stage_model", None)
    lat = samples.get("samples") if isinstance(samples, dict) else None
    if m is None or lat is None or lat.ndim != 5 or not all(hasattr(m, a) for a in ("_decode_pixels", "split_tiles", "_decode_temporal_chunks", "vae_ratio")):
        yield
        return
    try:
        b, _, t, h, w = lat.shape
        chunks = 1 if t == 1 else m._decode_temporal_chunks(t)[1]
        rows = len(m.split_tiles(h * m.vae_ratio)[0]) if getattr(m, "tiling", True) else 1
        cols = len(m.split_tiles(w * m.vae_ratio)[0]) if getattr(m, "tiling", True) else 1
        total = max(1, chunks * rows * cols)
    except Exception:
        yield
        return
    pbar = comfy.utils.ProgressBar(total)
    orig = m._decode_pixels

    def counted(z):
        comfy.model_management.throw_exception_if_processing_interrupted()
        out = orig(z)
        pbar.update(max(1, z.shape[0] // max(1, b)))  # a decoder call can carry a few tiles
        return out

    m._decode_pixels = counted
    try:
        yield
    finally:
        m._decode_pixels = orig


def _call_decode(cls, **kwargs):
    obj = cls()
    if hasattr(obj, "decode"):
        return obj.decode(**kwargs)
    if hasattr(obj, "execute"):
        return obj.execute(**kwargs)
    raise RuntimeError(f"LC VAE Decode: {cls.__name__} has no decode/execute")


class LCVaeDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tiled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "tiled",
                        "label_off": "full",
                        "tooltip": "Off = core VAE Decode. On = core VAE Decode (Tiled).",
                    },
                ),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 64, "max": 4096, "step": 32},
                ),
                "overlap": (
                    "INT",
                    {"default": 64, "min": 0, "max": 4096, "step": 32},
                ),
                "temporal_size": (
                    "INT",
                    {"default": 64, "min": 8, "max": 4096, "step": 4},
                ),
                "temporal_overlap": (
                    "INT",
                    {"default": 8, "min": 4, "max": 4096, "step": 4},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "LC AV/io"
    DESCRIPTION = (
        "Boolean wrapper: full = nodes.VAEDecode, tiled = nodes.VAEDecodeTiled."
    )

    def decode(
        self,
        samples,
        vae,
        tiled=False,
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
        **_kwargs,
    ):
        decode_cls, tiled_cls = _core_classes()
        if decode_cls is None:
            raise RuntimeError("LC VAE Decode: could not import nodes.VAEDecode")

        if tiled:
            if tiled_cls is None:
                raise RuntimeError("LC VAE Decode: could not import nodes.VAEDecodeTiled")
            with _tile_progress(), _self_tiling_progress(vae, samples):
                out = _call_decode(
                    tiled_cls,
                    vae=vae,
                    samples=samples,
                    tile_size=int(tile_size),
                    overlap=int(overlap),
                    temporal_size=int(temporal_size),
                    temporal_overlap=int(temporal_overlap),
                )
        else:
            with _self_tiling_progress(vae, samples):
                out = _call_decode(decode_cls, vae=vae, samples=samples)

        if isinstance(out, dict) and "result" in out:
            out = out["result"]
        if isinstance(out, (tuple, list)):
            return (out[0],)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "LCVaeDecode": LCVaeDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCVaeDecode": "LC VAE Decode 🧩",
}
