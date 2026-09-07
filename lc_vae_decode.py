"""
LC VAE Decode
-------------
Wrapper around core VAE Decode and VAE Decode (Tiled).
Same call as running those two nodes and picking one with a boolean.
"""

from __future__ import annotations


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
