"""
LC Get Latent Size
------------------
Pixel width/height the rest of the graph uses (latent × compression).
VAE wired → that model's compression; else the compression widget (8).
"""

from __future__ import annotations


def _tensor(samples):
    if samples is None:
        return None
    if isinstance(samples, dict):
        return samples.get("samples")
    return samples


def _dims(t):
    """Return batch, frames, latent_h, latent_w."""
    if t is None:
        return 0, 1, 0, 0
    shape = tuple(int(x) for x in t.shape)
    if len(shape) == 5:
        b, _c, fr, h, w = shape
        return b, fr, h, w
    if len(shape) == 4:
        b, _c, h, w = shape
        return b, 1, h, w
    if len(shape) == 3:
        _c, h, w = shape
        return 1, 1, h, w
    return 0, 1, 0, 0


def _compression(vae, fallback):
    if vae is not None:
        for name in ("spacial_compression_decode", "spacial_compression"):
            fn = getattr(vae, name, None)
            if fn is None:
                continue
            try:
                val = fn() if callable(fn) else fn
                if val:
                    return max(1, int(val))
            except Exception:
                pass
    return max(1, int(fallback or 8))


class LCGetLatentSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
            },
            "optional": {
                "vae": (
                    "VAE",
                    {"tooltip": "If wired, pixel size uses this VAE's spatial compression."},
                ),
                "compression": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Used when no VAE is wired. SD/SDXL/Flux/H3 is usually 8.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "batch", "frames")
    FUNCTION = "measure"
    CATEGORY = "LC AV/io"
    DESCRIPTION = (
        "Pixel width and height (what Empty Latent / Resize / Save use). "
        "batch and frames for video latents."
    )

    def measure(self, samples, vae=None, compression=8):
        b, fr, lh, lw = _dims(_tensor(samples))
        scale = _compression(vae, compression)
        return (int(lw * scale), int(lh * scale), int(b), int(fr))


NODE_CLASS_MAPPINGS = {
    "LCGetLatentSize": LCGetLatentSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCGetLatentSize": "LC Get Latent Size 📐",
}
