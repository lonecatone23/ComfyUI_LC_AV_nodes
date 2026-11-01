"""
LC H3 cast and prop cards on plain grey (no node). A card's own background, props and light leak into every shot it
is in, so cast and prop pictures are cut out with BiRefNet (ComfyUI's own background-removal loader, no extra
dependency) and put on flat grey, the background-card method that held faces and rooms in the 2026-11-01 tests.

  POST /lc_h3/story/cutout {project, pic} -> {"pic": {...}}: a picture dropped on a cast or prop card, on grey
LC H3 Save to Story runs the same cut on every new cast and prop picture it makes.
"""
from __future__ import annotations

import os
import re

import folder_paths

GREY = (128, 128, 128)
SUFFIX = "_grey"
MODEL_FILE = "BiRefNet-general.safetensors"
MODEL_REPO = "ZhengPeng7/BiRefNet"
_model = {}


def _model_path(cancel: bool) -> str:
    """BiRefNet on disk (any file in models/background_removal, the general one first), else download it once."""
    try:
        dirs = list(folder_paths.get_folder_paths("background_removal"))
    except Exception:
        dirs = []
    dirs = dirs or [os.path.join(folder_paths.models_dir, "background_removal")]
    found = []
    for d in dirs:
        if os.path.isdir(d):
            found += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith((".safetensors", ".pth", ".pt"))]
    pick = next((f for f in found if os.path.basename(f) == MODEL_FILE), None) or next(
        (f for f in found if "birefnet" in os.path.basename(f).lower()), None)
    if pick:
        return pick
    from .lc_first_download import download, hf_url

    return download(hf_url(MODEL_REPO, "model.safetensors"), os.path.join(dirs[0], MODEL_FILE), "BiRefNet background remover",
                    "about 900 MB", cancel=cancel, announce=cancel)


def _load(cancel: bool):
    path = _model_path(cancel)
    if _model.get("path") != path:
        from comfy.bg_removal_model import load

        m = load(path)
        if m is None:
            raise RuntimeError(f"[LC H3] {path} is not a background-removal model ComfyUI can load.")
        m.image_size = 1024
        _model.clear()
        _model.update(path=path, model=m)
    return _model["model"]


def on_grey(img, cancel: bool = True):
    """PIL RGB picture -> the same picture with its background replaced by flat grey."""
    import numpy as np
    import torch
    from PIL import Image

    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(rgb)[None]
    with torch.inference_mode():  # a route runs outside ComfyUI's own inference mode
        mask = _load(cancel).encode_image(t).detach().float().cpu().clamp(0, 1)[0].numpy()  # (H, W)
    if mask.shape != rgb.shape[:2]:
        mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR),
                          dtype=np.float32) / 255.0
    grey = np.array(GREY, dtype=np.float32) / 255.0
    out = rgb * mask[..., None] + grey * (1.0 - mask[..., None])
    return Image.fromarray((out * 255.0).round().clip(0, 255).astype(np.uint8))


def grey_name(filename: str) -> str:
    stem, _ext = os.path.splitext(os.path.basename(filename))
    return stem if stem.endswith(SUFFIX) else stem + SUFFIX


def cut_file(src_abs: str, out_dir: str, cancel: bool = True) -> str:
    """Cut src_abs onto grey as <out_dir>/<name>_grey.png (a new file, never over another one). Returns its path."""
    from PIL import Image, ImageOps

    os.makedirs(out_dir, exist_ok=True)
    base = grey_name(src_abs)
    out = os.path.join(out_dir, base + ".png")
    k = 2
    while os.path.exists(out):
        out = os.path.join(out_dir, f"{base}_{k}.png")
        k += 1
    with Image.open(src_abs) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    on_grey(img, cancel).save(out)
    return out


try:
    import asyncio

    from aiohttp import web
    from server import PromptServer

    from .lc_h3_refs import inside_comfy

    @PromptServer.instance.routes.post("/lc_h3/story/cutout")
    async def _cutout(request):
        try:
            data = await request.json()
            pic = data.get("pic") or {}
            from .lc_h3_story import _pic_abs, _pic_file
            from .lc_h3_story_write import _not_while_rendering

            src = inside_comfy(_pic_abs(pic))
            if not src or not os.path.isfile(src):
                return web.json_response({"error": "That picture isn't in ComfyUI's input or output folder."}, status=400)
            if os.path.splitext(os.path.basename(_pic_file(pic)))[0].endswith(SUFFIX):
                return web.json_response({"pic": pic})  # already on grey
            _not_while_rendering()
            safe = re.sub(r"[^A-Za-z0-9 _.-]", "_", str(data.get("project") or "story").strip())
            safe = safe if safe.strip(" .") else "story"
            sub = os.path.join("lc_h3_refs", safe)
            out_dir = os.path.join(folder_paths.get_input_directory(), sub)
            out = await asyncio.get_running_loop().run_in_executor(None, cut_file, src, out_dir, False)
            return web.json_response({"pic": {"type": "input", "subfolder": sub.replace("\\", "/"), "filename": os.path.basename(out)}})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)
except Exception as _e:  # pragma: no cover - outside ComfyUI
    print(f"[LC H3] cutout route not registered: {_e}")
