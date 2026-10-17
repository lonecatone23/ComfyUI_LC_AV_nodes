"""
LC H3 Upscale: upscales the segments of an H3 long-video project after you are happy with the cut. One run = one
segment: it takes that segment's saved latent (LC H3 Segment Save keeps it), runs it through the MiniMax H3 latent
upscaler and one light H3 pass over tiles, and writes seg_00N_up.mp4 next to the normal one. LC H3 Assemble with
version "upscaled" then joins the upscaled set.

Render and fix the movie at the normal size first (fast), then upscale once. Skipping it changes nothing.

The upscale engine is vendored from bbaudio-2025/Comfyui-MMH3-UltimateUpscale (MIT, vendor/mmh3_upscale).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import folder_paths
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

from .lc_h3_segment import FPS, PLAN_TYPE, _encode, _is_done, _read_audio, _seg_path, project_dir
from .vendor.mmh3_upscale.nodes import MMH3LatentUpscaleWithModelParams, MMH3SpatialSplitParams, MMH3UltimateUpscale

SCALES = {"2x": 2.0, "1.5x": 1.5}
DEFAULT_MODEL = "minimax_h3_latent_upscaler_3d_fp16.safetensors"


def _models():
    try:
        names = folder_paths.get_filename_list("latent_upscale_models")
    except Exception:
        names = []
    names = [n for n in names if "minimax" in n.lower() or "h3" in n.lower()] or names
    return names or [DEFAULT_MODEL]


def _out(node_output):
    """io.NodeOutput -> its values."""
    for attr in ("args", "result"):
        v = getattr(node_output, attr, None)
        if v is not None:
            return v
    return node_output


def _next(folder, count, index):
    """The segment to upscale: index, or the first one that is saved, has its latent and no upscale yet."""
    if index > 0:
        return min(index, count)
    for i in range(1, count + 1):
        if not _is_done(folder, i):
            break
        if os.path.exists(_seg_path(folder, i, "_latent.safetensors")) and not os.path.exists(_seg_path(folder, i, "_up.mp4")):
            return i
    return None


class LCH3Upscale:
    @classmethod
    def INPUT_TYPES(cls):
        models = _models()
        return {
            "required": {
                "plan": (PLAN_TYPE, {"tooltip": "From LC Vision Director 🎬 (a saved plan loads instantly)."}),
                "model": ("MODEL", {"tooltip": "The same H3 model (and turbo LoRA) you rendered with."}),
                "clip": ("CLIP", {"tooltip": "MiniMax H3 text encoder."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 60, "tooltip": "Which segment to upscale. 0 = the next one without an upscale. Set the run count to the number of segments and walk away."}),
                "upscale_by": (list(SCALES), {"default": "2x", "tooltip": "2x: 960 x 544 becomes 1920 x 1088."}),
                "steps": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "Steps of the light H3 pass after the latent upscaler. 1 is what tested best."}),
                "denoise": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.6, "step": 0.01, "tooltip": "How much the light pass redraws. 0.2 adds real detail and keeps the faces; higher starts to change them."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "brightness_match": ("BOOLEAN", {"default": True, "tooltip": "Keeps every tile and frame at the original brightness, so the upscale doesn't flicker or darken."}),
                "upscale_model": (models, {"default": DEFAULT_MODEL if DEFAULT_MODEL in models else models[0],
                                           "tooltip": "The MiniMax H3 latent upscaler, in models/latent_upscale_models. LC Model Optimizer Video can download it."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "info")
    FUNCTION = "upscale"
    OUTPUT_NODE = True
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = ("Upscales one segment per run from its saved latent (MiniMax H3 latent upscaler + a light H3 pass over "
                   "tiles) into seg_00N_up.mp4. Do it once the cut is final; LC H3 Assemble with version 'upscaled' joins them.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def upscale(self, plan, model, clip, vae, index, upscale_by, steps, denoise, seed, brightness_match, upscale_model):
        folder = project_dir(plan["project"])
        shots = plan.get("shots") or []
        i = _next(folder, len(shots), index)
        if i is None:
            done = sum(1 for k in range(1, len(shots) + 1) if os.path.exists(_seg_path(folder, k, "_up.mp4")))
            msg = (f"Every saved segment is upscaled ({done} of {len(shots)}). Run LC H3 Assemble with version 'upscaled'."
                   if done else "No segment has a saved latent yet. Segments saved before this update need rendering again.")
            print(f"[LC H3] {msg}")
            return {"ui": {"text": [msg]}, "result": (torch.zeros(1, 64, 64, 3), msg)}
        lat_path = _seg_path(folder, i, "_latent.safetensors")
        if not os.path.exists(lat_path):
            raise ValueError(f"[LC H3] Segment {i} has no saved latent (saved before LC H3 Upscale existed). Render it again with index {i}.")
        with open(_seg_path(folder, i, ".json"), encoding="utf-8") as f:
            meta = json.load(f)
        sd = comfy.utils.load_torch_file(lat_path, safe_load=True)
        latent = {"samples": comfy.nested_tensor.NestedTensor((sd["video"], sd["audio"]))}
        s = SCALES[upscale_by]
        w = int(round(meta["width"] * s / 32)) * 32
        h = int(round(meta["height"] * s / 32)) * 32
        # the shot's own words, without the memory references (the upscale has no reference video)
        shot = shots[i - 1]
        prompt = shot.get("prompt") or f"{plan.get('style', '')}\n\n{shot.get('action', '')}"
        positive = _out(MiniMaxH3ImageToVideo.execute(clip, vae, prompt, w, h, int(meta.get("frames", 124))))[0]
        up_param = _out(MMH3LatentUpscaleWithModelParams.execute(upscale_model, w, h, "cuda", "fp16", True))[0]
        cols, rows = (3, 2) if w >= 1600 else (2, 2)
        split = _out(MMH3SpatialSplitParams.execute(w, h, "rows_cols", 512, 512, rows, cols, 128, 128, 64, 64, 256,
                                                    "later", "linear", 0.05, bool(brightness_match), "off", 32))[0]
        total = max(steps, int(round(steps / denoise)))
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", total).cpu()[-(steps + 1):]
        out = _out(MMH3UltimateUpscale.execute(latent=latent, conditioning=positive, model=model, noise=Noise_RandomNoise(seed),
                                                sampler=comfy.samplers.sampler_object("res_multistep"), sigmas=sigmas, cfg=1.0,
                                                latent_upscale_param=up_param, spatial_split_param=split))[0]
        video = out["samples"].unbind()[0]
        images = vae.decode(video)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        lead = int(meta.get("lead_frames", 0))
        frames = (images[lead:].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        # the same frames as the normal segment, so Assemble's overlaps and the sound line up
        n = frames.shape[0]
        aud = _read_audio(_seg_path(folder, i, ".mp4"))
        audio = {"waveform": torch.from_numpy(np.ascontiguousarray(aud[:, :int(round(n * 48000 / FPS))])).unsqueeze(0), "sample_rate": 48000}
        _encode(frames, _seg_path(folder, i, "_up.mp4"), 16, audio=audio)
        meta["up"] = {"width": int(frames.shape[2]), "height": int(frames.shape[1]), "scale": upscale_by, "denoise": denoise, "steps": steps}
        with open(_seg_path(folder, i, ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)
        saved = [k for k in range(1, len(shots) + 1) if _is_done(folder, k)]
        left = [k for k in saved if not os.path.exists(_seg_path(folder, k, "_up.mp4"))]
        info = f"Upscaled segment {i} of {len(shots)} to {frames.shape[2]} x {frames.shape[1]}. "
        if left:
            info += f"{len(left)} saved segment{'s' if len(left) > 1 else ''} to go."
        elif len(saved) < len(shots):
            info += f"All {len(saved)} saved segments are upscaled; render the other {len(shots) - len(saved)} first, then upscale them too."
        else:
            info += "All done: run LC H3 Assemble with version 'upscaled'."
        print(f"[LC H3] {info}")
        preview = torch.from_numpy(frames[n // 2].astype(np.float32) / 255.0)[None]
        return {"ui": {"text": [info]}, "result": (preview, info)}


NODE_CLASS_MAPPINGS = {"LCH3Upscale": LCH3Upscale}
NODE_DISPLAY_NAME_MAPPINGS = {"LCH3Upscale": "LC H3 Upscale (BETA) 🔭"}
