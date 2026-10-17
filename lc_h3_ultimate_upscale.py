"""
LC H3 Ultimate Upscale (BETA)
-----------------------------
The MMH3 Ultimate Upscale chain for a normal (single clip) MiniMax H3 workflow, in one node: it reads the models, the
prompt, the references and the sampler from the LC MiniMax H3 pipe, re-encodes the prompt at the upscaled size,
runs the MiniMax H3 latent upscaler, then one light H3 pass over tiles. Latent in, upscaled latent out (decode it
with your usual VAE decode). Same engine as LC H3 Upscale (vendored bbaudio-2025/Comfyui-MMH3-UltimateUpscale, MIT).
"""
from __future__ import annotations

import comfy.samplers
import comfy.utils
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo

from .lc_h3_upscale import DEFAULT_MODEL, _models, _out
from .vendor.mmh3_upscale import nodes as mmh3
from .vendor.mmh3_upscale.nodes import MMH3LatentUpscaleWithModelParams, MMH3SpatialSplitParams, MMH3UltimateUpscale

PIPE = "LC_H3_PIPE_V2"
FROM_PIPE = "from pipe"


class LCH3UltimateUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        models = _models()
        return {
            "required": {
                "pipe": (PIPE, {"tooltip": "The LC MiniMax H3 pipe (V2): models, prompt, references, sampler, seed."}),
                "latent": ("LATENT", {"tooltip": "The finished H3 latent from the main sampler."}),
                "ref2va": ("BOOLEAN", {"default": False, "label_on": "REF2VA", "label_off": "FL2VA",
                                       "tooltip": "Which model rendered the clip. Wire your REF2VA / FL2VA switch here."}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.25,
                                         "tooltip": "2 = twice the width and height (rounded to 32)."}),
                "steps": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "Steps of the light H3 pass. 1 tested best."}),
                "denoise": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.6, "step": 0.01,
                                      "tooltip": "How much the light pass redraws. 0.2 adds detail and keeps faces; higher changes them."}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64,
                                      "tooltip": "Tile width and height for the light pass. Smaller = less VRAM, slower."}),
                "sampler": ([FROM_PIPE] + list(comfy.samplers.KSampler.SAMPLERS), {"default": FROM_PIPE,
                            "tooltip": "from pipe = the sampler your main render used."}),
                "brightness_match": ("BOOLEAN", {"default": True, "tooltip": "Keeps every tile and frame at the original brightness (no flicker)."}),
                "upscale_model": (models, {"default": DEFAULT_MODEL if DEFAULT_MODEL in models else models[0],
                                           "tooltip": "The MiniMax H3 latent upscaler (models/latent_upscale_models)."}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    OUTPUT_TOOLTIPS = ("The upscaled H3 latent: decode it like the normal one.", "Upscaled width.", "Upscaled height.")
    FUNCTION = "upscale"
    CATEGORY = "LC AV/H3"
    DESCRIPTION = ("MMH3 Ultimate Upscale in one node for a normal MiniMax H3 workflow: everything comes from the LC H3 pipe. "
                   "MiniMax H3 latent upscaler, then one light H3 pass over tiles. 2x with 1 step at denoise 0.2 is the tested default.")

    def upscale(self, pipe, latent, ref2va, upscale_by, steps, denoise, tile_size, sampler, brightness_match, upscale_model):
        p = pipe if isinstance(pipe, dict) else {}
        model = p.get("ref2va_model" if ref2va else "fl2va_model")
        clip = p.get("ref2va_clip" if ref2va else "fl2va_clip")
        vae, audio_vae = p.get("video_vae"), p.get("audio_vae")
        if model is None or clip is None or vae is None:
            raise ValueError(f"[LC H3 Ultimate Upscale] The pipe has no {'REF2VA' if ref2va else 'FL2VA'} model, clip or video VAE. "
                             "Check the REF2VA / FL2VA switch matches the model you rendered with.")
        w = int(round(int(p.get("width") or 1344) * upscale_by / 32)) * 32
        h = int(round(int(p.get("height") or 768) * upscale_by / 32)) * 32
        length = int(p.get("length") or 124)
        prompt = p.get("prompt") or ""
        if ref2va:
            pick = lambda prefix, n: {f"{prefix}_{i}": p.get(f"{prefix}_{i}") for i in range(n) if p.get(f"{prefix}_{i}") is not None}
            positive = _out(MiniMaxH3ReferenceToVideo.execute(
                clip, prompt, w, h, length, "match", vae=vae, audio_vae=audio_vae,
                ref_images=pick("ref_image", 9), ref_videos=pick("ref_video", 3),
                ref_video_audios=pick("ref_video_audio", 3), ref_audios=pick("ref_audio", 3)))[0]
        else:
            positive = _out(MiniMaxH3ImageToVideo.execute(clip, vae, prompt, w, h, length,
                                                          first_frame=p.get("ref_image_0"), last_frame=p.get("ref_image_1")))[0]
        up_param = _out(MMH3LatentUpscaleWithModelParams.execute(upscale_model, w, h, "cuda", "fp16", True))[0]
        cols = max(1, round(w / tile_size))
        rows = max(1, round(h / tile_size))
        split = _out(MMH3SpatialSplitParams.execute(w, h, "rows_cols", tile_size, tile_size, rows, cols, 128, 128, 64, 64, 256,
                                                    "later", "linear", 0.05, bool(brightness_match), "off", 32))[0]
        total = max(steps, int(round(steps / denoise)))
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), p.get("scheduler") or "simple", total).cpu()[-(steps + 1):]
        name = p.get("sampler_name") or "euler" if sampler == FROM_PIPE else sampler
        # one bar for the whole pass (tile 3 of 8), not a 0/1 bar per tile
        mmh3.OUTER = {"bar": comfy.utils.ProgressBar(rows * cols * steps), "done": 0, "total": rows * cols}
        try:
            out = _out(MMH3UltimateUpscale.execute(latent=latent, conditioning=positive, model=model,
                                                    noise=Noise_RandomNoise(int(p.get("seed") or 0)),
                                                    sampler=comfy.samplers.sampler_object(name), sigmas=sigmas, cfg=1.0,
                                                    latent_upscale_param=up_param, spatial_split_param=split))[0]
        finally:
            mmh3.OUTER = None
        print(f"[LC H3 Ultimate Upscale] {'REF2VA' if ref2va else 'FL2VA'} {w} x {h}, {rows} x {cols} tiles, {steps} step(s) at {denoise}, {name}")
        return (out, w, h)


NODE_CLASS_MAPPINGS = {"LCH3UltimateUpscale": LCH3UltimateUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"LCH3UltimateUpscale": "LC H3 Ultimate Upscale (BETA) 🔭"}
