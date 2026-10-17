"""
LC Frame Interpolate
--------------------
Smooth motion: makes in-between frames with RIFE so a 24 fps clip plays at 48, 60, or any rate you pick (not just 2x).

- Any target fps: each new frame is placed at its real moment in time, so 24 -> 30 or 24 -> 60 stays in sync.
- Scene cuts: a hard cut between two frames is held, not morphed (a morph across a cut is a smeared ghost frame).
- Audio match: wire the clip's audio and the output is exactly as long as the sound, so picture and audio end together.
- Model: RIFE 4.26 by default (the RIFE author's pick for AI video). crossfade needs no model, a quick fallback.

RIFE architecture vendored from ComfyUI-Frame-Interpolation (Fannovel16, MIT), based on Practical-RIFE (hzwer, MIT).
Weights load from models/rife/ or that pack's ckpts/rife/; a missing rife426.pth downloads once (21 MB).
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

import comfy.model_management as mm
import comfy.utils
import folder_paths

VERSIONS = {"rife426.pth": "4.26", "rife417.pth": "4.17", "rife49.pth": "4.7", "rife47.pth": "4.7"}
DEFAULT = "rife426.pth"
CROSSFADE = "crossfade (no model)"
URL = "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/"
_cache = {}


def _dirs():
    out = [os.path.join(folder_paths.models_dir, "rife")]
    out.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "ComfyUI-Frame-Interpolation", "ckpts", "rife"))
    return out


def _find(name):
    for d in _dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def model_choices():
    have = [n for n in VERSIONS if _find(n)]
    return ([DEFAULT] if DEFAULT not in have else []) + have + [CROSSFADE]


def _load(name):
    if name in _cache:
        return _cache[name]
    path = _find(name)
    if path is None:
        path = os.path.join(_dirs()[0], name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"[LC_AV] Frame Interpolate: downloading {name} to {path} (one time)")
        torch.hub.download_url_to_file(URL + name, path)
    from .vendor.rife.rife_arch import IFNet
    net = IFNet(arch_ver=VERSIONS[name])
    sd = torch.load(path, map_location="cpu", weights_only=False)
    # some checkpoints still carry their training-only parts (the distillation teacher, the time encoder)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items() if not k.startswith(("teacher.", "caltime."))}
    net.load_state_dict(sd)
    net.eval()
    _cache.clear()
    _cache[name] = net
    return net


def _thumb(frame):
    """[H,W,C] -> small grey [1,1,h,w] for cut detection."""
    x = frame[..., :3].float().mean(-1)[None, None]
    return F.interpolate(x, size=(64, 64), mode="area")


def plan(n, src_fps, dst_fps, out_count):
    """For each output frame: (i, t) = between source frame i and i+1, t of the way (0 = exactly frame i)."""
    ratio = src_fps / dst_fps
    out = []
    for k in range(out_count):
        pos = min(k * ratio, n - 1)
        i = min(int(math.floor(pos + 1e-6)), n - 1)
        t = pos - i
        out.append((i, 0.0 if t < 1e-3 or i >= n - 1 else t))
    return out


class LCFrameInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The clip's frames."}),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                         "tooltip": "The clip's frame rate now (H3 and most video models: 24)."}),
                "target_fps": ("FLOAT", {"default": 48.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                         "tooltip": "The frame rate you want. Any number: 48 and 60 are smooth, "
                                                    "30 matches phone video."}),
                "model": (model_choices(), {"default": DEFAULT,
                                            "tooltip": "rife426.pth (RIFE 4.26) is the best for AI video and downloads "
                                                       "once (21 MB) if missing. crossfade blends neighbors, no model: "
                                                       "fast but soft on motion."}),
                "scene_cut": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "How different two frames must be to count as a cut (held, not "
                                                   "morphed). 0.12 catches hard cuts and leaves fast motion alone. "
                                                   "0 = off."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "The clip's sound. The output is made exactly as long as the audio, "
                                               "so picture and sound end together."}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32,
                                       "tooltip": "New frames made per GPU pass. Higher is faster and uses more VRAM."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "fps", "info")
    OUTPUT_TOOLTIPS = ("The smooth clip.", "The new frame rate, for the save node.", "What was done.")
    FUNCTION = "run"
    CATEGORY = "LC_AV/video"
    DESCRIPTION = ("Smooth motion with RIFE 4.26: any target fps (not just 2x), scene cuts held instead of morphed, "
                   "and the length matched to the audio so picture and sound end together.")

    def run(self, images, source_fps, target_fps, model, scene_cut, audio=None, batch_size=4):
        n, h, w, c = images.shape
        if audio is not None:
            dur = audio["waveform"].shape[-1] / float(audio["sample_rate"])
        else:
            dur = n / float(source_fps)
        out_count = max(1, int(round(dur * target_fps)))
        steps = plan(n, float(source_fps), float(target_fps), out_count)

        cuts = set()
        if scene_cut > 0 and n > 1:
            th = [_thumb(images[i]) for i in range(n)]
            for i in range(n - 1):
                if float((th[i] - th[i + 1]).abs().mean()) > scene_cut:
                    cuts.add(i)

        out = torch.empty((out_count, h, w, 3), dtype=images.dtype)
        jobs = []
        for k, (i, t) in enumerate(steps):
            if t == 0.0:
                out[k] = images[i, ..., :3]
            elif i in cuts:
                out[k] = images[i if t < 0.5 else i + 1, ..., :3]
            else:
                jobs.append((k, i, t))

        # a progress bar like a sampler's: green on the node, a counter in the console
        pbar = comfy.utils.ProgressBar(max(1, len(jobs)))
        bar = tqdm(total=len(jobs), desc="LC Frame Interpolate", unit="frame")

        def step(k):
            pbar.update(k)
            bar.update(k)

        if model == CROSSFADE or not jobs:
            for k, i, t in jobs:
                mm.throw_exception_if_processing_interrupted()
                out[k] = torch.lerp(images[i, ..., :3].float(), images[i + 1, ..., :3].float(), t).to(out.dtype)
                step(1)
        else:
            dev = mm.get_torch_device()
            dtype = torch.float16 if dev.type == "cuda" else torch.float32
            net = _load(model).to(dev, dtype)
            ver = VERSIONS[model]
            sf = 1.0 if max(h, w) <= 2048 else 0.5  # RIFE's own advice: scale 0.5 for 4K
            scales = [16 / sf, 8 / sf, 4 / sf, 2 / sf, 1 / sf] if ver == "4.26" else [8 / sf, 4 / sf, 2 / sf, 1 / sf]
            # RIFE shrinks by the first scale, then twice more: the size must divide by that (2112 x 1216 broke 4.26)
            m = int(scales[0] * 16)
            ph, pw = (m - h % m) % m, (m - w % m) % m
            pad = lambda x: torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate") if ph or pw else x  # noqa: E731
            try:
                with torch.inference_mode():
                    for s in range(0, len(jobs), batch_size):
                        mm.throw_exception_if_processing_interrupted()
                        part = jobs[s:s + batch_size]
                        a = pad(torch.stack([images[i, ..., :3] for _, i, _ in part]).permute(0, 3, 1, 2).to(dev, dtype))
                        b = pad(torch.stack([images[i + 1, ..., :3] for _, i, _ in part]).permute(0, 3, 1, 2).to(dev, dtype))
                        ts = torch.tensor([t for _, _, t in part], device=dev, dtype=dtype).view(-1, 1, 1, 1)
                        mid = net(a, b, ts, scales, training=False, fastmode=True, ensemble=False)[:, :, :h, :w]
                        mid = mid.clamp(0, 1).permute(0, 2, 3, 1).to("cpu", out.dtype)
                        for j, (k, _, _) in enumerate(part):
                            out[k] = mid[j]
                        step(len(part))
            finally:
                net.to("cpu")
                mm.soft_empty_cache()
        bar.close()

        info = (f"{n} frames at {source_fps:g} fps -> {out_count} frames at {target_fps:g} fps "
                f"({out_count / target_fps:.3f} s{', matched to the audio' if audio is not None else ''}). "
                f"{len(jobs)} new frames ({model}), {len(cuts)} scene cut{'s' if len(cuts) != 1 else ''} held.")
        print(f"[LC_AV] Frame Interpolate: {info}")
        return (out, float(target_fps), info)


NODE_CLASS_MAPPINGS = {"LCFrameInterpolate": LCFrameInterpolate}
NODE_DISPLAY_NAME_MAPPINGS = {"LCFrameInterpolate": "LC Frame Interpolate 🎞️"}
