"""
LC Foley
--------
Sound effects for a video, in sync with what happens on screen, or from a text prompt alone. MMAudio (Ho Kei Cheng
et al., Sony, CVPR 2025) through the inference code vendored in vendor/mmaudio: nothing to pip install and no other
node pack needed. Uses the same model files as kijai's ComfyUI-MMAudio (models/mmaudio), so existing downloads are
reused; on a fresh install they download on first use (about 5.6 GB in all).

The MMAudio weights are CC BY-NC 4.0 (non-commercial). They are never bundled with this pack, only downloaded.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import time

# file -> (Hugging Face repo, size for the first-use toast)
_HF_MAIN = "Kijai/MMAudio_safetensors"
_VAE = "mmaudio_vae_44k_fp16.safetensors"
_SYNC = "mmaudio_synchformer_fp16.safetensors"
_CLIP = "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"
_DEFAULT_MODEL = "mmaudio_large_44k_v2_fp16.safetensors"
_SIZES = {_DEFAULT_MODEL: "2.1 GB", _VAE: "611 MB", _SYNC: "475 MB", _CLIP: "2.0 GB"}
_BIGVGAN_REPO = "nvidia/bigvgan_v2_44khz_128band_512x"
_BIGVGAN_FILES = (("config.json", ""), ("bigvgan_generator.pt", "489 MB"))  # weights + config only, no .py files

_SR = 44100
_CLIP_FPS, _SYNC_FPS = 8.0, 25.0
_ALL_AT_ONCE_BYTES = int(7.5 * 1024 ** 3)  # measured peak with all four models on the GPU: 6.0 GB (5 s) to 6.8 GB (10 s)
_CLIP_SIZE, _SYNC_SIZE = 384, 224
_CLIP_CFG = {"embed_dim": 1024, "quick_gelu": True,
             "vision_cfg": {"image_size": 378, "layers": 32, "width": 1280, "head_width": 80, "patch_size": 14},
             "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 1024, "heads": 16, "layers": 24}}

_cache = {}


# ---------------------------------------------------------------- files

def _dirs():
    import folder_paths

    out = []
    try:
        out += folder_paths.get_folder_paths("mmaudio")  # kijai's pack registers this name
    except Exception:
        pass
    base = os.path.join(folder_paths.models_dir, "mmaudio")
    if os.path.normcase(os.path.abspath(base)) not in [os.path.normcase(os.path.abspath(d)) for d in out]:
        out.append(base)
    return out


def _find(name):
    for d in _dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def _model_names():
    names = set()
    for d in _dirs():
        if os.path.isdir(d):
            for n in os.listdir(d):
                low = n.lower()
                if (low.endswith(".safetensors") and low.startswith("mmaudio") and "44k" in low
                        and "vae" not in low and "synchformer" not in low):
                    names.add(n)
    names.discard(_DEFAULT_MODEL)
    return [_DEFAULT_MODEL] + sorted(names)


def _fetch(name):
    """models/mmaudio/<name>, downloaded from Kijai/MMAudio_safetensors on first use."""
    p = _find(name)
    if p:
        return p
    if name not in _SIZES:
        raise FileNotFoundError(f"LC Foley: {name} is not in models/mmaudio.")
    from .lc_first_download import download, hf_url

    return download(hf_url(_HF_MAIN, name), os.path.join(_dirs()[0], name), f"MMAudio {name}", _SIZES[name])


def _fetch_bigvgan():
    rel = os.path.join("nvidia", "bigvgan_v2_44khz_128band_512x")
    for d in _dirs():
        folder = os.path.join(d, rel)
        if all(os.path.isfile(os.path.join(folder, f)) for f, _ in _BIGVGAN_FILES):
            return folder
    from .lc_first_download import download, hf_url

    folder = os.path.join(_dirs()[0], rel)
    for f, size in _BIGVGAN_FILES:
        dest = os.path.join(folder, f)
        if not os.path.isfile(dest):
            download(hf_url(_BIGVGAN_REPO, f), dest, f"BigVGAN v2 vocoder {f}", size, announce=bool(size))
    return folder


# ---------------------------------------------------------------- loading

@contextlib.contextmanager
def _empty_params():
    """Build modules with their parameters on the meta device (no memory, no copy); buffers stay real.
    The same thing accelerate's init_empty_weights() does by default."""
    import torch.nn as nn

    old = nn.Module.register_parameter

    def register(self, name, param):
        old(self, name, param)
        if param is not None:
            p = self._parameters[name]
            self._parameters[name] = nn.Parameter(p.to("meta"), requires_grad=p.requires_grad)

    nn.Module.register_parameter = register
    try:
        yield
    finally:
        nn.Module.register_parameter = old


def _set_params(model, sd, device, dtype):
    """Every parameter of model from the state dict, at dtype on device (buffers are left alone)."""
    import torch.nn as nn

    names = [n for n, _ in model.named_parameters()]
    missing = [n for n in names if n not in sd]
    if missing:
        raise RuntimeError(f"LC Foley: the weights file lacks {len(missing)} tensors (first: {missing[0]}).")
    for n in names:
        mod_name, _, attr = n.rpartition(".")
        mod = model.get_submodule(mod_name) if mod_name else model
        old = mod._parameters[attr]
        mod._parameters[attr] = nn.Parameter(sd[n].to(dtype).to(device), requires_grad=old.requires_grad)


def _load_file(path):
    import comfy.utils

    import torch

    return comfy.utils.load_torch_file(path, device=torch.device("cpu"), safe_load=True)


def _load_net(model_name, dtype):
    from .vendor.mmaudio.model.networks import MMAudio

    sd = _load_file(_fetch(model_name))
    width = sd["audio_input_proj.0.bias"].shape[0]
    if width == 448:  # small
        heads = 7
        kw = dict(depth=12, fused_depth=8)
    elif width == 896:  # large
        heads = 14
        kw = dict(depth=21, fused_depth=14, v2=sd["t_embed.mlp.0.weight"].shape[1] == 896)
    else:
        raise RuntimeError(f"LC Foley: {model_name} is not an MMAudio model this node knows.")
    with _empty_params():
        net = MMAudio(latent_dim=40, clip_dim=1024, sync_dim=768, text_dim=1024, hidden_dim=64 * heads,
                      num_heads=heads, latent_seq_len=345, clip_seq_len=64, sync_seq_len=192, **kw)
    net = net.eval()
    _set_params(net, sd, "cpu", dtype)
    del sd
    return net


def _load_features(device, dtype):
    from .vendor.mmaudio.clip.model import CLIP
    from .vendor.mmaudio.ext.autoencoder import AutoEncoderModule
    from .vendor.mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN
    from .vendor.mmaudio.ext.synchformer import Synchformer
    from .vendor.mmaudio.model.utils.features_utils import FeaturesUtils

    sd = _load_file(_fetch(_SYNC))
    with _empty_params():
        sync = Synchformer().eval()
    _set_params(sync, sd, device, dtype)

    vocoder = BigVGAN.from_folder(_fetch_bigvgan()).eval().to(device=device, dtype=dtype)
    vae = AutoEncoderModule(vae_state_dict=_load_file(_fetch(_VAE)), bigvgan_vocoder=vocoder, mode="44k")
    vae = vae.eval().to(device=device, dtype=dtype)

    sd = _load_file(_fetch(_CLIP))
    with _empty_params():
        clip = CLIP(**_CLIP_CFG).eval()
    _set_params(clip, sd, device, dtype)
    clip.to(device=device, dtype=dtype)
    del sd
    return FeaturesUtils(vae=vae, synchformer=sync, enable_conditions=True, clip_model=clip)


def load(model_name=_DEFAULT_MODEL, device=None, dtype=None):
    """(net, feature_utils), cached. Kept on the CPU between runs."""
    import torch

    import comfy.model_management as mm

    device = device or mm.get_torch_device()
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    key = (model_name, str(dtype))
    if _cache.get("key") != key:
        _cache.clear()
        mm.soft_empty_cache()
        net = _load_net(model_name, dtype)
        fu = _load_features(device, dtype)
        _cache.update(key=key, net=net, fu=fu)
    return _cache["net"], _cache["fu"]


# ---------------------------------------------------------------- video frames

def _transforms():
    from torchvision.transforms import v2

    import torch

    clip_t = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToPILImage(), v2.ToTensor(), v2.ConvertImageDtype(torch.float32),
    ])
    sync_t = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToPILImage(), v2.ToTensor(), v2.ConvertImageDtype(torch.float32),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return clip_t, sync_t


def _pick(n_frames, fps, target_fps, count):
    """Indices of the frames nearest to each tick of target_fps (what ffmpeg's fps filter does)."""
    return [min(n_frames - 1, int(math.floor(k * fps / target_fps + 0.5))) for k in range(count)]


def video_frames(images, fps, duration, frame_rate_aware=True, device=None):
    """IMAGE [N, H, W, 3] -> (clip frames [1, T8, 3, 384, 384], sync frames [1, T25, 3, 224, 224], duration).

    frame_rate_aware=True samples the video at 8 and 25 frames per second using its real fps, the way MMAudio's
    own demo reads a video file. False takes the first frames as they are, the way kijai's MMAudio Sampler does
    (it has no fps input), which only lines up for 25 fps video; kept for the parity test."""
    import torch

    clip_t, sync_t = _transforms()
    n = images.shape[0]
    if frame_rate_aware:
        duration = min(duration, n / fps)
        clip_idx = _pick(n, fps, _CLIP_FPS, int(_CLIP_FPS * duration))
        sync_idx = _pick(n, fps, _SYNC_FPS, int(_SYNC_FPS * duration))
    else:
        if n < int(_CLIP_FPS * duration):
            duration = n / _CLIP_FPS
        if n < int(_SYNC_FPS * duration):
            duration = n / _SYNC_FPS
        clip_idx = list(range(int(_CLIP_FPS * duration)))
        sync_idx = list(range(int(_SYNC_FPS * duration)))
    # one frame on the GPU at a time (a whole 1344 x 768 clip at full size is 1.5 GB)
    clip = torch.stack([clip_t(images[i].to(device).permute(2, 0, 1)) for i in clip_idx])
    sync = torch.stack([sync_t(images[i].to(device).permute(2, 0, 1)) for i in sync_idx])
    return clip.unsqueeze(0), sync.unsqueeze(0), duration


# ---------------------------------------------------------------- generation

def render(prompt, negative_prompt="", images=None, fps=24.0, duration=0.0, steps=25, cfg=4.5, seed=0,
           model_name=_DEFAULT_MODEL, frame_rate_aware=True, callback=None, offload=True, fast_convs=True,
           force_low_vram=None):
    """-> (waveform float32 tensor [1, 1, T] at 44.1 kHz, duration, info dict).

    fast_convs: run the MMAudio network's convolutions without cuDNN. For some lengths (5.1 s and 3.3 s in tests;
    6, 8 and 10 s were fine) cuDNN picks an FFT algorithm that makes each step about 10x slower (0.64 s instead of 0.06 s on
    an RTX 5090); PyTorch's own convolutions are fast at every length. The result differs from the cuDNN path by
    rounding only (about the same as cuDNN differs from itself between runs)."""
    import torch

    import comfy.model_management as mm

    from .vendor.mmaudio.eval_utils import generate
    from .vendor.mmaudio.model.flow_matching import FlowMatching
    from .vendor.mmaudio.model.sequence_config import CONFIG_44K

    device = mm.get_torch_device()
    t0 = time.perf_counter()
    net, fu = load(model_name, device)
    t_load = time.perf_counter() - t0

    clip_frames = sync_frames = None
    if images is not None:
        if duration <= 0:
            duration = images.shape[0] / (fps if frame_rate_aware else _SYNC_FPS)
        clip_frames, sync_frames, duration = video_frames(images, fps, duration, frame_rate_aware, device)
    if duration <= 0:
        raise ValueError("LC Foley: set a duration (seconds) when no video frames are connected.")
    if sync_frames is not None and sync_frames.shape[1] < 16:
        raise ValueError("LC Foley: the video is too short (MMAudio needs at least 0.64 s).")

    seq_cfg = dataclasses.replace(CONFIG_44K, duration=duration)
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=int(steps))

    t1 = time.perf_counter()
    low = False
    if device.type == "cuda":
        try:
            mm.free_memory(8 * 1024 ** 3, device)  # room next to other models: ~5 GB of weights + activations
        except Exception:
            pass
        try:  # all four models at once need about 6-7 GB; less free than that: one model on the GPU at a time
            low = mm.get_free_memory(device) < _ALL_AT_ONCE_BYTES or bool(getattr(mm, "vram_state", None) in (
                getattr(mm.VRAMState, "LOW_VRAM", None), getattr(mm.VRAMState, "NO_VRAM", None)))
        except Exception:
            low = False
    if force_low_vram is not None:
        low = bool(force_low_vram)
    if fast_convs:
        predict_flow = net.predict_flow

        def no_cudnn_flow(*a, **k):
            with torch.backends.cudnn.flags(enabled=False):
                return predict_flow(*a, **k)

        net.predict_flow = no_cudnn_flow  # instance attribute: ode_wrapper calls self.predict_flow
    try:
        if low:
            audio = _generate_low_vram(clip_frames, sync_frames, prompt, negative_prompt, fu, net, fm, rng,
                                       float(cfg), callback, device, mm)
        else:
            fu.to(device)
            net.to(device)
            audio = generate(clip_frames, sync_frames, [prompt], negative_text=[negative_prompt], feature_utils=fu,
                             net=net, fm=fm, rng=rng, cfg_strength=float(cfg), callback=callback)
    finally:
        net.__dict__.pop("predict_flow", None)
        if offload or low:
            net.to("cpu")
            fu.to("cpu")
            mm.soft_empty_cache()
    wave = audio.float().cpu()
    t_run = time.perf_counter() - t1
    return wave, duration, {"load_s": t_load, "run_s": t_run, "low_vram": low}


def _generate_low_vram(clip_video, sync_video, prompt, negative_prompt, fu, net, fm, rng, cfg, callback, device, mm):
    """MMAudio's generate() with one model on the GPU at a time: CLIP + Synchformer read the video and the prompts,
    then the main network samples, then the VAE + vocoder make the sound. Same maths, same order, same result;
    peak VRAM is the biggest single model instead of all four."""
    import torch

    dtype = next(net.parameters()).dtype

    def swap(on, off):
        for m in off:
            m.to("cpu")
        mm.soft_empty_cache()
        for m in on:
            m.to(device)

    with torch.inference_mode():  # as MMAudio's own generate()
        return _low_vram_steps(clip_video, sync_video, prompt, negative_prompt, fu, net, fm, rng, cfg, callback,
                               device, dtype, swap)


def _low_vram_steps(clip_video, sync_video, prompt, negative_prompt, fu, net, fm, rng, cfg, callback, device, dtype,
                    swap):
    import torch

    swap([fu.clip_model, fu.synchformer], [net, fu.tod])
    clip_f = sync_f = None
    if clip_video is not None:
        clip_f = fu.encode_video_with_clip(clip_video.to(device, dtype), batch_size=1)
    if sync_video is not None:
        sync_f = fu.encode_video_with_sync(sync_video.to(device, dtype), batch_size=1)
    text_f = fu.encode_text([prompt])
    neg_f = fu.encode_text([negative_prompt])

    swap([net], [fu.clip_model, fu.synchformer])
    if clip_f is None:
        clip_f = net.get_empty_clip_sequence(1)
    if sync_f is None:
        sync_f = net.get_empty_sync_sequence(1)
    x0 = torch.randn(1, net.latent_seq_len, net.latent_dim, device=device, dtype=dtype, generator=rng)
    cond = net.preprocess_conditions(clip_f, sync_f, text_f)
    empty = net.get_empty_conditions(1, negative_text_features=neg_f)
    x1 = fm.to_data(lambda t, x: net.ode_wrapper(t, x, cond, empty, cfg), x0, callback)
    x1 = net.unnormalize(x1)
    del cond, empty, clip_f, sync_f, text_f, neg_f

    swap([fu.tod], [net])
    return fu.vocode(fu.decode(x1))


class LCFoley:
    @classmethod
    def INPUT_TYPES(cls):
        names = _model_names()
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "The sounds to make: what makes them, the room, the surface. "
                                                 "Example: soft footsteps on a wooden floor, cups clink."}),
                "negative_prompt": ("STRING", {"default": "music, singing, speech, voices",
                                               "tooltip": "Sounds to keep out."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                  "tooltip": "Frame rate of the connected frames, so the sound lines up with them."}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1,
                                       "tooltip": "Seconds of sound. 0 = the length of the frames. Set it when "
                                                  "no frames are connected. MMAudio was trained on 8 s clips."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200,
                                  "tooltip": "Sampling steps. 25 is MMAudio's default."}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1,
                                  "tooltip": "How closely the sound follows the prompt and the picture."}),
                "seed": ("INT", {"default": 8675309, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),
                "model": (names, {"default": names[0],
                                  "tooltip": "MMAudio model in models/mmaudio. The large 44k v2 model "
                                             "downloads on first use with its helpers (about 5.6 GB in all)."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "The video frames. Leave empty to make sound from the prompt "
                                                "alone."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "make"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = ("Makes sound effects in sync with a video, or from a prompt alone (MMAudio, 44.1 kHz). Nothing to "
                   "install: the model files download to models/mmaudio on first use. Under about 7.5 GB free VRAM it "
                   "loads one model at a time (about 3 GB), same result. The MMAudio weights are "
                   "CC BY-NC 4.0 (non-commercial).")

    def make(self, prompt, negative_prompt, fps, duration, steps, cfg, seed, model=_DEFAULT_MODEL, images=None):
        import comfy.model_management as mm
        import comfy.utils

        bar = comfy.utils.ProgressBar(int(steps))

        def step(i, total):
            bar.update_absolute(i, total)
            mm.throw_exception_if_processing_interrupted()

        mm.throw_exception_if_processing_interrupted()
        wave, dur, t = render(prompt, negative_prompt, images, float(fps), float(duration), int(steps), float(cfg),
                              int(seed), model, callback=step)
        wave = wave[..., :int(round(dur * _SR))].contiguous()
        src = f"{images.shape[0]} frames at {fps:g} fps" if images is not None else "prompt only"
        info = (f"LC Foley: {dur:.2f} s of sound ({src}), {steps} steps, cfg {cfg:g}, seed {seed}, {model}. "
                f"Made in {t['run_s']:.1f} s (model load {t['load_s']:.1f} s)"
                + ("; low VRAM: one model on the GPU at a time." if t.get("low_vram") else "."))
        return ({"waveform": wave, "sample_rate": _SR}, info)


NODE_CLASS_MAPPINGS = {"LCFoley": LCFoley}
NODE_DISPLAY_NAME_MAPPINGS = {"LCFoley": "LC Foley 🔊"}
