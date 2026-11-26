"""
LC Audio Split
--------------
Splits a soundtrack into speech, sound effects and music with Dynamic Band Splitting (DBS, DnR v2 models). The three
parts add back up to the input, so LC Audio Mix can rebuild it with each part at its own level, or with one replaced.

Model code vendored in vendor/dbs_split (Apache-2.0, via ComfyUI-DBS, MIT); loader and overlap-add call adapted from
ComfyUI-DBS runtime.py. Weights download on first use into models/dbs (pinned revision, SHA-256 checked).
"""

from __future__ import annotations

import hashlib
import json
import math
import os

_SR = 44100
_FILES = ("config.json", "model.safetensors")
_MODELS = {  # name: (repo, revision, chunk s, hop s, {file: sha256})
    "TIGER (best)": ("JusperLee/DBS-TIGER-DnR", "22da7d4cd3ea63f9365181008bff1c0d2ab6fb2b", 6.0, 3.0, {
        "config.json": "c7f7893323e78f9b99747a13090558a13c4f4b77383a38acb9979709fd04d1d8",
        "model.safetensors": "3d8205743f2a6f5a6704f5d4104d7cf5bf7bf02dab2fd728da31f4c69225cadb"}),
    "BSRNN (faster on CPU)": ("JusperLee/DBS-BSRNN-DnR", "5af343f57f106fbb95efe8a60f1a3999eeb0aef8", 3.0, 1.5, {
        "config.json": "57055e2b71913d98a4e1ed426b71617f0dfc372901cc5a8a3bc4a5c69415043e",
        "model.safetensors": "26290f93e7b501113325e922e7b0c18edca944634e6b9be39856a55204628405"}),
}
_cache = {}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _folder(choice):
    """models/dbs/<repo name>: the folder ComfyUI-DBS uses too, so both share one copy. Downloads what is missing."""
    import folder_paths

    repo, rev, _, _, sums = _MODELS[choice]
    name = repo.split("/")[1]
    bases = [os.path.join(folder_paths.models_dir, "dbs")]
    try:
        bases = folder_paths.get_folder_paths("dbs") + bases
    except Exception:
        pass
    for base in bases:
        d = os.path.join(base, name)
        if all(os.path.isfile(os.path.join(d, f)) for f in _FILES):
            return d
    from .lc_first_download import download

    d = os.path.join(bases[-1], name)
    for f in _FILES:
        dest = os.path.join(d, f)
        download(f"https://huggingface.co/{repo}/resolve/{rev}/{f}", dest, f"DBS {name}", "5-20 MB")
        if _sha(dest) != sums[f]:
            os.remove(dest)
            raise RuntimeError(f"{name}/{f} failed its checksum and was removed. Try again.")
    return d


def _load(choice, device):
    import torch
    from safetensors.torch import load_model

    from .vendor.dbs_split.dbs.models import DBSBSRNN, DBSTIGER

    key = (choice, str(device))
    if key not in _cache:
        d = _folder(choice)
        with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        model = (DBSTIGER if choice.startswith("TIGER") else DBSBSRNN)(**cfg)
        load_model(model, os.path.join(d, "model.safetensors"), strict=True)
        _cache.clear()
        _cache[key] = model.to(device).eval()
    return _cache[key]


def _resample(x, orig, new):
    if orig == new:
        return x
    import comfy.audio

    return comfy.audio.resample(x, orig, new)


class LCAudioSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "A soundtrack: an H3 clip's audio, a film scene, etc."}),
                "model": (list(_MODELS), {"default": "TIGER (best)",
                                          "tooltip": "TIGER is the cleaner split. BSRNN is faster on CPU. Both are tiny "
                                                     "(5-20 MB) and download on first use."}),
                "keep_stereo": ("BOOLEAN", {"default": False,
                                            "tooltip": "Off: splits the mono mix (the way the model was trained, fastest). "
                                                       "On: splits each channel on its own and keeps stereo (twice as long)."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("speech", "sfx", "music", "info")
    FUNCTION = "split"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = ("Splits a soundtrack into speech, sound effects and music. The three add back up to the input, so LC "
                   "Audio Mix can rebuild it with each part at its own level, or with one part swapped for a new one "
                   "(a TTS line, new Foley). Trained on English speech; singing tends to land in music.")

    def split(self, audio, model, keep_stereo=False):
        import time

        import torch

        import comfy.model_management as mm
        import comfy.utils

        from .vendor.dbs_split.dbs.engine.inference import chunked

        wave, sr = audio["waveform"], int(audio["sample_rate"])
        if wave.ndim == 2:
            wave = wave.unsqueeze(0)
        if wave.shape[-1] == 0:
            raise ValueError("LC Audio Split: the audio is empty (a video with no sound?).")
        dev = mm.get_torch_device()
        dev = dev if dev.type == "cuda" else torch.device("cpu")
        net = _load(model, dev)
        _, _, chunk_s, hop_s, _ = _MODELS[model]
        chunk, hop = int(round(_SR * chunk_s)), int(round(_SR * hop_s))
        item = wave[0].float()
        signals = list(item) if keep_stereo else [item.mean(0)]
        n44 = math.ceil(item.shape[-1] * _SR / sr)
        steps = max(1, len(signals) * (1 if n44 <= chunk else math.ceil((n44 - chunk) / hop) + 1))
        bar = comfy.utils.ProgressBar(steps)
        done = [0]

        def run(piece):
            mm.throw_exception_if_processing_interrupted()
            out = net(piece)
            done[0] += 1
            bar.update_absolute(min(done[0], steps), steps)
            return out

        t0 = time.perf_counter()
        stems = []
        with torch.inference_mode():
            for x in signals:
                x = _resample(x.cpu(), sr, _SR)
                stems.append(chunked(run, x.reshape(1, 1, -1).to(dev), chunk, hop)[0].float().cpu())  # [3, T]
        out = torch.stack(stems, dim=1)  # [3 (music, sfx, speech), channels, T]
        out = _resample(out, _SR, sr)[..., :item.shape[-1]]
        part = {n: {"waveform": out[i].unsqueeze(0).contiguous(), "sample_rate": sr}
                for i, n in enumerate(("music", "sfx", "speech"))}
        secs = time.perf_counter() - t0
        dur = item.shape[-1] / sr
        info = (f"{model}: {dur:.1f} s of audio split in {secs:.1f} s ({dur / max(secs, 1e-6):.1f}x real time), "
                f"{'stereo' if keep_stereo else 'mono'}, {sr} Hz.")
        return (part["speech"], part["sfx"], part["music"], info)


NODE_CLASS_MAPPINGS = {"LCAudioSplit": LCAudioSplit}
NODE_DISPLAY_NAME_MAPPINGS = {"LCAudioSplit": "LC Audio Split 🎚️"}
