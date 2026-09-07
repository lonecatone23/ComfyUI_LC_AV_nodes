"""
LC Audio Separate
-----------------
Four-stem split (bass, drums, other, vocals) matching the common
AudioSeparation layout (christian-byrne / Hybrid Demucs).

Uses torchaudio HDEMUCS when available. No Numba, no SoX.
Channel L/R lives on LC Audio Channel.
"""

from __future__ import annotations

import numpy as np

from .lc_av_media import audio_parts, make_audio

_STEMS = ("bass", "drums", "other", "vocals")
_model_cache = {}


def _fade_window(n, shape):
    if n <= 0:
        return np.ones(0, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if shape == "half_sine":
        return np.sin(0.5 * np.pi * t)
    if shape == "logarithmic":
        return np.log1p(9.0 * t) / np.log(10.0)
    if shape == "exponential":
        return (np.exp(t) - 1.0) / (np.e - 1.0)
    return t


def _get_demucs():
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["sr"]
    import torchaudio
    bundle = torchaudio.pipelines.HDEMUCS_HIGH_FINDLY
    model = bundle.get_model()
    model.eval()
    sr = int(bundle.sample_rate)
    _model_cache["model"] = model
    _model_cache["sr"] = sr
    return model, sr


def _resample(wave, sr, target):
    if sr == target:
        return wave
    try:
        import torch
        import torchaudio
        t = torch.from_numpy(wave)
        out = torchaudio.functional.resample(t, sr, target)
        return out.numpy()
    except Exception:
        n = wave.shape[-1]
        new_n = int(round(n * target / sr))
        x = np.linspace(0, n - 1, new_n)
        src = np.arange(n)
        return np.vstack([np.interp(x, src, wave[c]) for c in range(wave.shape[0])]).astype(np.float32)


def _separate_chunk(model, chunk_2ch):
    import torch
    t = torch.from_numpy(chunk_2ch).float().unsqueeze(0)
    with torch.inference_mode():
        est = model(t)
    arr = est[0].detach().cpu().numpy()
    names = list(getattr(model, "sources", _STEMS))
    return {names[i]: arr[i] for i in range(arr.shape[0])}


class LCAudioSeparate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "Full mix. Split into four Hybrid Demucs stems: Bass, Drums, Other, Vocals. Mono is doubled to stereo first. Needs torchaudio; no Numba/SoX. Use LC Audio Channel if you only want L/R.",
                    },
                ),
                "chunk_fade_shape": (
                    ["linear", "half_sine", "logarithmic", "exponential"],
                    {
                        "default": "linear",
                        "tooltip": "Crossfade shape between overlapping chunks. Linear is safest. half_sine is smoother. logarithmic / exponential change how fast the next chunk takes over at the seam.",
                    },
                ),
                "chunk_length": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 1.0,
                        "max": 600.0,
                        "step": 0.5,
                        "tooltip": "Seconds sent to Demucs at a time. Longer = fewer seams, more VRAM. 8–12 s is a good default on 16 GB; drop toward 4–6 if it OOMs.",
                    },
                ),
                "chunk_overlap": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 0.9,
                        "step": 0.05,
                        "tooltip": "Fraction of each chunk that overlaps the next (0.1 = 10%). Higher hides splice clicks; 0 is fastest and can click at joins.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("Bass", "Drums", "Other", "Vocals")
    FUNCTION = "separate"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = (
        "Hybrid Demucs four-stem split: Bass, Drums, Other, Vocals. "
        "Same layout as common AudioSeparation nodes. Needs torchaudio "
        "(HDEMUCS_HIGH_FINDLY). No Numba/SoX. Chunk length/overlap trade VRAM vs seams."
    )

    def separate(self, audio, chunk_fade_shape="linear", chunk_length=10.0, chunk_overlap=0.1):
        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0:
            empty = make_audio([[0.0]], max(sr, 1))
            return (empty, empty, empty, empty)
        if wave.ndim == 1:
            wave = wave[None, :]
        if wave.shape[0] == 1:
            wave = np.repeat(wave, 2, axis=0)
        elif wave.shape[0] > 2:
            wave = wave[:2]

        try:
            model, model_sr = _get_demucs()
        except Exception as e:
            raise RuntimeError(
                "LC Audio Separate needs torchaudio Hybrid Demucs "
                f"(HDEMUCS_HIGH_FINDLY). Install failed or model download blocked: {e}"
            )

        work = _resample(wave.astype(np.float32), sr, model_sr)
        total = work.shape[-1]
        hop_s = max(1.0, float(chunk_length) * (1.0 - float(chunk_overlap)))
        chunk_n = max(1, int(float(chunk_length) * model_sr))
        hop_n = max(1, int(hop_s * model_sr))
        overlap_n = max(0, chunk_n - hop_n)
        fade = _fade_window(overlap_n, chunk_fade_shape) if overlap_n else None

        acc = {k: np.zeros_like(work) for k in _STEMS}
        weight = np.zeros((1, total), dtype=np.float32)

        pos = 0
        while pos < total:
            end = min(total, pos + chunk_n)
            chunk = work[:, pos:end]
            if chunk.shape[-1] < chunk_n:
                pad = np.zeros((2, chunk_n - chunk.shape[-1]), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=1)
            stems = _separate_chunk(model, chunk)
            use = end - pos
            win = np.ones(use, dtype=np.float32)
            if fade is not None and pos > 0:
                n = min(overlap_n, use)
                win[:n] = fade[:n]
            for name in _STEMS:
                src = stems.get(name)
                if src is None:
                    continue
                if src.ndim == 3:
                    src = src[0]
                if src.shape[0] == 1:
                    src = np.repeat(src, 2, axis=0)
                acc[name][:, pos:end] += src[:, :use] * win
            weight[:, pos:end] += win
            if end >= total:
                break
            pos += hop_n

        weight = np.maximum(weight, 1e-6)
        out = []
        for name in _STEMS:
            stem = acc[name] / weight
            stem = _resample(stem, model_sr, sr)
            out.append(make_audio(stem, sr))
        return tuple(out)


NODE_CLASS_MAPPINGS = {
    "LCAudioSeparate": LCAudioSeparate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCAudioSeparate": "LC Audio Separate 🎼",
}
