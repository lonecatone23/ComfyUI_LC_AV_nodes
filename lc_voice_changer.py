"""
LC Voice Changer
----------------
Speech in, the same performance out in another voice: the words, timing and feeling of `audio`, spoken by the person
in `voice`. ChatterBox VC (Resemble AI, MIT), vendored in vendor/chatterbox_vc: the speech becomes S3 speech tokens,
then ChatterBox's decoder speaks them again with the target's voice (from the first 10 s of `voice`).

Model: models/TTS/chatterbox/English/s3gen.safetensors (ResembleAI/chatterbox, 1.06 GB), the same file
TTS-Audio-Suite uses, so an existing download is reused. Downloads on first use otherwise.
"""

from __future__ import annotations

import os

import numpy as np

from .lc_av_media import audio_parts, make_audio, resample

REPO = "ResembleAI/chatterbox"
FILE = "s3gen.safetensors"
S3_SR, GEN_SR = 16000, 24000
REF_SECONDS = 10  # the decoder only looks at the first 10 s of the target voice
CHUNK_SECONDS = 30  # longer speech goes through in pieces cut at quiet spots

_cache = {}


def _path():
    import folder_paths

    folder = os.path.join(folder_paths.models_dir, "TTS", "chatterbox", "English")
    path = os.path.join(folder, FILE)
    if not os.path.isfile(path):
        from .lc_first_download import download, hf_url, toast

        toast("ChatterBox voice changer", "1.06 GB")
        download(hf_url(REPO, FILE), path, "ChatterBox voice changer", "1.06 GB", announce=False)
    return path


def _model(device):
    import torch
    from safetensors.torch import load_file

    if _cache.get("device") == device:
        return _cache["model"]
    _cache.clear()
    from .vendor.chatterbox_vc.models.s3gen import S3Gen

    m = S3Gen()
    missing, unexpected = m.load_state_dict(load_file(_path()), strict=False)
    missing = [k for k in missing if k != "tokenizer.window"]  # a Hann window the tokenizer builds itself
    if missing or unexpected:
        print(f"[LC Voice Changer] checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
    m.to(device).eval()
    _cache.update(model=m, device=device)
    return m


def unload():
    _cache.clear()
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:
        pass


def _mono(audio):
    wave, sr = audio_parts(audio)
    if wave is None or sr <= 0 or wave.size == 0:
        return None, 0
    w = np.asarray(wave, dtype=np.float32)
    return (w.mean(0) if w.ndim == 2 else w), int(sr)


def _pieces(x, sr, limit=CHUNK_SECONDS):
    """Cut speech longer than `limit` seconds at the quietest 50 ms in the last third of each piece."""
    out, start, n, win = [], 0, len(x), int(0.05 * sr)
    while n - start > limit * sr:
        lo, hi = start + int(limit * sr * 2 / 3), start + int(limit * sr)
        seg = x[lo:hi]
        k = max(1, len(seg) // win)
        energy = (seg[:k * win].reshape(k, win) ** 2).mean(1)
        cut = lo + int(np.argmin(energy)) * win + win // 2
        out.append(x[start:cut])
        start = cut
    out.append(x[start:])
    return out


class LCVoiceChanger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The performance: what is said, and how. Speech only (LC Audio Split "
                                               "takes music and effects off first)."}),
                "voice": ("AUDIO", {"tooltip": "Who it should sound like: one person talking. Only the first 10 s are "
                                               "used, so start the clip on clear speech. LC Load Voice's audio works."}),
                "passes": ("INT", {"default": 1, "min": 1, "max": 5,
                                   "tooltip": "Runs the result through again, closer to the target voice each time. "
                                              "Past 3 it starts to smear."}),
                "seed": ("INT", {"default": 8675309, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": "fixed",
                                 "tooltip": "Each pass uses seed + its number."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    OUTPUT_TOOLTIPS = ("The same performance in the new voice (24 kHz).", "What was done.")
    FUNCTION = "change"
    CATEGORY = "LC AV/voice"
    DESCRIPTION = ("Keeps the words, timing and feeling of a recording and swaps the voice for another one "
                   "(ChatterBox VC). Give a flat line a voice with range, or re-voice a performance.")

    def change(self, audio, voice, passes=1, seed=8675309):
        import time

        import comfy.model_management as mm
        import torch

        x, sr = _mono(audio)
        ref, rsr = _mono(voice)
        if x is None:
            raise ValueError("LC Voice Changer: audio has no sound.")
        if ref is None:
            raise ValueError("LC Voice Changer: voice has no sound.")
        t0 = time.time()
        device = mm.get_torch_device()
        m = _model(device)
        ref24 = resample(ref, rsr, GEN_SR)[:REF_SECONDS * GEN_SR]
        cur = resample(x, sr, S3_SR)
        out = cur
        with torch.inference_mode():
            ref_dict = m.embed_ref(torch.from_numpy(np.ascontiguousarray(ref24)).float(), GEN_SR, device=device)
            for p in range(int(passes)):
                torch.manual_seed(int(seed) + p + 1)
                parts = []
                for piece in _pieces(cur, S3_SR):
                    mm.throw_exception_if_processing_interrupted()
                    if len(piece) < S3_SR // 10:
                        continue
                    wav16 = torch.from_numpy(np.ascontiguousarray(piece)).float().to(device)[None]
                    tokens, _ = m.tokenizer(wav16)
                    wav, _ = m.inference(speech_tokens=tokens, ref_dict=ref_dict)
                    parts.append(wav.squeeze(0).float().cpu().numpy())
                out = np.concatenate(parts) if parts else np.zeros(1, np.float32)
                if p + 1 < int(passes):
                    cur = resample(out, GEN_SR, S3_SR)
        peak = float(np.abs(out).max()) if out.size else 0.0
        if peak > 0.99:
            out = out * (0.99 / peak)
        info = (f"{len(out) / GEN_SR:.1f} s, {passes} pass{'es' if passes > 1 else ''}, voice from the first "
                f"{min(len(ref) / rsr, REF_SECONDS):.1f} s of the clip ({time.time() - t0:.1f} s)")
        return (make_audio(out[None].astype(np.float32), GEN_SR), info)


NODE_CLASS_MAPPINGS = {"LCVoiceChanger": LCVoiceChanger}
NODE_DISPLAY_NAME_MAPPINGS = {"LCVoiceChanger": "LC Voice Changer 🔁 (BETA)"}
