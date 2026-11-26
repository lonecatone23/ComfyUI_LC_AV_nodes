"""Frame / audio helpers. No Numba, no SoX."""

from __future__ import annotations

import numpy as np


def frames_from_any(src):
    """IMAGE tensor, AV pipe slot, or Comfy VIDEO → frame tensor or None."""
    if src is None:
        return None
    if isinstance(src, dict):
        for key in ("video", "images", "frames", "image"):
            if src.get(key) is not None:
                return src[key]
        return None
    if hasattr(src, "get_components"):
        try:
            comp = src.get_components()
            if isinstance(comp, dict) and comp.get("images") is not None:
                return comp["images"]
        except Exception:
            pass
    for attr in ("images", "_images", "frames"):
        if hasattr(src, attr):
            val = getattr(src, attr)
            if val is not None:
                return val
    return src


def tensor_to_uint8_frames(images):
    t = frames_from_any(images)
    if t is None:
        raise ValueError("LC AV: no frames.")
    if hasattr(t, "cpu"):
        t = t.detach().cpu()
    arr = np.asarray(t)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError("LC AV: expected IMAGE [B,H,W,C] or [H,W,C].")
    if arr.shape[-1] > 4:
        arr = arr[..., :4]
    if arr.dtype != np.uint8:
        arr = np.clip(arr.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def audio_parts(audio):
    """Comfy AUDIO dict → (waveform ndarray [C, T], sample_rate) or (None, 0)."""
    if audio is None:
        return None, 0
    if isinstance(audio, dict):
        wave = audio.get("waveform")
        sr = int(audio.get("sample_rate") or 0)
        if wave is None:
            return None, sr
        if hasattr(wave, "cpu"):
            wave = wave.detach().cpu()
        arr = np.asarray(wave)
        # [B, C, T] or [C, T]
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr.astype(np.float32), sr
    return None, 0


def audio_duration_sec(audio) -> float:
    wave, sr = audio_parts(audio)
    if wave is None or sr <= 0:
        return 0.0
    return float(wave.shape[-1]) / float(sr)


def _pcm_bytes(arr, bits):
    pcm = np.clip(np.asarray(arr, dtype=np.float64), -1.0, 1.0).T.reshape(-1)  # interleaved
    if bits == 24:
        v = np.round(pcm * 8388607.0).astype("<i4")
        return v.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    return np.round(pcm * 32767.0).astype("<i2").tobytes()


def write_wav(path, wave, sr, bits=16):
    """PCM WAV (16 or 24 bit) with the standard library only. wave is [C, T] float32, or a file object as path."""
    import os
    import wave as wavmod

    arr = np.asarray(wave)
    if arr.ndim == 1:
        arr = arr[None, :]
    if isinstance(path, str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wavmod.open(path, "wb") as w:
        w.setnchannels(int(arr.shape[0]))
        w.setsampwidth(3 if bits == 24 else 2)
        w.setframerate(int(sr))
        w.writeframes(_pcm_bytes(arr, bits))


def resample(wave, sr, target):
    """[C, T] or [T] float audio to a new rate (scipy polyphase, anti-aliased). No torchaudio needed."""
    import math

    from scipy.signal import resample_poly

    sr, target = int(sr), int(target)
    x = np.asarray(wave, dtype=np.float32)
    if sr == target or x.shape[-1] == 0:
        return x
    g = math.gcd(sr, target)
    y = resample_poly(x, target // g, sr // g, axis=-1).astype(np.float32)
    return y[..., :max(1, x.shape[-1] * target // sr)]  # same length torchaudio gave


def read_audio(path):
    """Any audio (or a video's sound) -> (float32 [C, T], sample_rate). PyAV (a ComfyUI requirement), with a plain
    WAV reader as the fallback."""
    errors = []
    try:
        import av

        with av.open(path) as f:
            st = next(s for s in f.streams if s.type == "audio")
            sr = int(st.rate or st.codec_context.sample_rate)
            rs = av.AudioResampler(format="fltp", layout=st.codec_context.layout, rate=sr)
            parts = []
            for frame in f.decode(st):
                for out in rs.resample(frame):
                    parts.append(out.to_ndarray())
            for out in rs.resample(None):
                parts.append(out.to_ndarray())
        if parts:
            return np.concatenate(parts, axis=1).astype(np.float32), sr
        errors.append("pyav: no audio frames")
    except Exception as e:
        errors.append(f"pyav: {e}")
    try:
        import wave as wavmod

        with wavmod.open(path, "rb") as w:
            sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if sw == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sw == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
            data = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float32) / 8388608.0
        elif sw == 4:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif sw == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"wav width {sw}")
        return data.reshape(-1, ch).T.copy(), int(sr)
    except Exception as e:
        errors.append(f"wav: {e}")
    raise RuntimeError(f"Could not decode {path} ({'; '.join(errors)})")


def preview_ui(audio, prefix="lc_av"):
    """Write a temp WAV and return a Comfy ui payload for ▶ preview."""
    import os
    import time

    wave, sr = audio_parts(audio)
    if wave is None or sr <= 0:
        return {}
    try:
        import folder_paths
        folder = folder_paths.get_temp_directory()
        kind = "temp"
    except Exception:
        folder = os.path.join(os.path.dirname(__file__), "tmp")
        kind = "temp"
    name = f"{prefix}_{os.getpid()}_{time.time_ns()}.wav"
    dest = os.path.join(folder, name)
    try:
        write_wav(dest, wave, sr)
    except Exception as e:
        print(f"[LC AV] preview wav skip: {e}")
        return {}
    return {
        "audio": [{
            "filename": name,
            "subfolder": "",
            "type": kind,
        }],
    }


def with_preview(result, audio, prefix="lc_av"):
    ui = preview_ui(audio, prefix=prefix)
    if not ui:
        return result
    return {"ui": ui, "result": result}


def make_audio(wave, sample_rate: int):
    try:
        import torch
        t = torch.from_numpy(np.asarray(wave, dtype=np.float32))
        if t.ndim == 2:
            t = t.unsqueeze(0)
        return {"waveform": t, "sample_rate": int(sample_rate)}
    except Exception:
        return {"waveform": np.asarray(wave), "sample_rate": int(sample_rate)}
