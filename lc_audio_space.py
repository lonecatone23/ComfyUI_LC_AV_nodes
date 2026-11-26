"""
LC Audio Space
--------------
Puts a dry voice (or any sound) into a place. Rooms use a synthetic reverb: a few early reflections, then a tail of
exponentially decaying, low-passed noise (a different noise per channel, so stereo gets width). Devices (telephone,
AM radio, walkie-talkie, megaphone) band-limit the sound, saturate it a little, squash it and add a touch of noise.
The result keeps the input's loudness and never peaks above the input's peak.
"""

from __future__ import annotations

import math

import numpy as np

from .lc_av_media import audio_parts, make_audio
from .lc_loudness import compress, limit

# rooms: decay_s (RT60), wet, predelay_ms, brightness_hz, early-reflection spread ms, reflections
_ROOMS = {
    "Small room": (0.35, 0.22, 3.0, 6000.0, 15.0, 8),
    "Living room": (0.55, 0.25, 5.0, 5000.0, 25.0, 10),
    "Cafe": (0.9, 0.28, 10.0, 4500.0, 35.0, 12),
    "Hall": (1.8, 0.32, 20.0, 5500.0, 60.0, 14),
    "Church": (4.0, 0.40, 35.0, 3500.0, 90.0, 16),
    "Street (outdoors)": (0.6, 0.12, 0.0, 4000.0, 120.0, 3),
    "Car interior": (0.12, 0.25, 1.0, 3500.0, 4.0, 6),
    "Bathroom": (0.9, 0.40, 2.0, 8000.0, 8.0, 10),
}
# devices: low Hz, high Hz, resonance (Hz, dB) or None, drive, noise dB below the signal, compress, room (decay, wet)
_DEVICES = {
    "Telephone": (300.0, 3400.0, (1500.0, 3.0), 2.0, 45.0, True, None),
    "Radio (AM)": (150.0, 4500.0, (2500.0, 2.0), 1.5, 32.0, True, None),
    "Walkie-talkie": (400.0, 2800.0, (1800.0, 4.0), 4.0, 28.0, True, None),
    "Megaphone": (500.0, 4000.0, (1800.0, 8.0), 3.0, 50.0, True, (0.6, 0.18)),
}
_PRESETS = list(_ROOMS) + list(_DEVICES) + ["Custom"]


def _peaking(sr, f0, gain_db, q=1.2):
    """RBJ peaking EQ biquad as one SOS row."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * min(f0, sr * 0.45) / sr
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
    a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])


def _band(x, sr, low, high, order=4):
    from scipy import signal

    high = min(high, sr * 0.45)
    low = min(low, high * 0.5)
    sos = signal.butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, x, axis=-1)


def _impulse(sr, channels, decay_s, predelay_ms, brightness, er_ms, n_er, seed=7):
    """Synthetic room impulse response [C, N]: direct-free, early reflections + decaying filtered noise tail."""
    from scipy import signal

    rng = np.random.default_rng(seed)
    pre = int(sr * predelay_ms / 1000.0)
    n = pre + int(sr * min(decay_s * 1.2, 8.0)) + 1
    t = np.arange(n - pre) / sr
    env = np.exp(-6.91 * t / max(decay_s, 0.05))  # -60 dB at decay_s
    lp = signal.butter(2, min(brightness, sr * 0.45), btype="lowpass", fs=sr, output="sos")
    hp = signal.butter(1, 80.0, btype="highpass", fs=sr, output="sos")
    ir = np.zeros((channels, n))
    fade_in = np.clip(t / max(er_ms / 1000.0, 0.002), 0.0, 1.0)  # tail builds up after the early reflections
    for c in range(channels):
        tail = signal.sosfilt(lp, rng.standard_normal(n - pre)) * env * fade_in
        er = np.zeros(n - pre)
        for k in range(n_er):
            d = int(sr * (er_ms / 1000.0) * (k + 1 + rng.uniform(-0.4, 0.4)) / n_er)
            if 0 <= d < er.size:
                er[d] += rng.choice([-1.0, 1.0]) * (0.8 ** k) * 1.5
        er = signal.sosfilt(lp, er)
        tail = tail / max(np.sqrt((tail ** 2).sum()), 1e-9)
        ir[c, pre:] = signal.sosfilt(hp, tail + er * 0.6)
        ir[c] /= max(np.sqrt((ir[c] ** 2).sum()), 1e-9)
    return ir


def _reverb(x, sr, decay_s, wet, predelay_ms, brightness, er_ms, n_er):
    from scipy.signal import fftconvolve

    ir = _impulse(sr, x.shape[0], decay_s, predelay_ms, brightness, er_ms, n_er)
    rev = np.stack([fftconvolve(x[c], ir[c])[:x.shape[-1] + ir.shape[-1] - 1] for c in range(x.shape[0])])
    dry = np.pad(x, ((0, 0), (0, rev.shape[-1] - x.shape[-1])))
    rms_d = np.sqrt(np.mean(x ** 2)) + 1e-12
    rms_r = np.sqrt(np.mean(rev ** 2)) + 1e-12
    rev = rev * (rms_d / rms_r)
    return (1.0 - wet) * dry + wet * rev


def _device(x, sr, low, high, res, drive, noise_db, squash, seed=11):
    from scipy import signal

    y = _band(x, sr, low, high)
    if res is not None:
        y = signal.sosfilt(_peaking(sr, res[0], res[1]), y, axis=-1)
    pk = np.abs(y).max() + 1e-12
    y = np.tanh(drive * y / pk) / math.tanh(drive)
    if squash:
        y, _ = compress(y.astype(np.float32), sr, threshold_db=-18.0, ratio=4.0, attack_ms=3.0, release_ms=80.0)
    if noise_db > 0:
        rng = np.random.default_rng(seed)
        rms = np.sqrt(np.mean(y ** 2)) + 1e-12
        hiss = _band(rng.standard_normal(y.shape), sr, low, high, order=2)
        hiss *= rms * 10 ** (-noise_db / 20.0) / (np.sqrt(np.mean(hiss ** 2)) + 1e-12)
        y = y + hiss
    return _band(y, sr, low * 0.8, high * 1.1, order=2)


class LCAudioSpace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "preset": (_PRESETS, {"default": "Living room",
                                      "tooltip": "The place or device. Custom uses the sliders below."}),
                "keep_length": ("BOOLEAN", {"default": True,
                                            "tooltip": "On: same length as the input (keeps video sync). "
                                                       "Off: lets the echo ring out at the end."}),
            },
            "optional": {
                "decay_s": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 6.0, "step": 0.05,
                                      "tooltip": "Custom only. How long the echo rings (room size)."}),
                "wet": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider",
                                  "tooltip": "Custom only. How much echo against the dry sound."}),
                "predelay_ms": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 200.0, "step": 1.0,
                                          "tooltip": "Custom only. Gap before the echo starts. Bigger feels like "
                                                     "a bigger room."}),
                "brightness": ("FLOAT", {"default": 5000.0, "min": 500.0, "max": 16000.0, "step": 100.0,
                                         "tooltip": "Custom only. Low-pass on the echo in Hz. Low = soft rooms, "
                                                    "high = tiles and glass."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "place"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = "Room reverb and device sounds (phone, radio, walkie-talkie, megaphone) for dry voices."

    def place(self, audio, preset="Living room", keep_length=True, decay_s=1.0, wet=0.3, predelay_ms=10.0,
              brightness=5000.0):
        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0 or wave.size == 0 or float(np.abs(wave).max()) < 1e-6:
            return (audio, "no sound")
        x = np.asarray(wave, dtype=np.float64)
        n = x.shape[-1]
        peak_in = float(np.abs(x).max())
        rms_in = float(np.sqrt(np.mean(x ** 2)))
        if preset in _DEVICES:
            low, high, res, drive, noise_db, squash, room = _DEVICES[preset]
            y = _device(x, sr, low, high, res, drive, noise_db, squash)
            if room:
                y = _reverb(y, sr, room[0], room[1], 0.0, 3000.0, 120.0, 3)
            desc = f"{preset}: {low:.0f}-{min(high, sr * 0.45):.0f} Hz, drive {drive:.1f}"
        else:
            if preset in _ROOMS:
                d, w, p, b, er, ner = _ROOMS[preset]
            else:
                d, w, p, b, er, ner = (float(decay_s), float(wet), float(predelay_ms), float(brightness),
                                       min(100.0, 10.0 + float(decay_s) * 20.0), 10)
            y = _reverb(x, sr, d, w, p, b, er, ner)
            desc = f"{preset}: decay {d:.2f} s, wet {w:.2f}, predelay {p:.0f} ms, brightness {b:.0f} Hz"
        if keep_length:
            y = y[:, :n].copy()
            f = min(int(sr * 0.02), n // 4)
            if f > 1:  # the cut-off echo tail fades out instead of clicking
                y[:, -f:] *= np.linspace(1.0, 0.0, f)
        rms_out = float(np.sqrt(np.mean(y[:, :n] ** 2))) + 1e-12
        y = y * (rms_in / rms_out)
        y, lim = limit(y.astype(np.float32), sr, ceiling_db=20 * math.log10(peak_in) - 0.1)
        peak_out = float(np.abs(y).max())
        info = (f"{desc} | {n / sr:.2f} s -> {y.shape[-1] / sr:.2f} s | peak {20 * math.log10(peak_in):.1f} -> "
                f"{20 * math.log10(max(peak_out, 1e-9)):.1f} dBFS | limiter {lim:.1f} dB")
        return (make_audio(y, sr), info)


NODE_CLASS_MAPPINGS = {"LCAudioSpace": LCAudioSpace}
NODE_DISPLAY_NAME_MAPPINGS = {"LCAudioSpace": "LC Audio Space 🏛️"}
