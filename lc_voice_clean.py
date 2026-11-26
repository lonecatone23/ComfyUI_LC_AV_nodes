"""
LC Voice Clean
--------------
Three clean-up stages for voice recordings, each with its own amount (0 = off):
  noise_reduction  spectral gating. The noise print comes from the quietest 10% of the clip, per frequency; a soft
                   mask, smoothed over time and frequency, turns down whatever is not louder than that print. The
                   smoothing keeps away the "musical noise" warble.
  deess            turns down the 5-10 kHz band in the moments it spikes above the rest of the voice (harsh S / SH).
  gate             turns the gaps between words down to gate_floor_db (never a hard cut), with a quick open and a
                   slower close so word tails are kept.
The info output says how many dB each stage took off.
"""

from __future__ import annotations

import math

import numpy as np

from .lc_av_media import audio_parts, make_audio


def _floor_db(x, sr):
    """Level of the quietest 10% of 20 ms frames (dBFS): the noise floor."""
    win = max(1, int(sr * 0.02))
    p = np.mean(x.astype(np.float64) ** 2, axis=0)
    n = p.shape[0] // win
    if n < 1:
        return float(10 * np.log10(max(p.mean(), 1e-14)))
    fr = p[:n * win].reshape(n, win).mean(1)
    fr = fr[fr > 1e-14]
    if fr.size == 0:
        return -140.0
    k = max(1, int(fr.size * 0.1))
    return float(10 * np.log10(np.sort(fr)[:k].mean()))


def _spectral(x, sr, noise_amount, deess_amount):
    """Noise gate + de-esser in one STFT pass. Returns (audio, most de-ess dB, % of frames de-essed)."""
    from scipy import ndimage, signal

    nper = int(2 ** round(math.log2(max(256.0, sr * 0.021))))
    nover = nper * 3 // 4
    n = x.shape[-1]
    _, _, Z = signal.stft(x, fs=sr, nperseg=nper, noverlap=nover, window="hann")  # [C, F, T]
    mag = np.abs(Z)
    freqs = np.fft.rfftfreq(nper, 1.0 / sr)
    gain = np.ones(mag.shape, dtype=np.float64)
    if noise_amount > 0:
        energy = (mag ** 2).sum(axis=1)  # [C, T]
        prof = np.zeros(mag.shape[:2])
        for c in range(mag.shape[0]):
            e = energy[c]
            valid = np.where(e > e.max() * 1e-10)[0]
            if valid.size == 0:
                continue
            k = max(1, int(valid.size * 0.1))
            quiet = valid[np.argsort(e[valid])[:k]]
            prof[c] = mag[c][:, quiet].mean(axis=1)
        snr_db = 20 * np.log10(np.maximum(mag, 1e-12) / np.maximum(prof[:, :, None], 1e-12))
        mask = 1.0 / (1.0 + np.exp(-(snr_db - 9.0) / 2.0))  # half way at 9 dB over the print (it reads low)
        mask = ndimage.gaussian_filter(mask, sigma=(0, 1.0, 1.5), mode="nearest")
        floor = 10 ** (-noise_amount * 25.0 / 20.0)
        gain *= floor + (1.0 - floor) * mask
    most, share = 0.0, 0.0
    if deess_amount > 0 and sr / 2 > 6000:
        band = (freqs >= 5000) & (freqs <= 10000)
        rest = (freqs >= 300) & (freqs < 5000)
        pw = (mag ** 2).sum(axis=0)  # both channels together so the stereo image stays put
        b_db = 10 * np.log10(np.maximum(pw[band].sum(0), 1e-14))
        r_db = 10 * np.log10(np.maximum(pw[rest].sum(0), 1e-14))
        excess = (b_db - r_db) + 8.0  # starts when the hiss band gets within 8 dB of the voice body
        excess[b_db < b_db.max() - 45] = 0.0  # ignore the hiss in silent gaps
        red = np.clip(excess, 0.0, 20.0) * deess_amount
        red = ndimage.maximum_filter1d(red, size=3, mode="nearest")
        red = ndimage.gaussian_filter1d(red, sigma=1.0, mode="nearest")
        w = np.clip((freqs - 4000.0) / 1000.0, 0.0, 1.0)
        w = np.where(freqs > 10000, 0.6, w)
        gain *= 10 ** (-(w[:, None] * red[None, :]) / 20.0)[None]
        most = float(red.max())
        share = float((red > 1.0).mean() * 100)
    _, y = signal.istft(Z * gain, fs=sr, nperseg=nper, noverlap=nover, window="hann")
    y = y[:, :n]
    if y.shape[-1] < n:
        y = np.pad(y, ((0, 0), (0, n - y.shape[-1])))
    return y.astype(np.float32), most, share


def _gate(x, sr, threshold_db, floor_db):
    """Downward gate with a soft 6 dB knee, 30 ms look-ahead / hold, 2 ms open, 80 ms close."""
    from scipy import ndimage

    hop = max(1, int(sr / 1000))
    n = x.shape[-1]
    frames = -(-n // hop)
    p = np.pad(np.mean(x.astype(np.float64) ** 2, axis=0), (0, frames * hop - n)).reshape(frames, hop).mean(1)
    p = ndimage.uniform_filter1d(p, size=5, mode="nearest")  # 5 ms RMS
    level = 10 * np.log10(np.maximum(p, 1e-14))
    depth = -float(floor_db)
    red = depth * np.clip((threshold_db - level) / 6.0, 0.0, 1.0)
    red = ndimage.minimum_filter1d(red, size=61, mode="nearest")  # stay open 30 ms around every sound
    open_c = math.exp(-1.0 / 2.0)
    close_c = math.exp(-1.0 / 80.0)
    g = np.empty(frames)
    cur = 0.0
    for i in range(frames):
        t = red[i]
        c = close_c if t > cur else open_c
        cur = c * cur + (1 - c) * t
        g[i] = cur
    gain = 10 ** (-np.interp(np.arange(n), np.arange(frames) * hop + hop / 2, g) / 20)
    return (x * gain[None, :]).astype(np.float32), float(g.max() if frames else 0.0), float((g > 3.0).mean() * 100)


class LCVoiceClean:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "noise_reduction": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                              "display": "slider",
                                              "tooltip": "Removes steady noise (hiss, hum, fan, rain). 0 = off, "
                                                         "0.5 = up to -12 dB, 1 = up to -25 dB. Too high sounds watery."}),
                "deess": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider",
                                    "tooltip": "Tames harsh S and SH sounds. 0 = off. 0.3 is gentle, 1 is strong."}),
                "gate_threshold_db": ("FLOAT", {"default": -50.0, "min": -90.0, "max": -10.0, "step": 1.0,
                                                "tooltip": "Gate: anything quieter than this (dBFS) between words "
                                                           "gets turned down."}),
                "gate_floor_db": ("FLOAT", {"default": -30.0, "min": -80.0, "max": 0.0, "step": 1.0,
                                            "tooltip": "How far the gate turns gaps down. 0 = gate off. "
                                                       "-30 keeps a natural room tone."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "clean"
    CATEGORY = "LC AV/voice"
    DESCRIPTION = "Noise reduction, de-esser and soft gate for voice recordings. Each stage has its own amount."

    def clean(self, audio, noise_reduction=0.5, deess=0.3, gate_threshold_db=-50.0, gate_floor_db=-30.0):
        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0 or wave.size == 0 or float(np.abs(wave).max()) < 1e-6:
            return (audio, "no sound")
        x = np.asarray(wave, dtype=np.float32)
        nr = float(noise_reduction)
        de = float(deess)
        parts = []
        f0 = _floor_db(x, sr)
        if nr > 0 or de > 0:
            x, most, share = _spectral(x, sr, nr, de)
            f1 = _floor_db(x, sr)
            parts.append(f"noise reduction {f0 - f1:.1f} dB (floor {f0:.1f} -> {f1:.1f} dBFS)" if nr > 0
                         else "noise reduction off")
            parts.append(f"de-ess up to {most:.1f} dB on {share:.0f}% of the clip" if de > 0 and sr / 2 > 6000
                         else "de-ess off")
        else:
            parts.append("noise reduction off")
            parts.append("de-ess off")
        if float(gate_floor_db) < 0:
            g0 = _floor_db(x, sr)
            x, depth, closed = _gate(x, sr, float(gate_threshold_db), float(gate_floor_db))
            parts.append(f"gate up to {depth:.1f} dB, closed {closed:.0f}% (floor {g0:.1f} -> "
                         f"{_floor_db(x, sr):.1f} dBFS)")
        else:
            parts.append("gate off")
        peak = float(np.abs(wave).max())
        if float(np.abs(x).max()) > peak:
            x = x * (peak / float(np.abs(x).max()))
        return (make_audio(x, sr), " | ".join(parts))


NODE_CLASS_MAPPINGS = {"LCVoiceClean": LCVoiceClean}
NODE_DISPLAY_NAME_MAPPINGS = {"LCVoiceClean": "LC Voice Clean 🧽"}
