"""lc_dsp: numpy + scipy port of the librosa functions this pack uses.

Ported from librosa 0.11.0 (https://librosa.org), ISC License,
Copyright (c) 2013--2023, librosa development team.
Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

Ported: effects.time_stretch, util.fix_length, beat.beat_track (tempo prior,
DP tracker, trim), onset.onset_detect / onset_strength, pyin (YIN difference
function, parabolic refinement, beta/boltzmann priors, HMM Viterbi),
feature.rms, note_to_hz, hz_to_midi, midi_to_note, plus their internals
(stft/istft, phase_vocoder, slaney mel filterbank, power_to_db, tempogram,
peak_pick). Numba kernels are rewritten in numpy with the same arithmetic
order; the pyin Viterbi exploits the sparse transition band so it stays exact
without a dense 2N x 2N step per frame. No librosa, no numba.
"""

import re

import numpy as np
import scipy.fft
import scipy.signal
import scipy.stats

__all__ = ["time_stretch", "fix_length", "beat_track", "onset_detect", "pyin", "rms",
           "note_to_hz", "hz_to_midi", "midi_to_note"]

_MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10


# ---------------------------------------------------------------- utilities
def _tiny(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating):
        return np.finfo(x.dtype).tiny
    return np.finfo(np.float32).tiny


def _frame(x, frame_length, hop_length):
    """librosa.util.frame along the last axis -> (..., frame_length, n_frames)."""
    if x.shape[-1] < frame_length:
        raise ValueError(f"Input is too short (n={x.shape[-1]}) for frame_length={frame_length}")
    xw = np.lib.stride_tricks.sliding_window_view(x, frame_length, axis=-1)
    return np.moveaxis(xw, -1, -2)[..., ::hop_length]


def _window(n, fftbins=True, name="hann"):
    return scipy.signal.get_window(name, n, fftbins=fftbins)


def _pad_center(data, size):
    n = data.shape[-1]
    lpad = int((size - n) // 2)
    if lpad < 0:
        raise ValueError(f"Target size ({size}) must be at least input size ({n})")
    return np.pad(data, [(0, 0)] * (data.ndim - 1) + [(lpad, int(size - n - lpad))])


def _dtype_r2c(d):
    return np.dtype({np.dtype(np.float32): np.complex64}.get(np.dtype(d), np.complex128))


def _dtype_c2r(d):
    return np.dtype({np.dtype(np.complex64): np.float32}.get(np.dtype(d), np.float64))


def fix_length(data, *, size, axis=-1, **kwargs):
    """librosa.util.fix_length: trim or zero-pad ``data`` to exactly ``size`` along ``axis``."""
    kwargs.setdefault("mode", "constant")
    n = data.shape[axis]
    if n > size:
        sl = [slice(None)] * data.ndim
        sl[axis] = slice(0, size)
        return data[tuple(sl)]
    if n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)
    return data


def _normalize_inf(S, axis):
    """librosa.util.normalize(S, norm=np.inf, axis=axis) with default threshold/fill."""
    mag = np.abs(S).astype(float)
    length = np.max(mag, axis=axis, keepdims=True)
    length[length < _tiny(S)] = 1.0
    out = np.empty_like(S)
    out[:] = S / length
    return out


def _localmax(x):
    """librosa.util.localmax on a 1-d array (first sample is never a max)."""
    out = np.zeros(x.shape, dtype=bool)
    if x.shape[-1] > 2:
        out[1:-1] = (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])
    if x.shape[-1] > 1:
        out[-1] = x[-1] > x[-2]
    return out


def _localmin(x):
    out = np.zeros(x.shape, dtype=bool)
    if x.shape[-1] > 2:
        out[1:-1] = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
    if x.shape[-1] > 1:
        out[-1] = x[-1] < x[-2]
    return out


def _autocorrelate(y, max_size=None, axis=-1):
    if max_size is None:
        max_size = y.shape[axis]
    max_size = int(min(max_size, y.shape[axis]))
    n_pad = scipy.fft.next_fast_len(2 * y.shape[axis] - 1, real=True)
    spec = np.fft.rfft(y, n=n_pad, axis=axis)
    powspec = spec.real ** 2 + spec.imag ** 2
    ac = np.fft.irfft(powspec, n=n_pad, axis=axis)
    sl = [slice(None)] * ac.ndim
    sl[axis] = slice(max_size)
    return ac[tuple(sl)]


# ---------------------------------------------------------------- STFT
def _stft(y, n_fft=2048, hop_length=None, win_length=None, center=True):
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = int(win_length // 4)
    if not np.all(np.isfinite(y)):
        raise ValueError("Audio buffer is not finite everywhere")
    win = _pad_center(_window(win_length), n_fft)[:, None]
    if center:
        y = np.pad(y, [(0, 0)] * (y.ndim - 1) + [(n_fft // 2, n_fft // 2)])
    frames = _frame(y, n_fft, hop_length)
    out = np.zeros(frames.shape[:-2] + (1 + n_fft // 2, frames.shape[-1]),
                   dtype=_dtype_r2c(y.dtype), order="F")
    n_cols = max(int(_MAX_MEM_BLOCK // (np.prod(frames.shape[:-1]) * frames.itemsize)), 1)
    for s in range(0, frames.shape[-1], n_cols):
        t = min(s + n_cols, frames.shape[-1])
        out[..., s:t] = np.fft.rfft(win * frames[..., s:t], axis=-2)
    return out


def _istft(D, hop_length=None, win_length=None, n_fft=None, center=True, dtype=None, length=None):
    if n_fft is None:
        n_fft = 2 * (D.shape[-2] - 1)
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = int(win_length // 4)
    win = _pad_center(_window(win_length), n_fft)[:, None]
    if length:
        padded = length + 2 * (n_fft // 2) if center else length
        n_frames = min(D.shape[-1], int(np.ceil(padded / hop_length)))
    else:
        n_frames = D.shape[-1]
    if dtype is None:
        dtype = _dtype_c2r(D.dtype)
    out_len = n_fft + hop_length * (n_frames - 1)
    if length:
        out_len = length
    elif center:
        out_len -= 2 * (n_fft // 2)
    off = n_fft // 2 if center else 0
    buf_len = max(n_fft + hop_length * (n_frames - 1), off + out_len)
    buf = np.zeros(D.shape[:-2] + (buf_len,), dtype=dtype)
    n_cols = max(int(_MAX_MEM_BLOCK // (np.prod(D.shape[:-1]) * D.itemsize)), 1)
    for s in range(0, n_frames, n_cols):
        t = min(s + n_cols, n_frames)
        ytmp = win * np.fft.irfft(D[..., s:t], n=n_fft, axis=-2)
        for f in range(t - s):
            p = (s + f) * hop_length
            buf[..., p:p + n_fft] += ytmp[..., :, f]
    y = buf[..., off:off + out_len]
    # window sum-square envelope (filters.window_sumsquare, norm=None)
    wss = np.zeros(n_fft + hop_length * (n_frames - 1), dtype=dtype)
    win_sq = _pad_center(_window(win_length, fftbins=True) ** 2, n_fft)
    for i in range(n_frames):
        p = i * hop_length
        wss[p:p + n_fft] += win_sq[:max(0, min(n_fft, len(wss) - p))]
    wss = fix_length(wss[off:], size=y.shape[-1])
    nz = wss > _tiny(wss)
    y[..., nz] /= wss[nz]
    return np.ascontiguousarray(y)


def _phase_vocoder(D, rate, hop_length=None, n_fft=None):
    if n_fft is None:
        n_fft = 2 * (D.shape[-2] - 1)
    if hop_length is None:
        hop_length = int(n_fft // 4)
    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)
    shape = list(D.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros_like(D, shape=shape)
    phi_advance = hop_length * np.fft.rfftfreq(n=n_fft, d=1.0 / (2 * np.pi))
    phase_acc = np.angle(D[..., 0])
    D = np.pad(D, [(0, 0)] * (D.ndim - 1) + [(0, 2)])
    for t, step in enumerate(time_steps):
        cols = D[..., int(step):int(step + 2)]
        alpha = np.mod(step, 1.0)
        mag = (1.0 - alpha) * np.abs(cols[..., 0]) + alpha * np.abs(cols[..., 1])
        z = np.cos(phase_acc) + 1j * np.sin(phase_acc)
        if phase_acc.dtype == np.float32:
            z = z.astype(np.complex64)
        z *= mag
        d_stretch[..., t] = z
        dphase = np.angle(cols[..., 1]) - np.angle(cols[..., 0]) - phi_advance
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        phase_acc += phi_advance + dphase
    return d_stretch


def time_stretch(y, *, rate, n_fft=2048, hop_length=None, win_length=None, center=True):
    """librosa.effects.time_stretch (phase vocoder). rate > 1 = faster/shorter."""
    if rate <= 0:
        raise ValueError("rate must be a positive number")
    D = _stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)
    D = _phase_vocoder(D, rate=rate, hop_length=hop_length, n_fft=n_fft)
    n = int(round(y.shape[-1] / rate))
    return _istft(D, hop_length=hop_length, win_length=win_length, n_fft=n_fft,
                  center=center, dtype=y.dtype, length=n)


# ---------------------------------------------------------------- mel / onset
def _hz_to_mel(f):
    f = np.asanyarray(f)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, np.log(6.4) / 27.0
    if f.ndim:
        lt = f >= min_log_hz
        mels[lt] = min_log_mel + np.log(f[lt] / min_log_hz) / logstep
    elif f >= min_log_hz:
        mels = min_log_mel + np.log(f / min_log_hz) / logstep
    return mels


def _mel_to_hz(m):
    m = np.asanyarray(m)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, np.log(6.4) / 27.0
    if m.ndim:
        lt = m >= min_log_mel
        freqs[lt] = min_log_hz * np.exp(logstep * (m[lt] - min_log_mel))
    elif m >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (m - min_log_mel))
    return freqs


def _mel_filters(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, dtype=np.float32):
    """librosa.filters.mel with htk=False, norm='slaney'."""
    if fmax is None:
        fmax = float(sr) / 2
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


def _power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    S = np.asarray(S)
    log_spec = 10.0 * np.log10(np.maximum(amin, S))
    log_spec -= 10.0 * np.log10(np.maximum(amin, np.abs(ref)))
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


def _onset_strength(y, sr=22050, n_fft=2048, hop_length=512, lag=1, aggregate=np.mean, center=True):
    """librosa.onset.onset_strength with the default mel feature (n_mels=128, fmax=sr/2)."""
    S = np.abs(_stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    mel = np.einsum("...ft,mf->...mt", S, _mel_filters(sr, n_fft, fmax=0.5 * sr), optimize=True)
    S = np.atleast_2d(_power_to_db(np.abs(mel)))
    env = np.maximum(0.0, S[..., lag:] - S[..., :-lag])
    env = aggregate(env, axis=-2)[..., None, :]
    pad = lag + (n_fft // (2 * hop_length) if center else 0)
    env = np.pad(env, [(0, 0)] * (env.ndim - 1) + [(int(pad), 0)])
    if center:
        env = env[..., :S.shape[-1]]
    return env[..., 0, :]


def _float_mean_seq(x, starts, ends):
    """numba np.mean semantics: sequential sum in x.dtype, / count, cast back to x.dtype."""
    n = len(x)
    L = ends - starts
    acc = np.zeros(len(starts), dtype=x.dtype)
    for o in range(int(L.max()) if len(L) else 0):
        m = o < L
        acc = np.where(m, acc + x[np.minimum(starts + o, n - 1)], acc)
    return (acc.astype(np.float64) / L).astype(x.dtype)


def _peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
    """librosa.util.peak_pick (sparse, 1-d), same float32 threshold arithmetic."""
    pre_max, post_max, pre_avg, post_avg, wait = (int(np.ceil(v)) for v in
                                                  (pre_max, post_max, pre_avg, post_avg, wait))
    delta = np.float32(delta)
    n = len(x)
    idx = np.arange(n)
    xp = np.pad(x, (pre_max, post_max), constant_values=-np.inf)
    mx = np.lib.stride_tricks.sliding_window_view(xp, pre_max + post_max)[:n].max(axis=1)
    avg = _float_mean_seq(x, np.maximum(0, idx - pre_avg), np.minimum(idx + post_avg, n))
    ok = (x == mx) & (x >= avg + delta)
    peaks = []
    a0 = _float_mean_seq(x, np.array([0]), np.array([min(post_avg, n)]))
    first = bool(x[0] >= np.max(x[:min(post_max, n)])) and bool((x[:1] >= a0 + delta)[0])
    nxt = 1
    if first:
        peaks.append(0)
        nxt = wait + 1
    for c in np.flatnonzero(ok[1:]) + 1:
        if c >= nxt:
            peaks.append(int(c))
            nxt = c + wait + 1
    return np.asarray(peaks, dtype=np.int64)


def onset_detect(*, y, sr=22050, hop_length=512, backtrack=False, units="frames",
                 normalize=True, **kwargs):
    """librosa.onset.onset_detect (sparse output; backtrack not supported)."""
    if backtrack:
        raise NotImplementedError("lc_dsp.onset_detect: backtrack=True is not ported")
    env = _onset_strength(y, sr=sr, hop_length=hop_length)
    if normalize:
        env = env - np.min(env, keepdims=True, axis=-1)
        env /= np.max(env, keepdims=True, axis=-1) + _tiny(env)
    if not env.any() or not np.all(np.isfinite(env)):
        onsets = np.array([], dtype=int)
    else:
        kwargs.setdefault("pre_max", 0.03 * sr // hop_length)
        kwargs.setdefault("post_max", 0.00 * sr // hop_length + 1)
        kwargs.setdefault("pre_avg", 0.10 * sr // hop_length)
        kwargs.setdefault("post_avg", 0.10 * sr // hop_length + 1)
        kwargs.setdefault("wait", 0.03 * sr // hop_length)
        kwargs.setdefault("delta", 0.07)
        onsets = _peak_pick(env, **kwargs)
    if units == "frames":
        return onsets
    if units == "samples":
        return (onsets * hop_length).astype(int)
    if units == "time":
        return (onsets * hop_length).astype(int) / float(sr)
    raise ValueError(f"Invalid unit type: {units}")


# ---------------------------------------------------------------- tempo / beats
def _tempo(onset_envelope, sr=22050, hop_length=512, start_bpm=120.0, std_bpm=1.0,
           ac_size=8.0, max_tempo=320.0):
    """librosa.feature.tempo (static, log-normal prior, mean aggregate)."""
    win_length = int(np.asarray(ac_size * sr).astype(int) // hop_length)
    n = onset_envelope.shape[-1]
    oenv = np.pad(onset_envelope, (win_length // 2, win_length // 2), mode="linear_ramp",
                  end_values=[0, 0])
    frames = _frame(oenv, win_length, 1)[..., :n]
    tg = _normalize_inf(_autocorrelate(frames * _window(win_length)[:, None], axis=-2), axis=-2)
    tg = np.mean(tg, axis=-1, keepdims=True)
    bpms = np.zeros(win_length, dtype=np.float64)
    bpms[0] = np.inf
    bpms[1:] = 60.0 * sr / (hop_length * np.arange(1.0, win_length))
    logprior = -0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm) ** 2
    if max_tempo is not None:
        logprior[:int(np.argmax(bpms < max_tempo))] = -np.inf
    best = np.argmax(np.log1p(1e6 * tg) + logprior[:, None], axis=-2)
    return np.take(bpms, best)


def _beat_local_score(oenv, fpb):
    oenv = oenv.astype(np.float64)
    N = len(oenv)
    window = np.exp(-0.5 * (np.arange(-fpb, fpb + 1) * 32.0 / fpb) ** 2)
    K = len(window)
    ls = np.zeros(N)
    i = np.arange(N)
    for k in range(K):  # same summation order as the numba kernel
        j = i + K // 2 - k
        v = (j >= 1) & (j < N)
        ls[v] += window[k] * oenv[j[v]]
    return ls


def _beat_track_dp(ls, fpb, tightness):
    N = len(ls)
    thresh = 0.01 * ls.max()
    h = int(np.round(fpb / 2))
    d = np.arange(h, int(2 * fpb) + 1)
    pen = float(np.float32(tightness)) * (np.log(d.astype(np.float64)) - np.log(fpb)) ** 2
    cum = np.zeros(N)
    loc_best = np.full(N, -1, dtype=np.int64)
    step = max(h, 1)
    for s in range(0, N, step):
        ii = np.arange(s, min(s + step, N))
        loc = ii[:, None] - d[None, :]
        valid = loc >= 0
        if h < 1:  # degenerate tempo: fall back to strict sequential order
            for i in ii:
                best, bl = -np.inf, -1
                for dd, p in zip(d, pen):
                    lo = i - dd
                    if lo < 0:
                        break
                    sc = cum[lo] - p
                    if sc > best:
                        best, bl = sc, lo
                cum[i] = ls[i] + best if bl >= 0 else ls[i]
                loc_best[i] = bl
            continue
        sc = np.where(valid, cum[np.maximum(loc, 0)] - pen[None, :], -np.inf)
        a = np.argmax(sc, axis=1)
        has = valid[np.arange(len(ii)), a] & (sc[np.arange(len(ii)), a] > -np.inf)
        bl = np.where(has, loc[np.arange(len(ii)), a], -1)
        cum[ii] = np.where(has, ls[ii] + sc[np.arange(len(ii)), a], ls[ii])
        loc_best[ii] = bl
    backlink = loc_best.copy()
    above = np.flatnonzero(~(ls < thresh))
    first = above[0] if len(above) else N
    backlink[:first] = -1
    return backlink, cum


def _last_beat(cum):
    lmax = _localmax(cum)
    thr = 0.5 * np.median(cum[lmax]) if lmax.any() else 0.0
    for n in range(len(cum) - 1, -1, -1):
        if lmax[n] and cum[n] >= thr:
            return n
    return len(cum) - 1


def _trim_beats(ls, beats, trim):
    out = beats.copy()
    w = np.hanning(5)
    sb = np.convolve(ls[beats], w)[len(w) // 2:len(ls) + len(w) // 2]
    thr = 0.5 * ((sb ** 2).mean() ** 0.5) if trim else 0.0
    n = 0
    while n < len(ls) and ls[n] <= thr:
        out[n] = False
        n += 1
    n = len(ls) - 1
    while n >= 0 and ls[n] <= thr:
        out[n] = False
        n -= 1
    return out


def beat_track(*, y, sr=22050, hop_length=512, start_bpm=120.0, tightness=100, trim=True,
               bpm=None, units="frames"):
    """librosa.beat.beat_track (sparse). Returns (tempo ndarray shape (1,), beats)."""
    oenv = _onset_strength(y, sr=sr, hop_length=hop_length, aggregate=np.median)
    if not oenv.any():
        return (0.0, np.array([], dtype=int))
    if bpm is None:
        bpm = _tempo(oenv, sr=sr, hop_length=hop_length, start_bpm=start_bpm)
    _bpm = np.atleast_1d(bpm).astype(np.float64)
    if np.any(_bpm <= 0):
        raise ValueError(f"bpm={bpm} must be strictly positive")
    fpb = float(np.round(float(sr) / hop_length * 60.0 / _bpm[0]))
    norm = oenv / (oenv.std(ddof=1, axis=-1, keepdims=True) + _tiny(oenv))
    ls = _beat_local_score(norm, fpb)
    backlink, cum = _beat_track_dp(ls, fpb, tightness)
    beats = np.zeros(len(oenv), dtype=bool)
    n = _last_beat(cum)
    while n >= 0:
        beats[n] = True
        n = backlink[n]
    beats = np.flatnonzero(_trim_beats(ls, beats, trim))
    if units == "frames":
        return (bpm, beats)
    if units == "samples":
        return (bpm, (beats * hop_length).astype(int))
    if units == "time":
        return (bpm, (beats * hop_length).astype(int) / float(sr))
    raise ValueError(f"Invalid unit type: {units}")


# ---------------------------------------------------------------- rms
def rms(*, y, frame_length=2048, hop_length=512, center=True, pad_mode="constant", dtype=np.float32):
    """librosa.feature.rms from a time series. Returns shape (..., 1, n_frames)."""
    if center:
        y = np.pad(y, [(0, 0)] * (y.ndim - 1) + [(int(frame_length // 2),) * 2], mode=pad_mode)
    x = _frame(y, frame_length, hop_length)
    power = np.mean(np.square(x, dtype=dtype), axis=-2, keepdims=True)
    return np.sqrt(power)


# ---------------------------------------------------------------- pYIN
def _cmnd(y_frames, min_period, max_period):
    acf = _autocorrelate(y_frames, max_size=max_period + 1, axis=-2)
    yin = np.square(y_frames)
    np.cumsum(yin, out=yin, axis=-2)
    k = slice(1, max_period + 1)
    yin[..., 0, :] = 0
    yin[..., k, :] = 2 * (acf[..., 0:1, :] - acf[..., k, :]) - yin[..., :k.stop - 1, :]
    num = yin[..., min_period:max_period + 1, :]
    k_range = np.r_[k][:, None]
    cm = np.cumsum(yin[..., k, :], axis=-2) / k_range
    den = cm[..., min_period - 1:max_period, :]
    return num / (den + _tiny(den))


def _parabolic_shifts(x):
    """Numba stencil semantics: (x1 + x-1) in x.dtype, then float64 math."""
    out = np.zeros_like(x)
    xm, x0, xp = x[:-2], x[1:-1], x[2:]
    a = (xp + xm).astype(np.float64) - 2 * x0.astype(np.float64)
    b = (xp - xm).astype(np.float64) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(np.abs(b) >= np.abs(a), 0.0, -b / a)
    out[1:-1] = s
    return out


def _pyin_obs(yin_frames, shifts, sr, thresholds, boltz, beta_probs, no_trough_prob,
              min_period, fmin, n_pitch_bins, n_bins_per_semitone):
    yin_probs = np.zeros_like(yin_frames)
    for i, yf in enumerate(yin_frames.T):
        is_trough = _localmin(yf)
        is_trough[0] = yf[0] < yf[1]
        (ti,) = np.nonzero(is_trough)
        if len(ti) == 0:
            continue
        th = yf[ti]
        tt = np.less.outer(th, thresholds[1:])
        pos = np.cumsum(tt, axis=0) - 1
        n_tr = np.count_nonzero(tt, axis=0)
        # scipy.stats.boltzmann.pmf(pos, boltz, n_tr), valid where tt is True
        prior = np.zeros(tt.shape)
        if tt.any():
            Nv = np.broadcast_to(n_tr, tt.shape)[tt]
            kv = pos[tt]
            fact = (1 - np.exp(-boltz)) / (1 - np.exp(-boltz * Nv))
            prior[tt] = np.clip(fact * np.exp(-boltz * kv), 0, 1)
        probs = prior.dot(beta_probs)
        gm = np.argmin(th)
        nbelow = np.count_nonzero(~tt[gm, :])
        probs[gm] += no_trough_prob * np.sum(beta_probs[:nbelow])
        yin_probs[ti, i] = probs
    yp, fi = np.nonzero(yin_probs)
    pc = min_period + yp
    pc = pc + shifts[yp, fi]
    f0c = sr / pc
    bi = 12 * n_bins_per_semitone * np.log2(f0c / fmin)
    bi = np.clip(np.round(bi), 0, n_pitch_bins).astype(int)
    obs = np.zeros((2 * n_pitch_bins, yin_frames.shape[1]))
    obs[bi, fi] = yin_probs[yp, fi]
    vp = np.clip(np.sum(obs[:n_pitch_bins, :], axis=0, keepdims=True), 0, 1)
    obs[n_pitch_bins:, :] = (1 - vp) / n_pitch_bins
    return obs, vp


def _transition_local(n_states, width):
    T = np.zeros((n_states, n_states), dtype=np.float64)
    row0 = _pad_center(scipy.signal.get_window("triangle", width, fftbins=False), n_states)
    for i in range(n_states):
        r = np.roll(row0, n_states // 2 + i + 1)
        r[min(n_states, i + width // 2 + 1):] = 0
        r[:max(0, i - width // 2)] = 0
        T[i] = r
    T /= T.sum(axis=1, keepdims=True)
    return T


def _viterbi(prob, transition, p_init):
    """librosa.sequence.viterbi (exact argmax/tie semantics) for a sparse transition.

    Entries with transition == 0 all share log(eps); those are handled with one
    sorted pass per step instead of a dense n x n max, so the result is identical
    to the dense numba kernel."""
    n_states, n_steps = prob.shape
    eps = _tiny(prob)
    log_trans = np.log(transition + eps)
    log_prob = np.log(prob + eps).T
    log_p_init = np.log(p_init + eps)
    c = np.log(0.0 + eps)
    nz = log_trans != c  # in-band entries (k -> j)
    cnt = nz.sum(axis=0)
    M = int(cnt.max())
    # per target j: sorted source indices, padded
    src = np.zeros((n_states, M), dtype=np.int64)
    srcv = np.zeros((n_states, M), dtype=bool)
    for j in range(n_states):
        r = np.flatnonzero(nz[:, j])
        src[j, :len(r)] = r
        srcv[j, :len(r)] = True
    lt_band = np.where(srcv, log_trans[src, np.arange(n_states)[:, None]], 0.0)
    big = np.iinfo(np.int64).max
    src_masked = np.where(srcv, src, big)
    nzT = nz.T  # [j, k]
    has_out = cnt < n_states
    top = min(M + 1, n_states)
    value = log_prob[0] + log_p_init
    ptr = np.zeros((n_steps, n_states), dtype=np.int64)
    jj = np.arange(n_states)[:, None]
    for t in range(1, n_steps):
        cand = np.where(srcv, value[src] + lt_band, -np.inf)
        bv = cand.max(axis=1)
        bi = np.where(cand == bv[:, None], src_masked, big).min(axis=1)
        # best source outside the band: highest value + c, smallest index on ties
        w = value + c
        order = np.argsort(-w, kind="stable")[:top]
        inband = nzT[jj, order[None, :]]
        first = np.argmax(~inband, axis=1)
        ok = has_out & ~inband[np.arange(n_states), first]
        oi = order[first]
        ov = np.where(ok, w[oi], -np.inf)
        take = ok & ((ov > bv) | ((ov == bv) & (oi < bi)))
        bi = np.where(take, oi, bi)
        bv = np.where(take, ov, bv)
        ptr[t] = bi
        value = log_prob[t] + bv
    state = np.zeros(n_steps, dtype=np.int64)
    state[-1] = np.argmax(value)
    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]
    return state


def pyin(y, *, fmin, fmax, sr=22050, frame_length=2048, hop_length=None, n_thresholds=100,
         beta_parameters=(2, 18), boltzmann_parameter=2, resolution=0.1,
         max_transition_rate=35.92, switch_prob=0.01, no_trough_prob=0.01,
         fill_na=np.nan, center=True, pad_mode="constant"):
    """librosa.pyin for a mono signal. Returns (f0, voiced_flag, voiced_prob)."""
    if fmin is None or fmax is None:
        raise ValueError('both "fmin" and "fmax" must be provided')
    if fmax > sr / 2:
        raise ValueError(f"fmax={fmax:.3f} cannot exceed Nyquist frequency {sr / 2}")
    if fmin >= fmax:
        raise ValueError(f"fmin={fmin:.3f} must be less than fmax={fmax:.3f}")
    if fmin <= 0:
        raise ValueError(f"fmin={fmin:.3f} must be strictly positive")
    if sr / fmin >= frame_length - 1:
        raise ValueError(f"fmin={fmin:.3f} is too small for frame_length={frame_length} and sr={sr}")
    if y.ndim != 1:
        raise ValueError("lc_dsp.pyin expects a mono (1-d) signal")
    if hop_length is None:
        hop_length = frame_length // 4
    if not np.all(np.isfinite(y)):
        raise ValueError("Audio buffer is not finite everywhere")
    if center:
        y = np.pad(y, (frame_length // 2, frame_length // 2), mode=pad_mode)
    y_frames = _frame(y, frame_length, hop_length)
    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - 1)
    yin_frames = _cmnd(y_frames, min_period, max_period)
    shifts = _parabolic_shifts(yin_frames)
    thresholds = np.linspace(0, 1, n_thresholds + 1)
    beta_probs = np.diff(scipy.stats.beta.cdf(thresholds, beta_parameters[0], beta_parameters[1]))
    nbps = int(np.ceil(1.0 / resolution))
    n_bins = int(np.floor(12 * nbps * np.log2(fmax / fmin))) + 1
    obs, vp = _pyin_obs(yin_frames, shifts, sr, thresholds, boltzmann_parameter, beta_probs,
                        no_trough_prob, min_period, fmin, n_bins, nbps)
    max_semitones = round(max_transition_rate * 12 * hop_length / sr)
    width = max_semitones * nbps + 1
    L = _transition_local(n_bins, width)
    t_switch = np.array([[1 - switch_prob, switch_prob], [switch_prob, 1 - switch_prob]])
    transition = np.kron(t_switch, L)
    p_init = np.ones(2 * n_bins) / (2 * n_bins)
    states = _viterbi(obs, transition, p_init)
    freqs = fmin * 2 ** (np.arange(n_bins) / (12 * nbps))
    f0 = freqs[states % n_bins]
    voiced = states < n_bins
    if fill_na is not None:
        f0[~voiced] = fill_na
    return f0, voiced, vp[0]


# ---------------------------------------------------------------- notes
_NOTE_RE = re.compile(r"^(?P<note>[A-Ga-g])(?P<accidental>[#♯𝄪b!♭𝄫♮n]*)"
                      r"(?P<octave>[+-]?\d+)?(?P<cents>[+-]\d+)?$")
_PITCH = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACC = {"#": 1, "": 0, "b": -1, "!": -1, "♯": 1, "𝄪": 2, "♭": -1, "𝄫": -2, "♮": 0}
_NOTES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTES_SHARP_U = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]


def _note_to_midi(note):
    if not isinstance(note, str):
        return np.array([_note_to_midi(n) for n in note])
    m = _NOTE_RE.match(note)
    if not m:
        raise ValueError(f"Improper note format: {note:s}")
    offset = np.sum([_ACC[o] for o in m.group("accidental")])
    octave = int(m.group("octave")) if m.group("octave") else 0
    cents = int(m.group("cents")) * 1e-2 if m.group("cents") else 0
    return int(np.round(12 * (octave + 1) + _PITCH[m.group("note").upper()] + offset + cents))


def note_to_hz(note):
    """librosa.note_to_hz, e.g. 'C2', 'A#3', 'Bb4' (or a list of them)."""
    return 440.0 * (2.0 ** ((np.asanyarray(_note_to_midi(note)) - 69.0) / 12.0))


def hz_to_midi(frequencies):
    return 12 * (np.log2(np.asanyarray(frequencies)) - np.log2(440.0)) + 69


def midi_to_note(midi, *, octave=True, cents=False, unicode=True):
    """librosa.midi_to_note with key='C:maj' (sharps)."""
    names = _NOTES_SHARP_U if unicode else _NOTES_SHARP

    def one(m):
        n = int(np.round(m))
        c = int(100 * np.around(m - n, 2))
        s = names[n % 12]
        if octave:
            s = "{:s}{:0d}".format(s, int(n / 12) - 1)
        if cents:
            s = f"{s:s}{c:+02d}"
        return s

    if np.ndim(midi) == 0:
        return one(midi)
    return np.vectorize(one)(np.asarray(midi))
