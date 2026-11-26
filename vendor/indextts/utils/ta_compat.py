"""The few torchaudio functions IndexTTS-2 uses, ported to plain torch so torchaudio is not needed.

Ported from torchaudio 2.11.0 (https://github.com/pytorch/audio), BSD 2-Clause License,
Copyright (c) 2017 Facebook Inc. (Soumith Chintala). All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met: 1. Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. 2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or other materials provided with the
distribution. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Ported (same arithmetic, same defaults): transforms.Resample (sinc_interp_hann, kernel built in float64 on the CPU and
kept as float32), compliance.kaldi.fbank (default povey window, no VTLN), functional._hz_to_mel / _mel_to_hz.
"""

import math

import torch
from torch import Tensor

# ---------------------------------------------------------------- resample (transforms.Resample)
_KERNELS = {}


def _sinc_kernel(orig_freq, new_freq, gcd, lowpass_filter_width=6, rolloff=0.99, dtype=None):
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx = torch.arange(-width, width + orig_freq, dtype=dtype if dtype is not None else torch.float64)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=dtype)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale
    return (kernels.to(dtype=torch.float32) if dtype is None else kernels), width


def resample(waveform: Tensor, orig_freq: int, new_freq: int, functional: bool = False) -> Tensor:
    """torchaudio.transforms.Resample(orig_freq, new_freq)(waveform), waveform [..., T] float.
    functional=True is torchaudio.functional.resample instead (kernel built in the waveform's dtype)."""
    orig_freq, new_freq = int(orig_freq), int(new_freq)
    if orig_freq == new_freq:
        return waveform
    gcd = math.gcd(orig_freq, new_freq)
    key = (orig_freq, new_freq, waveform.dtype if functional else None)
    if key not in _KERNELS:
        _KERNELS[key] = _sinc_kernel(orig_freq, new_freq, gcd, dtype=waveform.dtype if functional else None)
    kernel, width = _KERNELS[key]
    kernel = kernel.to(device=waveform.device, dtype=waveform.dtype)
    o, n = orig_freq // gcd, new_freq // gcd
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])
    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + o))
    out = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=o)
    out = out.transpose(1, 2).reshape(num_wavs, -1)
    target_length = torch.ceil(torch.as_tensor(n * length / o)).long()
    out = out[..., :target_length]
    return out.view(shape[:-1] + out.shape[-1:])


# ---------------------------------------------------------------- mel scale (functional)
def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))
    f_sp = 200.0 / 3
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: Tensor, mel_scale: str = "htk") -> Tensor:
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


# ---------------------------------------------------------------- kaldi fbank (compliance.kaldi)
_EPSILON = torch.tensor(torch.finfo(torch.float).eps)


def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _get_strided(waveform: Tensor, window_size: int, window_shift: int) -> Tensor:
    # snip_edges=True
    num_samples = waveform.size(0)
    strides = (window_shift * waveform.stride(0), waveform.stride(0))
    if num_samples < window_size:
        return torch.empty((0, 0), dtype=waveform.dtype, device=waveform.device)
    m = 1 + (num_samples - window_size) // window_shift
    return waveform.as_strided((m, window_size), strides)


def _get_log_energy(strided_input: Tensor, epsilon: Tensor, energy_floor: float) -> Tensor:
    log_energy = torch.max(strided_input.pow(2).sum(1), epsilon).log()
    if energy_floor == 0.0:
        return log_energy
    return torch.max(log_energy, torch.tensor(math.log(energy_floor), device=strided_input.device,
                                              dtype=strided_input.dtype))


def _mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def _mel_scale(freq: Tensor) -> Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()


def _get_mel_banks(num_bins, window_length_padded, sample_freq, low_freq, high_freq):
    # vtln_warp_factor == 1.0
    num_fft_bins = window_length_padded / 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = _mel_scale_scalar(low_freq)
    mel_high_freq = _mel_scale_scalar(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)
    b = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low_freq + b * mel_freq_delta
    center_mel = mel_low_freq + (b + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (b + 2.0) * mel_freq_delta
    mel = _mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    return torch.max(torch.zeros(1), torch.min(up_slope, down_slope))


def fbank(waveform: Tensor, num_mel_bins: int = 23, dither: float = 0.0, sample_frequency: float = 16000.0,
          frame_length: float = 25.0, frame_shift: float = 10.0, energy_floor: float = 1.0,
          low_freq: float = 20.0, high_freq: float = 0.0, preemphasis_coefficient: float = 0.97) -> Tensor:
    """torchaudio.compliance.kaldi.fbank with its defaults (channel -1 = first, povey window, power, log, snip_edges,
    remove_dc_offset, raw_energy, round_to_power_of_two, no energy column, no VTLN). waveform [C, T]."""
    if dither != 0.0:
        raise ValueError("this fbank port only covers dither=0")
    device, dtype = waveform.device, waveform.dtype
    epsilon = _EPSILON.to(device=device, dtype=dtype)
    waveform = waveform[0, :]
    window_shift = int(sample_frequency * frame_shift * 0.001)
    window_size = int(sample_frequency * frame_length * 0.001)
    padded_window_size = _next_power_of_2(window_size)
    if not 2 <= window_size <= len(waveform):
        raise ValueError(f"choose a window size {window_size} that is [2, {len(waveform)}]")

    strided = _get_strided(waveform, window_size, window_shift)
    strided = strided - torch.mean(strided, dim=1).unsqueeze(1)  # remove_dc_offset
    _get_log_energy(strided, epsilon, energy_floor)  # raw_energy (unused by fbank without use_energy)
    if preemphasis_coefficient != 0.0:
        offset = torch.nn.functional.pad(strided.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
        strided = strided - preemphasis_coefficient * offset[:, :-1]
    window = torch.hann_window(window_size, periodic=False, device=device, dtype=dtype).pow(0.85).unsqueeze(0)
    strided = strided * window
    if padded_window_size != window_size:
        strided = torch.nn.functional.pad(strided.unsqueeze(0), (0, padded_window_size - window_size),
                                          mode="constant", value=0).squeeze(0)

    spectrum = torch.fft.rfft(strided).abs().pow(2.0)
    mel = _get_mel_banks(num_mel_bins, padded_window_size, sample_frequency, low_freq, high_freq)
    mel = mel.to(device=device, dtype=dtype)
    mel = torch.nn.functional.pad(mel, (0, 1), mode="constant", value=0)
    mel = torch.mm(spectrum, mel.T)
    return torch.max(mel, epsilon).log()
