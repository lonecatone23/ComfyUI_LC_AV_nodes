"""The few outside pieces ChatterBox VC uses, so LC AV needs no omegaconf, diffusers, conformer, librosa, torchaudio
or onnx for it.

- DictConfig / OmegaConf.create: a dict with attribute access (the configs are only read).
- get_activation, GELU, Attention (+ AttnProcessor2_0's math), LoRACompatibleLinear, maybe_allow_in_graph: ported
  from diffusers 0.39.0 (https://github.com/huggingface/diffusers, Apache-2.0), only the paths ChatterBox runs. The
  parameter names match, so the checkpoint loads unchanged.
- librosa mel filters: LC AV's lc_dsp port (librosa 0.11, ISC). torchaudio Resample / kaldi.fbank: the ports in
  vendor/indextts/utils/ta_compat.py (torchaudio 2.11.0, BSD 2-Clause).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...lc_dsp import _mel_filters
from ..indextts.utils import ta_compat


# ---------------------------------------------------------------- omegaconf
class DictConfig(dict):
    """Read-only config: cfg.key and cfg["key"] both work."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return DictConfig(v) if isinstance(v, dict) and not isinstance(v, DictConfig) else v


OmegaConf = SimpleNamespace(create=lambda d: DictConfig(d))


# ---------------------------------------------------------------- librosa / torchaudio
def librosa_mel_fn(*, sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm="slaney", dtype=np.float32):
    if htk or norm != "slaney":
        raise NotImplementedError("LC AV: only librosa's default mel filters (slaney) are ported.")
    return _mel_filters(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=dtype)


librosa = SimpleNamespace(filters=SimpleNamespace(mel=librosa_mel_fn))


class _Resample(nn.Module):
    """torchaudio.transforms.Resample(orig, new)."""

    def __init__(self, orig_freq, new_freq):
        super().__init__()
        self.orig, self.new = int(orig_freq), int(new_freq)

    def forward(self, x):
        return ta_compat.resample(x, self.orig, self.new)


ta = SimpleNamespace(transforms=SimpleNamespace(Resample=_Resample))
Kaldi = SimpleNamespace(fbank=ta_compat.fbank)


# ---------------------------------------------------------------- conformer (block type ChatterBox doesn't use)
class ConformerBlock(nn.Module):
    def __init__(self, *a, **k):
        raise NotImplementedError("LC AV: the conformer block type is not part of the vendored ChatterBox.")


# ---------------------------------------------------------------- diffusers
_ACT = {"swish": nn.SiLU, "silu": nn.SiLU, "mish": nn.Mish, "gelu": nn.GELU, "relu": nn.ReLU}


def get_activation(act_fn: str) -> nn.Module:
    act_fn = act_fn.lower()
    if act_fn not in _ACT:
        raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(_ACT)}")
    return _ACT[act_fn]()


def maybe_allow_in_graph(cls):
    return cls


LoRACompatibleLinear = nn.Linear


class GELU(nn.Module):
    """diffusers.models.activations.GELU: linear projection, then GELU."""

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, hidden_states):
        return F.gelu(self.proj(hidden_states), approximate=self.approximate)


def _unused(name):
    class _U(nn.Module):
        def __init__(self, *a, **k):
            raise NotImplementedError(f"LC AV: diffusers {name} is not part of the vendored ChatterBox.")
    _U.__name__ = name
    return _U


GEGLU, ApproximateGELU, AdaLayerNorm, AdaLayerNormZero = (_unused(n) for n in
                                                          ("GEGLU", "ApproximateGELU", "AdaLayerNorm", "AdaLayerNormZero"))


class Attention(nn.Module):
    """diffusers.models.attention_processor.Attention with AttnProcessor2_0, for the arguments ChatterBox passes
    (no group / spatial / qk norm, no residual, rescale 1). Same parameter names: to_q, to_k, to_v, to_out.0."""

    def __init__(self, query_dim: int, cross_attention_dim: int | None = None, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0, bias: bool = False, upcast_attention: bool = False, out_bias: bool = True,
                 **_unused_kw):
        super().__init__()
        inner = dim_head * heads
        kv = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner, bias=bias)
        self.to_k = nn.Linear(kv, inner, bias=bias)
        self.to_v = nn.Linear(kv, inner, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner, query_dim, bias=out_bias), nn.Dropout(dropout)])

    def prepare_attention_mask(self, attention_mask, target_length, batch_size):
        if attention_mask is None:
            return attention_mask
        if attention_mask.shape[-1] != target_length:
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
        if attention_mask.shape[0] < batch_size * self.heads:
            attention_mask = attention_mask.repeat_interleave(self.heads, dim=0,
                                                              output_size=attention_mask.shape[0] * self.heads)
        return attention_mask

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **_kw):
        batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None
                                          else encoder_hidden_states.shape)
        if attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])
        query = self.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        head_dim = key.shape[-1] // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0,
                                                       is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim).to(query.dtype)
        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)
