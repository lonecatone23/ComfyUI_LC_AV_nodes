"""The parts of OpenCLIP's CLIP model (open_clip_torch 3.3.0, MIT, see LICENSE) that MMAudio uses with
apple/DFN5B-CLIP-ViT-H-14-384: a ViT-H/14 image tower (QuickGELU, class-token pooling) and the text transformer.

Module and parameter names match open_clip, so its state dict loads as is. The arithmetic is open_clip's own
(model.py CLIP, transformer.py VisionTransformer / TextTransformer / Transformer / ResidualAttentionBlock with
cast_dtype=None, i.e. the plain LayerNorm that casts back to the input dtype). Options this model never uses
(attentional pooling, patch dropout, layer scale, custom blocks, HF/timm towers, grad checkpointing) are left out.
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0, act_layer=QuickGELU):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ls_1 = nn.Identity()
        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model)),
        ]))
        self.ls_2 = nn.Identity()

    def get_weight_dtype(self) -> torch.dtype:
        return self.mlp.c_fc.weight.dtype

    def attention(self, q_x: torch.Tensor, attn_mask=None):
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, q_x, q_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, q_x: torch.Tensor, attn_mask=None):
        x = q_x + self.ls_1(self.attention(self.ln_1(q_x), attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: float = 4.0, act_layer=QuickGELU):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = True
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer) for _ in range(layers)])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].get_weight_dtype()

    def forward(self, x: torch.Tensor, attn_mask=None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size: int, patch_size: int, width: int, layers: int, heads: int, mlp_ratio: float,
                 output_dim: int, act_layer=QuickGELU):
        super().__init__()
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.patch_dropout = nn.Identity()
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x)
        pooled = x[:, 0]  # pool_type 'tok'
        return pooled @ self.proj


class CLIP(nn.Module):
    """open_clip.CLIP for a ViT vision tower and the default text tower (no HF/timm towers)."""

    def __init__(self, embed_dim: int, vision_cfg: dict, text_cfg: dict, quick_gelu: bool = False,
                 init_logit_scale: float = np.log(1 / 0.07), nonscalar_logit_scale: bool = False, **_ignored):
        super().__init__()
        act_layer = QuickGELU if quick_gelu else nn.GELU
        v = dict(vision_cfg)
        self.visual = VisionTransformer(image_size=v["image_size"], patch_size=v["patch_size"], width=v["width"],
                                        layers=v["layers"], heads=v["width"] // v.get("head_width", 64),
                                        mlp_ratio=v.get("mlp_ratio", 4.0), output_dim=embed_dim, act_layer=act_layer)
        t = dict(text_cfg)
        width = t.get("width", 512)
        self.context_length = t.get("context_length", 77)
        self.vocab_size = t.get("vocab_size", 49408)
        self.transformer = Transformer(width, t.get("layers", 12), t.get("heads", 8), t.get("mlp_ratio", 4.0),
                                       act_layer=act_layer)
        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.ln_final = LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # causal mask, -inf above the diagonal
        self.register_buffer("attn_mask", mask, persistent=False)
        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]  # pool_type 'argmax'
        x = x @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x
