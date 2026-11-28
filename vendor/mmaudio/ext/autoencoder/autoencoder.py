from typing import Literal, Optional

import torch
import torch.nn as nn

from .vae import VAE, get_my_vae
from ..bigvgan_v2.bigvgan import BigVGAN as BigVGANv2
from ...model.utils.distributions import DiagonalGaussianDistribution


class AutoEncoderModule(nn.Module):

    def __init__(self,
                 *,
                 vae_state_dict: dict,
                 vocoder_folder: Optional[str] = None,
                 mode: Literal['44k'] = '44k',
                 need_vae_encoder: bool = True):
        # LC_AV: takes the VAE state dict (the caller reads the safetensors file) and the local BigVGAN v2 folder
        # (config.json + bigvgan_generator.pt, see BigVGAN.from_folder) instead of checkpoint paths and a hub download.
        # The 16k mode (BigVGAN v1) is not vendored.
        super().__init__()
        self.vae: VAE = get_my_vae(mode).eval()
        self.vae.load_state_dict(vae_state_dict)
        self.vae.remove_weight_norm()

        if mode == '44k':
            self.vocoder = BigVGANv2.from_folder(vocoder_folder).eval()
            self.vocoder.remove_weight_norm()
        else:
            raise ValueError(f'Unknown mode: {mode}')

        for param in self.parameters():
            param.requires_grad = False

        if not need_vae_encoder:
            del self.vae.encoder

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return self.vae.encode(x)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    @torch.inference_mode()
    def vocode(self, spec: torch.Tensor) -> torch.Tensor:
        return self.vocoder(spec)
