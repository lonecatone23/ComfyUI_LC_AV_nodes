# Vendored code: MMAudio (inference only)

| | |
| --- | --- |
| Upstream | https://github.com/hkchengrex/MMAudio (Ho Kei Cheng et al., "MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis", CVPR 2025) |
| Taken from | the `mmaudio/` folder of kijai's https://github.com/kijai/ComfyUI-MMAudio, revision `8eaeb72edc3aaf2059b57f2d96a1f6f689f19ae2` (2026-02-01) |
| License (code) | MIT. MMAudio: `LICENSE` (Copyright (c) 2024 Sony Research Inc.). ComfyUI-MMAudio: `LICENSE.ComfyUI-MMAudio` (Copyright (c) 2025 Jukka Seppänen). BigVGAN v2: `ext/bigvgan_v2/LICENSE` (NVIDIA, MIT) plus `incl_licenses/`. Synchformer: `ext/synchformer/LICENSE` (Vladimir Iashin, MIT). OpenCLIP parts and the CLIP BPE vocabulary: `clip/LICENSE` (MIT) |
| License (weights) | MMAudio weights (`mmaudio_large_44k_v2`, VAE, Synchformer conversion): **CC BY-NC 4.0, non-commercial**. Apple DFN5B-CLIP: Apple's own license (see the apple/DFN5B-CLIP-ViT-H-14-384 model card). BigVGAN v2: MIT. No weights are in this repository: they download on first use. |

Used by LC Foley (`lc_foley.py`). The point of this copy is a clean install: MMAudio's usual requirements
(librosa -> numba -> numpy pins, torchdiffeq, timm, omegaconf, open_clip_torch, ftfy, accelerate, huggingface_hub,
torchaudio) are not needed. Everything here runs on torch, torchvision, numpy, einops, regex and pyyaml, which every
ComfyUI install has.

Only the 44.1 kHz inference path is vendored: no training code, no 16 kHz mode (BigVGAN v1, mel converter), no
audio encoding.

## Copied unchanged (byte for byte from the revision above)

`__init__.py`, `model/__init__.py`, `model/embeddings.py`, `model/low_level.py`, `model/networks.py`,
`model/sequence_config.py`, `model/transformer_layers.py`, `model/utils/__init__.py`, `model/utils/distributions.py`,
`ext/__init__.py`, `ext/rotary_embeddings.py`, `ext/autoencoder/__init__.py`, `ext/autoencoder/edm2_utils.py`,
`ext/autoencoder/vae_modules.py`, `ext/bigvgan_v2/__init__.py`, `ext/bigvgan_v2/activations.py`, `ext/bigvgan_v2/env.py`,
`ext/bigvgan_v2/utils.py`, `ext/bigvgan_v2/alias_free_activation/torch/*.py`, `ext/synchformer/__init__.py`,
`ext/synchformer/divided_224_16x4.yaml`, and the license files.

## Changed

| File | Upstream | Here |
| --- | --- | --- |
| `model/flow_matching.py` | euler loop plus an adaptive mode through `torchdiffeq.odeint`; ComfyUI and tqdm progress bars | the same euler loop only; a per-step callback (progress bar and Cancel live in the node) |
| `model/utils/features_utils.py` | `open_clip.get_tokenizer`, `MelConverter` (librosa mel filters), BigVGAN v1 import, `compile()`, `encode_audio()` | the vendored `clip.tokenizer.SimpleTokenizer`; the mel converter, `encode_audio`, `compile` and the v1 import removed (generation never uses them). `torchvision.transforms.Normalize` kept |
| `eval_utils.py` | `generate()` plus `ModelConfig` | `generate()` only, with the callback passed through |
| `ext/autoencoder/autoencoder.py` | imports BigVGAN v1 (omegaconf) for a type hint | import removed |
| `ext/autoencoder/vae.py` | statistics buffers created with `.cuda()` | created on the CPU (moved with the module), so loading works on any device |
| `ext/bigvgan_v2/bigvgan.py` | `huggingface_hub.PyTorchModelHubMixin` / `hf_hub_download`; optional CUDA kernel compiled with nvcc at run time | plain `nn.Module`; torch activation only; `BigVGAN.from_folder()` reads `config.json` and `bigvgan_generator.pt` with `torch.load(weights_only=True)`. The `.py` files in the downloaded model folder are never imported |
| `ext/synchformer/motionformer.py` | `OmegaConf.load`, `timm.layers.trunc_normal_`, config/checkpoint download helpers | `yaml.safe_load` into a small attribute dict; `torch.nn.init.trunc_normal_`; no downloads (the config ships here) |
| `ext/synchformer/video_model_builder.py` | `timm.layers.trunc_normal_` | `torch.nn.init.trunc_normal_` (initialisation only, the weights load over it) |
| `ext/synchformer/vit_helper.py` | `timm.layers.to_2tuple`; `load_pretrained()` (torch.hub download) | a two-line `to_2tuple`; `load_pretrained` removed |
| `ext/synchformer/synchformer.py` | a `__main__` test block | removed |
| `ext/bigvgan_v2/alias_free_activation/__init__.py` | (absent) | empty, so the folder is a package |

## Written for this pack

- `clip/model.py`: the parts of OpenCLIP's `CLIP` (open_clip_torch 3.3.0) used with DFN5B-CLIP-ViT-H-14-384: the
  ViT-H/14 image tower and the text transformer, with open_clip's module names (its state dict loads as is) and its
  arithmetic (`nn.MultiheadAttention`, LayerNorm cast back to the input dtype, QuickGELU, class-token pooling).
- `clip/tokenizer.py`: open_clip's `SimpleTokenizer` (lower-case cleaning, 77 tokens) with `ftfy.fix_text` replaced by
  a small cleanup (NFC, curly quotes, ligatures, full-width ASCII, control characters). Plain ASCII prompts give the
  same tokens. `clip/bpe_simple_vocab_16e6.txt.gz` is the OpenAI CLIP vocabulary shipped with open_clip.

`lc_foley.py` does what kijai's loader nodes do (parameters built on the meta device, then set from the safetensors
files at the chosen dtype) without accelerate. Checked against kijai's MMAudioModelLoader + MMAudioFeatureUtilsLoader +
MMAudioSampler (fp16, 44k, same frames, prompt, seed, steps and cfg): the audio is bit-identical.

## Model files (not in this repository)

All in `models/mmaudio` (the folder kijai's pack uses, so existing downloads are reused):

| File | Source |
| --- | --- |
| `mmaudio_large_44k_v2_fp16.safetensors` | https://huggingface.co/Kijai/MMAudio_safetensors |
| `mmaudio_vae_44k_fp16.safetensors` | https://huggingface.co/Kijai/MMAudio_safetensors |
| `mmaudio_synchformer_fp16.safetensors` | https://huggingface.co/Kijai/MMAudio_safetensors |
| `apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors` | https://huggingface.co/Kijai/MMAudio_safetensors |
| `nvidia/bigvgan_v2_44khz_128band_512x/config.json`, `bigvgan_generator.pt` | https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x (only these two files) |
