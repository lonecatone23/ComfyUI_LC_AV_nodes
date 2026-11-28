# Vendored code: MMAudio (inference only)

| | |
| --- | --- |
| MMAudio | https://github.com/hkchengrex/MMAudio, revision `974010a026c731054592d8f777218bd9d85a6c24` (2026-02-23), the `mmaudio/` folder. Ho Kei Cheng et al., "MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis", CVPR 2025 |
| BigVGAN v2 | MMAudio's own copy in `mmaudio/ext/bigvgan_v2` (its `activations.py`, `alias_free_activation/torch/filter.py` and `incl_licenses/` match NVIDIA's repo). Original: https://github.com/NVIDIA/BigVGAN, revision `7d2b454564a6c7d014227f635b7423881f14bdac` (v2.4, 2024-09-04) |
| Synchformer | MMAudio's own adaptation in `mmaudio/ext/synchformer` (its `divided_224_16x4.yaml` matches the original). Original: https://github.com/v-iashin/Synchformer, revision `b66668a1521d7567cc760e5544b2b5b53179b687` (2025-09-15) |
| OpenCLIP | https://github.com/mlfoundations/open_clip: `clip/` is written for this pack from open_clip_torch 3.3.0 (tag `v3.3.0`, `30573618fc375b12f094ef64cb3a1391cf611c45`); the BPE vocabulary is checked byte for byte against revision `34b17dc1eebc4eaeabbdc7a145111d8d758c0366` |
| License (code) | All MIT. MMAudio: `LICENSE` (Copyright (c) 2024 Sony Research Inc.). BigVGAN v2: `ext/bigvgan_v2/LICENSE` (Copyright (c) 2024 NVIDIA CORPORATION) plus `ext/bigvgan_v2/incl_licenses/` (the licenses of the code BigVGAN builds on). Synchformer: `ext/synchformer/LICENSE` (Copyright (c) 2024 Vladimir Iashin). OpenCLIP and the CLIP BPE vocabulary: `clip/LICENSE` (Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi, Ludwig Schmidt). Each license file is the committed file of its repository, byte for byte |
| License (weights) | MMAudio weights (`mmaudio_large_44k_v2`, VAE, Synchformer conversion): **CC BY-NC 4.0, non-commercial**. Apple DFN5B-CLIP: Apple's own license (see the apple/DFN5B-CLIP-ViT-H-14-384 model card). BigVGAN v2: MIT. No weights are in this repository: they download on first use. |

Used by LC Foley (`lc_foley.py`). The point of this copy is a clean install: MMAudio's usual requirements
(librosa -> numba -> numpy pins, torchdiffeq, timm, omegaconf, open_clip_torch, ftfy, accelerate, huggingface_hub,
torchaudio, av, colorlog) are not needed. Everything here runs on torch, torchvision, numpy, einops, regex and pyyaml,
which every ComfyUI install has.

Only the 44.1 kHz inference path is vendored: no training code, no 16 kHz mode (BigVGAN v1, mel converter), no
audio encoding.

## Import paths (every `.py` file)

MMAudio imports its own modules as `from mmaudio.x.y import ...`. Here each one is the shortest relative import from
the file's own package (`from .resample import ...`, `from ...model.utils.distributions import ...`). Nothing else
changes in a file that is not listed under "Changed".

## Copied unchanged (apart from import paths)

`__init__.py`, `model/__init__.py`, `model/embeddings.py`, `model/low_level.py`, `model/networks.py`,
`model/sequence_config.py`, `model/transformer_layers.py`, `model/utils/__init__.py`, `model/utils/distributions.py`,
`ext/__init__.py`, `ext/rotary_embeddings.py`, `ext/autoencoder/__init__.py`, `ext/autoencoder/edm2_utils.py`,
`ext/autoencoder/vae.py`, `ext/autoencoder/vae_modules.py`, `ext/bigvgan_v2/__init__.py`,
`ext/bigvgan_v2/activations.py`, `ext/bigvgan_v2/env.py`, `ext/bigvgan_v2/utils.py`,
`ext/bigvgan_v2/alias_free_activation/torch/*.py`, `ext/synchformer/__init__.py`, `ext/synchformer/divided_224_16x4.yaml`,
and the license files.

## Changed (LC AV)

| File | MMAudio | Here |
| --- | --- | --- |
| `eval_utils.py` | `generate()` plus `ModelConfig`, logging setup and video/image loaders (av, colorlog, PIL) | `generate()` only, run under `torch.inference_mode()`, with a per-step `callback` passed to the euler loop (progress bar and Cancel live in the node) |
| `model/flow_matching.py` | euler loop plus an adaptive mode through `torchdiffeq.odeint`; training helpers | the same euler loop only, calling `callback(step, total)` after each step |
| `model/utils/features_utils.py` | builds CLIP (`open_clip.create_model_from_pretrained`, hub download), Synchformer and the VAE + vocoder from checkpoint paths; `open_clip.get_tokenizer`; `MelConverter`, `encode_audio()`, `compile()` | takes the ready-made `tod`, `synchformer` and `clip_model` modules (lc_foley.py builds them from local safetensors files); the vendored `clip.tokenizer.SimpleTokenizer`; the mel converter, `encode_audio` and `compile` removed |
| `ext/autoencoder/autoencoder.py` | loads the VAE from a checkpoint path; BigVGAN v1 (16k) or `BigVGANv2.from_pretrained` (hub) | takes the VAE state dict and the local BigVGAN v2 folder (`BigVGAN.from_folder`); 44k only, the v1 import removed |
| `ext/bigvgan_v2/bigvgan.py` | `huggingface_hub.PyTorchModelHubMixin` / `hf_hub_download`; optional CUDA kernel compiled with nvcc at run time | plain `nn.Module`; torch activation only; `BigVGAN.from_folder()` reads `config.json` and `bigvgan_generator.pt` with `torch.load(weights_only=True)`. The `.py` files in the downloaded model folder are never imported |
| `ext/synchformer/motionformer.py` | `OmegaConf.load`, `timm.layers.trunc_normal_`, config/checkpoint download helpers | `yaml.safe_load` into a small attribute dict; `torch.nn.init.trunc_normal_`; no downloads (the config ships here) |
| `ext/synchformer/video_model_builder.py` | `timm.layers.trunc_normal_` | `torch.nn.init.trunc_normal_` (initialization only, the weights load over it) |
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

`lc_foley.py` builds the models with their parameters on the meta device and sets them from the safetensors files at
the chosen dtype (what accelerate's `init_empty_weights` does, without accelerate), and turns ComfyUI IMAGE frames into
8-bit RGB frames for MMAudio's own transforms from `eval_utils.load_video` (bicubic resize, `ToDtype(scale=True)`,
center crop and normalization for Synchformer), sampled at 8 and 25 fps from the clip's real frame rate.

## Model files (not in this repository)

All in `models/mmaudio`:

| File | Made by | Downloaded from |
| --- | --- | --- |
| `mmaudio_large_44k_v2_fp16.safetensors` | MMAudio authors (Sony Research) | https://huggingface.co/Kijai/MMAudio_safetensors (fp16 safetensors conversion) |
| `mmaudio_vae_44k_fp16.safetensors` | MMAudio authors | https://huggingface.co/Kijai/MMAudio_safetensors |
| `mmaudio_synchformer_fp16.safetensors` | Synchformer (Vladimir Iashin et al.), as MMAudio's state-dict extract | https://huggingface.co/Kijai/MMAudio_safetensors |
| `apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors` | Apple (DFN5B-CLIP) | https://huggingface.co/Kijai/MMAudio_safetensors |
| `nvidia/bigvgan_v2_44khz_128band_512x/config.json`, `bigvgan_generator.pt` | NVIDIA (BigVGAN v2) | https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x (only these two files) |
