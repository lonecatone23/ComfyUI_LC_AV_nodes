# Vendored code: IndexTTS-2 (bilibili index-tts)

| | |
| --- | --- |
| Upstream | https://github.com/index-tts/index-tts (IndexTTS-2 inference code, bilibili) |
| Taken from | TTS-Audio-Suite 5.11.1, `engines/index_tts/indextts` (https://github.com/diodiogod/TTS-Audio-Suite, commit `59d4cd46d3be74ef1148b97dc512d8b339601417`, 2026-11-24), which bundles index-tts with its own patches |
| License | IndexTTS code and model: bilibili Model Use License Agreement (`LICENSE.txt`, Chinese text `LICENSE_ZH.txt`). TTS-Audio-Suite wrapper patches: MIT (`LICENSE.TTS-Audio-Suite`). Files adapted by index-tts from other projects keep their own header notices (Apache-2.0: HuggingFace transformers copies, WeNet conformer, 3D-Speaker CAM++, alias-free-torch; MIT: NVIDIA BigVGAN, hifi-gan, snake, Amphion MaskGCT codec; Meta: encodec / gpt-fast). |
| Weights | `IndexTeam/IndexTTS-2`, `facebook/w2v-bert-2.0`, `amphion/MaskGCT` (semantic codec), `funasr/campplus`, `nvidia/bigvgan_v2_22khz_80band_256x` on Hugging Face, found or downloaded on first use by `lc_indextts.py` (not in this repository) |

Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this Derivative Work.

Used by LC IndexTTS Engine (`lc_tts_engine.py` -> `lc_indextts.py`) so LC Dialogue, LC Emotion Editor and LC Voice Match run
IndexTTS-2 without TTS-Audio-Suite and without installing anything: only what ComfyUI itself requires (torch, numpy, scipy,
einops, transformers, sentencepiece, safetensors, pyyaml, tqdm) plus this pack's `lc_dsp.py`.

Only the files IndexTTS-2 inference runs are vendored. Imports are relative (`from indextts.x import y` became
`from .x import y`). Empty `__init__.py` files were added where upstream used namespace folders.

## Copied unchanged (apart from relative imports)

`gpt/conformer_encoder.py`, `gpt/perceiver.py`, `gpt/transformers_beam_search.py`, `gpt/_attn_mask_utils_t4.py`,
`gpt/conformer/attention.py`, `gpt/conformer/embedding.py`, `gpt/conformer/subsampling.py`, `utils/arch_util.py`,
`utils/checkpoint.py`, `utils/xtransformers.py`, `utils/typical_sampling.py`, `utils/module_device.py`,
`utils/maskgct/models/codec/kmeans/repcodec_model.py`, `utils/maskgct/models/codec/amphion_codec/quantize/*.py`,
`s2mel/modules/flow_matching.py`, `s2mel/modules/length_regulator.py`, `s2mel/modules/diffusion_transformer.py`,
`s2mel/modules/gpt_fast/model.py`, `s2mel/modules/wavenet.py`, `s2mel/modules/encodec.py`, `s2mel/dac/nn/quantize.py`,
`s2mel/dac/nn/layers.py`, `s2mel/modules/bigvgan/activations.py`, `s2mel/modules/bigvgan/env.py`,
`s2mel/modules/bigvgan/alias_free_activation/torch/*.py`, `s2mel/modules/campplus/DTDNN.py`,
`s2mel/modules/campplus/layers.py`.

Kept from TTS-Audio-Suite's local transformers-5 fix: `gpt/transformers_gpt2.py` falls back to `gpt/_attn_mask_utils_t4.py`
(a copy of transformers 4's `modeling_attn_mask_utils`) when `transformers.modeling_attn_mask_utils` is gone, and
`gpt/model_v2.py` falls back to the local `assert_device_map` / `get_device_map`. TTS-Audio-Suite's `weights_only=True`
patches on every `torch.load` are kept too.

## Changed

| File | Change |
| --- | --- |
| `infer_v2.py` | Rewritten around the same synthesis code. OmegaConf -> `yaml.safe_load` + a small attribute dict. No `HF_HUB_CACHE` override, no downloader, no `sys.path` edits: the five weight locations are constructor arguments. Reference audio comes in decoded (`{"waveform", "sample_rate", "key"}`, cached by content key) instead of a file path read with librosa; `torchaudio.transforms.Resample` and `torchaudio.compliance.kaldi.fbank` -> the ports in `utils/ta_compat.py` (bit-identical). The 16 kHz emotion-reference load (upstream `librosa.load(sr=16000)`, soxr_hq) uses the same sinc resampler as the speaker path. Returns `(22050, int16 array)` like upstream's in-memory branch; no file output, no streaming generator. Removed: DeepSpeed, GPT2 flash-attn acceleration, torch.compile, BigVGAN CUDA kernel, XPU workarounds. QwenEmotion loads on first text-emotion request, on one device, without accelerate's `device_map`. An `on_segment` hook lets ComfyUI's Stop cancel between text segments. |
| `gpt/model_v2.py` | `LANGUAGE_DICT` (from `utils/tokenizer.py`, which needs whisper + tiktoken) replaced by its size (106; only IndexTTS-2.5's campplus mode uses it). Acceleration engine, DeepSpeed and the modelscope speaker-verification pipeline (IndexTTS-2.5 only) raise instead of importing. |
| `gpt/transformers_gpt2.py` | flash-attention 2 import removed (IndexTTS-2 never selects it). |
| `gpt/transformers_generation_utils.py`, `gpt/transformers_modeling_utils.py` | A local `is_accelerate_available()` returning False, so these copies never import accelerate (only needed for `device_map` / offload loading, which IndexTTS-2 does not use). |
| `utils/front.py` | wetext / WeTextProcessing normalizers -> `utils/lite_normalizer.py` (see below). |
| `utils/common.py` | `load_audio` (torchaudio) removed. |
| `utils/maskgct_utils.py` | Only `build_semantic_model` (now given the local w2v-bert-2.0 folder) and `build_semantic_codec`; the MaskGCT S2A / acoustic codec builders and their librosa / json5 / huggingface_hub imports removed. |
| `utils/maskgct/models/codec/kmeans/vocos.py` | `torchaudio.functional._hz_to_mel / _mel_to_hz` -> `utils/ta_compat.py`. |
| `s2mel/modules/audio.py` | `librosa.filters.mel` -> `lc_dsp._mel_filters` (this pack's numpy port, bit-identical). |
| `s2mel/modules/commons.py` | `munch.Munch` -> a small dict subclass; matplotlib plotting, the JDC F0 loader and the training `build_model` removed. |
| `s2mel/modules/bigvgan/bigvgan.py` | No `huggingface_hub.PyTorchModelHubMixin`; `from_pretrained(folder)` reads `config.json` + `bigvgan_generator.pt` locally (`weights_only=True`). The CUDA-kernel branches raise (kernel not included). |
| `s2mel/modules/bigvgan/utils.py` | matplotlib plotting and `save_audio` removed (they imported matplotlib and `meldataset`, which needs librosa). |

## Written for this pack

- `utils/ta_compat.py`: torchaudio 2.11.0's `transforms.Resample`, `functional.resample`, `compliance.kaldi.fbank` and
  `_hz_to_mel` / `_mel_to_hz`, ported to plain torch (BSD 2-Clause notice inside). Same arithmetic; checked bit-identical.
- `utils/lite_normalizer.py`: a rule-based English text normalizer in place of wetext (numbers, years, decimals, ordinals,
  money, percent, clock times, simple fractions, decades, a few titles). Not covered: dates, units, phone numbers, URLs,
  and Chinese normalization (Chinese text keeps its digits).

## Not vendored

IndexTTS-1 (`infer.py`), IndexTTS-2.5 (`infer_v2_5.py`, tiktoken / multilingual frontend), `accel/`, `BigVGAN/` (the v1
vocoder), `vqvae/`, `codec/`, the rest of `utils/maskgct/` and `s2mel/` (training, other codecs, RMVPE, OpenVoice,
HiFi-GAN, Vocos), `utils/tokenizer.py`, `cli.py`.
