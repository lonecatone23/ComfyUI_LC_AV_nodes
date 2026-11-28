# ChatterBox VC (vendored)

Used by **LC Voice Changer** (`lc_voice_changer.py`): speech in, the same performance out in another voice.

| Part | From | License |
|---|---|---|
| `models/s3gen`, `models/s3tokenizer` | ChatterBox by Resemble AI (https://github.com/resemble-ai/chatterbox, `src/chatterbox/models`), taken from TTS-Audio-Suite 5.11.1 by Shmuel Ronen (https://github.com/diodiogod/TTS-Audio-Suite, commit `59d4cd46d3be74ef1148b97dc512d8b339601417`, `engines/chatterbox/models`), which carries it with its own patches | ChatterBox: MIT, Copyright (c) 2025 Resemble AI (`LICENSE.chatterbox`). TTS-Audio-Suite: MIT, Copyright (c) 2025 Shmuel Ronen (`LICENSE.TTS-Audio-Suite`). Files ChatterBox adapted from other projects (CosyVoice, WeNet, ESPnet, FunASR, Matcha-TTS, snake) keep their own header notices. |
| `s3tok/model.py`, `s3tok/model_v2.py`, `s3tok/utils.py` | s3tokenizer 0.3.0 by Xingchen Song (https://github.com/xingchensong/S3Tokenizer) | Apache-2.0 (`LICENSE.S3Tokenizer`, headers kept) |
| `compat.py` | LC AV: the diffusers 0.39.0 pieces ChatterBox runs (Apache-2.0), plus LC AV's own librosa / torchaudio ports | see the file header |

Changes, all import-level (the model code is untouched):

- omegaconf `DictConfig` / `OmegaConf.create` -> `compat.DictConfig` (a dict with attribute access).
- diffusers `get_activation`, `GELU`, `Attention` (AttnProcessor2_0 math), `LoRACompatibleLinear`, `maybe_allow_in_graph` -> `compat` (same parameter names, so the checkpoint loads as is). `GEGLU`, `ApproximateGELU`, `AdaLayerNorm`, `AdaLayerNormZero` and conformer's `ConformerBlock` are stubs: ChatterBox doesn't build them.
- librosa mel filters -> `lc_dsp._mel_filters`; torchaudio `Resample` / `compliance.kaldi.fbank` -> `vendor/indextts/utils/ta_compat.py`.
- `s3tok/utils.py`: only `make_non_pad_mask`, `mask_to_bias`, `padding`, `merge_tokenized_segments` (no onnx / torchaudio).
- The watermarker (perth) is left out, as in TTS-Audio-Suite's default.

Weights: `models/TTS/chatterbox/English/s3gen.safetensors` from ResembleAI/chatterbox (MIT), downloaded on first use, never bundled. Speaker similarity matches TTS-Audio-Suite's ChatterBox VC on the same clips (CAM++ 0.881 / 0.860 / 0.853 vs 0.883 / 0.853 / 0.852).
