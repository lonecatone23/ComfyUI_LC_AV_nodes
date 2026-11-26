# Vendored code: SNAC (Multi-Scale Neural Audio Codec)

| | |
| --- | --- |
| Upstream | https://github.com/hubertsiuzdak/snac |
| Version | 1.2.1 (PyPI `snac`) |
| License | MIT (`LICENSE`, Copyright (c) 2023, Descript) |
| Weights | `hubertsiuzdak/snac_24khz` on Hugging Face (MIT), downloaded on first use to `models/maya1-TTS/snac_24khz` |

Used by LC Voice Design to turn Maya1's audio tokens into sound, so the `snac` package (and the Maya1 node pack) are not needed.

## Copied unchanged

`attention.py`, `layers.py`, `vq.py`, `LICENSE`.

## Changed

- `snac.py`: `from_pretrained` (downloads with `huggingface_hub`, `torch.load` without `weights_only`) replaced by
  `from_folder`, which reads `config.json` + `pytorch_model.bin` from a local folder with `torch.load(weights_only=True)`.
- `__init__.py`: written for this pack (exports `SNAC`).
