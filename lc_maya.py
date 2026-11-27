"""
Maya1 voice design engine for LC Voice Design: a plain Llama 3B (transformers, no remote code) writes SNAC audio
tokens from a voice description + text, and the SNAC 24 kHz decoder (vendored in vendor/snac, MIT) turns them into
sound. Nothing to pip install; both models download on first use:
  models/maya1-TTS/maya1        maya-research/maya1 (Apache-2.0, ~6.6 GB)
  models/maya1-TTS/snac_24khz   hubertsiuzdak/snac_24khz (MIT, 80 MB)
The prompt format and token ids follow Maya1's own transformers_inference.py.
"""

from __future__ import annotations

import os
import shutil

import numpy as np

MAYA_REPO = "maya-research/maya1"
MAYA_FILES = ("config.json", "generation_config.json", "model.safetensors.index.json",
              "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "special_tokens_map.json",
              "tokenizer.json", "tokenizer_config.json")
SNAC_REPO = "hubertsiuzdak/snac_24khz"
SNAC_FILES = ("config.json", "pytorch_model.bin")
SR = 24000

SOH, EOH, SOA, SOS, EOT, EOS_AUDIO = 128259, 128260, 128261, 128257, 128009, 128258
SNAC_START, SNAC_END, CODEBOOK = 128266, 156937, 4096

_cache = {}


def _root():
    import folder_paths

    return os.path.join(folder_paths.models_dir, "maya1-TTS")


def _hf_cache_copy(repo, folder, files):
    """A copy already in the Hugging Face cache (the old Maya1 pack put SNAC there): reuse it, no download."""
    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--" + repo.replace("/", "--"),
                       "snapshots")
    if not os.path.isdir(hub):
        return False
    for snap in sorted(os.listdir(hub)):
        src = os.path.join(hub, snap)
        if all(os.path.isfile(os.path.join(src, f)) for f in files):
            os.makedirs(folder, exist_ok=True)
            for f in files:
                shutil.copyfile(os.path.join(src, f), os.path.join(folder, f))
            return True
    return False


def _fetch(repo, folder, files, label, size):
    missing = [f for f in files if not os.path.isfile(os.path.join(folder, f))]
    if not missing or _hf_cache_copy(repo, folder, files):
        return folder
    from .lc_first_download import download, hf_url, toast

    toast(label, size)
    for f in missing:
        download(hf_url(repo, f), os.path.join(folder, f), f"{label}: {f}", announce=False)
    return folder


def maya_folder():
    """models/maya1-TTS/maya1 (the same folder the Maya1 pack used, so an existing download is reused)."""
    return _fetch(MAYA_REPO, os.path.join(_root(), "maya1"), MAYA_FILES, "Maya1 voice model", "6.6 GB")


def _load(device):
    import torch

    if _cache.get("device") == device:
        if _cache.get("parked"):  # moved to system RAM while Whisper checked the last take
            _cache["model"].to(device)
            _cache["parked"] = False
        return _cache["tok"], _cache["model"], _cache["snac"]
    _cache.clear()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .vendor.snac import SNAC

    path = maya_folder()
    snac_path = _fetch(SNAC_REPO, os.path.join(_root(), "snac_24khz"), SNAC_FILES, "SNAC audio decoder", "80 MB")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    snac = SNAC.from_folder(snac_path).to(device)
    _cache.update(device=device, tok=tok, model=model, snac=snac)
    return tok, model, snac


def park(need_bytes=int(2.5 * 1024 ** 3)):
    """Make room on the GPU for Whisper (about 1.7 GB) on small cards: Maya (6.6 GB) waits in system RAM and goes
    back for the next take. Does nothing when there is room already."""
    import torch

    m = _cache.get("model")
    if m is None or _cache.get("parked") or _cache.get("device") != "cuda":
        return
    try:
        if torch.cuda.mem_get_info()[0] >= need_bytes:
            return
    except Exception:
        return
    m.to("cpu")
    _cache["parked"] = True
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:
        torch.cuda.empty_cache()


def unload():
    _cache.clear()
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:
        pass


def _decode(tokens, snac, device):
    import torch

    t = [x for x in tokens if SNAC_START <= x <= SNAC_END]
    n = len(t) // 7
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    f = (np.asarray(t[:n * 7], dtype=np.int64) - SNAC_START).reshape(n, 7) % CODEBOOK
    codes = [f[:, 0], f[:, [1, 4]].reshape(-1), f[:, [2, 3, 5, 6]].reshape(-1)]  # 12 / 23 / 47 Hz levels
    codes = [torch.from_numpy(np.ascontiguousarray(c))[None].to(device) for c in codes]
    with torch.inference_mode():
        audio = snac.decoder(snac.quantizer.from_codes(codes))[0, 0].float().cpu().numpy()
    return audio[2048:] if audio.shape[0] > 2048 else audio  # SNAC warm-up samples


def speak(description, text, seed=8675309, max_new_tokens=4000, temperature=0.4, top_p=0.9, repetition_penalty=1.1,
          device="auto"):
    """-> float32 mono [T] at 24 kHz. Stop in ComfyUI ends the generation."""
    import comfy.model_management as mm
    import comfy.utils
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model, snac = _load(device)
    prompt = (tok.decode([SOH]) + tok.bos_token + f'<description="{description}"> {text}' + tok.decode([EOT])
              + tok.decode([EOH]) + tok.decode([SOA]) + tok.decode([SOS]))
    inputs = tok(prompt, return_tensors="pt").to(device)
    start = inputs["input_ids"].shape[1]
    bar = comfy.utils.ProgressBar(int(max_new_tokens))

    class _Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            bar.update_absolute(min(int(max_new_tokens), input_ids.shape[1] - start), int(max_new_tokens))
            return mm.processing_interrupted()

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=int(max_new_tokens), min_new_tokens=28, do_sample=True,
                             temperature=float(temperature), top_p=float(top_p),
                             repetition_penalty=float(repetition_penalty), pad_token_id=tok.pad_token_id,
                             eos_token_id=EOS_AUDIO, stopping_criteria=StoppingCriteriaList([_Stop()]),
                             use_cache=True)
    mm.throw_exception_if_processing_interrupted()
    return _decode(out[0, start:].tolist(), snac, device)
