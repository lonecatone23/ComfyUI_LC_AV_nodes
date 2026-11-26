"""
IndexTTS-2 inside LC AV: loads the vendored engine (vendor/indextts, bilibili index-tts via TTS-Audio-Suite, see
vendor/indextts/UPSTREAM.md) and finds or downloads its weights. No TTS-Audio-Suite, nothing to pip install.

Weights (first use downloads what is missing, with the LC toast + Stop):
  models/IndexTTS-2                 IndexTeam/IndexTTS-2: config.yaml, gpt.pth, s2mel.pth, bpe.model, feat1.pt, feat2.pt,
                                    wav2vec2bert_stats.pt (~4.7 GB); qwen0.6bemo4-merge/ (~1.2 GB) only for text emotion
  models/IndexTTS-2/w2v-bert-2.0    facebook/w2v-bert-2.0 (~2.3 GB)
  models/IndexTTS-2/MaskGCT         amphion/MaskGCT semantic_codec/model.safetensors (177 MB)
  models/IndexTTS-2/campplus        funasr/campplus campplus_cn_common.bin (28 MB)
  models/IndexTTS-2/bigvgan_v2_22khz_80band_256x   nvidia/bigvgan_v2_22khz_80band_256x (449 MB)
Copies TTS-Audio-Suite already downloaded (models/TTS/IndexTTS/...) or the Hugging Face cache holds are used in place.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

SR = 22050
EMOTIONS = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
MAIN_REPO = "IndexTeam/IndexTTS-2"
MAIN_FILES = ("config.yaml", "gpt.pth", "s2mel.pth", "bpe.model", "feat1.pt", "feat2.pt", "wav2vec2bert_stats.pt")
QWEN_DIR = "qwen0.6bemo4-merge"
QWEN_FILES = ("added_tokens.json", "chat_template.jinja", "config.json", "generation_config.json", "merges.txt",
              "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json")
# name: (repo, files, label, size)
AUX = {
    "w2v-bert-2.0": ("facebook/w2v-bert-2.0", ("config.json", "model.safetensors", "preprocessor_config.json"),
                     "W2V-BERT 2.0 (IndexTTS-2 speech features)", "2.3 GB"),
    "MaskGCT": ("amphion/MaskGCT", ("semantic_codec/model.safetensors",), "MaskGCT semantic codec", "177 MB"),
    "campplus": ("funasr/campplus", ("campplus_cn_common.bin",), "CAM++ speaker model", "28 MB"),
    "bigvgan_v2_22khz_80band_256x": ("nvidia/bigvgan_v2_22khz_80band_256x", ("config.json", "bigvgan_generator.pt"),
                                     "BigVGAN vocoder", "449 MB"),
}
# generation settings TTS-Audio-Suite's IndexTTS engine node used by default (LC IndexTTS Engine always ran with them)
GEN = dict(do_sample=True, top_p=0.8, top_k=30, length_penalty=0.0, num_beams=3, repetition_penalty=10.0,
           max_mel_tokens=1500)
MAX_TEXT_TOKENS, INTERVAL_SILENCE = 120, 200

_cache = {}


def _models_dir():
    import folder_paths

    return folder_paths.models_dir


def _has(folder, files):
    return bool(folder) and all(os.path.isfile(os.path.join(folder, f)) for f in files)


def model_choices():
    return ["local:IndexTTS-2"]


def main_folder(model="local:IndexTTS-2", download=True):
    """models/IndexTTS-2 (or the copy TTS-Audio-Suite keeps in models/TTS/IndexTTS/IndexTTS-2)."""
    name = str(model).split(":", 1)[-1] or "IndexTTS-2"
    root = _models_dir()
    candidates = [os.path.join(root, name), os.path.join(root, "TTS", "IndexTTS", name), os.path.join(root, "TTS", name)]
    for c in candidates:
        if _has(c, MAIN_FILES):
            return c
    if not download:
        return None
    dest = candidates[0]
    _fetch(MAIN_REPO, dest, MAIN_FILES, f"{name} voice model", "4.7 GB")
    return dest


def _hf_snapshots(repo, extra_roots=()):
    name = "models--" + repo.replace("/", "--")
    roots = list(extra_roots)
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(env):
            roots.append(os.environ[env])
    if os.environ.get("HF_HOME"):
        roots.append(os.path.join(os.environ["HF_HOME"], "hub"))
    roots.append(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    for r in roots:
        snaps = os.path.join(r, name, "snapshots")
        if os.path.isdir(snaps):
            for s in sorted(os.listdir(snaps)):
                yield os.path.join(snaps, s)


def aux_folder(key, main):
    """An existing copy (suite folder, our folder, the IndexTTS-2 hf_cache, the Hugging Face cache) or a download into
    models/IndexTTS-2/<key>."""
    repo, files, label, size = AUX[key]
    root = _models_dir()
    for c in (os.path.join(root, "TTS", "IndexTTS", key), os.path.join(main, key)):
        if _has(c, files):
            return c
    for snap in _hf_snapshots(repo, (os.path.join(main, "hf_cache"),)):
        if _has(snap, files):
            return snap
    dest = os.path.join(main, key)
    _fetch(repo, dest, files, label, size)
    return dest


def qwen_folder(main):
    folder = os.path.join(main, QWEN_DIR)
    if not _has(folder, QWEN_FILES):
        _fetch(MAIN_REPO, main, tuple(f"{QWEN_DIR}/{f}" for f in QWEN_FILES), "IndexTTS-2 text emotion model", "1.2 GB")
    return folder


def _fetch(repo, folder, files, label, size):
    from .lc_first_download import download, hf_url, toast

    missing = [f for f in files if not os.path.isfile(os.path.join(folder, f))]
    if not missing:
        return
    toast(label, size)
    for f in missing:
        download(hf_url(repo, f), os.path.join(folder, *f.split("/")), f"{label}: {f}", announce=False)


def _device():
    import torch

    try:
        import comfy.model_management as mm

        dev = mm.get_torch_device()
        if dev.type == "cuda":
            return f"cuda:{dev.index or 0}"
        return str(dev.type)
    except Exception:
        return "cuda:0" if torch.cuda.is_available() else "cpu"


def load(model="local:IndexTTS-2", low_vram=False, device=None):
    """The IndexTTS-2 engine (cached; one at a time)."""
    device = device or _device()
    main = main_folder(model)
    key = (main, device, bool(low_vram))
    if _cache.get("key") == key:
        return _cache["engine"]
    unload()
    paths = {k: aux_folder(k, main) for k in AUX}
    from .vendor.indextts.infer_v2 import IndexTTS2

    engine = IndexTTS2(cfg_path=os.path.join(main, "config.yaml"), model_dir=main,
                       w2v_bert_path=paths["w2v-bert-2.0"],
                       semantic_codec_path=os.path.join(paths["MaskGCT"], "semantic_codec", "model.safetensors"),
                       campplus_path=os.path.join(paths["campplus"], "campplus_cn_common.bin"),
                       bigvgan_path=paths["bigvgan_v2_22khz_80band_256x"],
                       use_fp16=True, device=device, low_vram=bool(low_vram))
    _cache.update(key=key, engine=engine, main=main)
    return engine


def unload():
    _cache.clear()
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:
        pass


def _mono(audio):
    """ComfyUI AUDIO -> {"waveform": 1-D float32 tensor, "sample_rate", "key"} (channels averaged, like the suite)."""
    import torch

    w = audio["waveform"]
    w = w[0] if w.ndim == 3 else w
    w = w.float().cpu()
    if w.ndim == 2:
        w = w.mean(0) if w.shape[0] > 1 else w[0]
    sr = int(audio["sample_rate"])
    key = hashlib.sha1(w.numpy().tobytes() + str(sr).encode()).hexdigest()
    return {"waveform": w.contiguous(), "sample_rate": sr, "key": key}


def _seed_all(seed):
    import random

    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def synthesize(text, voice, emotion=None, emotion_text=None, emotion_audio=None, emotion_alpha=1.0, temperature=0.8,
               seed=1, model="local:IndexTTS-2", low_vram=False, use_random=False):
    """Speak text in the voice of `voice` (ComfyUI AUDIO). Emotion from any of:
         emotion        {name: weight} over EMOTIONS (0..1.2 each) or a list of 8 weights
         emotion_text   a description ("furious and scared") read by IndexTTS-2's Qwen emotion model; True = the text itself
         emotion_audio  ComfyUI AUDIO whose delivery to copy (blended by emotion_alpha)
       -> (float32 numpy [samples] mono, 22050)."""
    import torch

    engine = load(model, low_vram)
    try:
        import comfy.model_management as mm

        engine.on_segment = lambda i, n: mm.throw_exception_if_processing_interrupted()
    except Exception:
        engine.on_segment = None
    vec = None
    if isinstance(emotion, dict):
        if any(float(emotion.get(k, 0.0)) > 0 for k in EMOTIONS):
            vec = [float(emotion.get(k, 0.0)) for k in EMOTIONS]
    elif emotion is not None:
        vec = [float(x) for x in emotion]
        if len(vec) != 8:
            raise ValueError(f"An emotion vector has 8 weights: {', '.join(EMOTIONS)}")
    use_text = bool(emotion_text)
    if use_text:
        qwen_folder(engine.model_dir)
    seed = int(seed)
    # TTS-Audio-Suite set float32 matmuls to TF32 ("high") for the whole ComfyUI process, so every take LC Dialogue made
    # with it used TF32. Do the same, only while IndexTTS-2 runs, so a seed still gives the same take.
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        if seed != 0:
            _seed_all(seed)
        with torch.no_grad():
            sr, wav = _infer(engine, text, voice, vec, emotion_text, use_text, emotion_audio, emotion_alpha,
                             temperature, use_random)
    finally:
        torch.set_float32_matmul_precision(precision)
    x = np.asarray(wav)
    x = x.astype(np.float32) / 32767.0 if x.dtype == np.int16 else x.astype(np.float32)
    x = x.mean(axis=-1) if x.ndim == 2 else x.reshape(-1)
    return x.astype(np.float32), int(sr)


def _infer(engine, text, voice, vec, emotion_text, use_text, emotion_audio, emotion_alpha, temperature, use_random):
    return engine.infer(_mono(voice), text, emo_audio_prompt=_mono(emotion_audio) if emotion_audio else None,
                        emo_alpha=float(emotion_alpha), emo_vector=vec, use_emo_text=use_text,
                        emo_text=emotion_text if isinstance(emotion_text, str) else None,
                        use_random=bool(use_random), interval_silence=INTERVAL_SILENCE,
                        max_text_tokens_per_segment=MAX_TEXT_TOKENS, temperature=float(temperature), **GEN)


# ---------------------------------------------------------------- speaker vector (LC Voice Match)
_cam = {}


def campplus():
    """CAM++ (IndexTTS-2's own speaker encoder, Apache-2.0 3D-Speaker) on the CPU, without loading the whole engine."""
    if "model" not in _cam:
        import torch

        from .vendor.indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

        main = main_folder(download=False) or os.path.join(_models_dir(), "IndexTTS-2")  # only CAM++ is needed here
        ckpt = os.path.join(aux_folder("campplus", main), "campplus_cn_common.bin")
        cam = CAMPPlus(feat_dim=80, embedding_size=192)
        cam.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        _cam["model"] = cam.eval()
    return _cam["model"]


def speaker_vector(audio):
    """A unit voice-print vector (192 floats) for a ComfyUI AUDIO clip."""
    import torch

    from .vendor.indextts.utils.ta_compat import fbank, resample

    w, sr = audio["waveform"], int(audio["sample_rate"])
    w = (w[0] if w.ndim == 3 else w).float().mean(0).cpu()
    if w.shape[-1] < sr * 0.5:
        raise ValueError("The clip is too short: give at least half a second of speech (2+ seconds is better).")
    x = resample(w, sr, 16000, functional=True)
    feat = fbank(x[None], num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(0, keepdim=True)
    with torch.no_grad():
        v = campplus()(feat[None]).reshape(-1)
    return (v / v.norm()).numpy()
