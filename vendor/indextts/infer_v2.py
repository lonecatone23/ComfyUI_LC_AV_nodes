# IndexTTS-2 inference (bilibili index-tts, via TTS-Audio-Suite 5.11.1). Modified for LC AV, see ../UPSTREAM.md:
# - no torchaudio / omegaconf / librosa / huggingface_hub / deepspeed / flash-attn / torch.compile / BigVGAN CUDA kernel
# - every weight path is handed in (the LC AV engine finds or downloads them), nothing is fetched here
# - reference audio comes in decoded ({"waveform": 1-D float32 tensor, "sample_rate": int, "key": str}), not as a file
# - QwenEmotion (text emotion) loads on first use, on one device, without accelerate's device_map
# The synthesis arithmetic (conditioning, emotion vectors, GPT sampling, s2mel CFM, BigVGAN) is unchanged.
import json
import os
import random
import re
import time
import warnings

import torch
import torch.nn.functional as F
import yaml

from .gpt.model_v2 import UnifiedVoice
from .s2mel.modules.audio import mel_spectrogram
from .s2mel.modules.bigvgan import bigvgan
from .s2mel.modules.campplus.DTDNN import CAMPPlus
from .s2mel.modules.commons import MyModel, load_checkpoint2
from .utils.checkpoint import load_checkpoint
from .utils.front import TextNormalizer, TextTokenizer
from .utils.maskgct_utils import build_semantic_codec, build_semantic_model
from .utils.module_device import move_module
from .utils.ta_compat import fbank, resample

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class AttrDict(dict):
    """LC AV: stands in for OmegaConf's DictConfig (attribute access; hasattr is False for a missing key)."""

    def __init__(self, d=None):
        super().__init__()
        for k, v in (d or {}).items():
            self[k] = AttrDict(v) if isinstance(v, dict) else v

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return AttrDict(yaml.safe_load(f))


def to_container(d):
    return {k: to_container(v) for k, v in d.items()} if isinstance(d, dict) else d


class IndexTTS2:
    def __init__(self, cfg_path, model_dir, w2v_bert_path, semantic_codec_path, campplus_path, bigvgan_path,
                 use_fp16=False, device=None, low_vram=False):
        """
        Args:
            cfg_path (str): config.yaml of the IndexTTS-2 folder.
            model_dir (str): the IndexTTS-2 folder (gpt.pth, s2mel.pth, bpe.model, feat1.pt, feat2.pt, ...).
            w2v_bert_path (str): folder with facebook/w2v-bert-2.0 (config.json, model.safetensors, preprocessor_config.json).
            semantic_codec_path (str): amphion/MaskGCT semantic_codec/model.safetensors.
            campplus_path (str): funasr/campplus campplus_cn_common.bin.
            bigvgan_path (str): folder with nvidia/bigvgan_v2_22khz_80band_256x (config.json, bigvgan_generator.pt).
            use_fp16 (bool): fp16 GPT (not on CPU).
            device (str): 'cuda:0', 'cpu', ... (None = pick).
            low_vram (bool): keep the models on the CPU and move each stage to the device only while it runs.
        """
        if device is not None:
            self.device = device
            self.use_fp16 = False if device == "cpu" else bool(use_fp16)
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_fp16 = bool(use_fp16)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
            self.use_fp16 = bool(use_fp16)
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.use_fp16 = False  # Use float16 on MPS is overhead than float32
        else:
            self.device = "cpu"
            self.use_fp16 = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = load_config(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.use_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.low_vram = bool(low_vram)
        self.load_device = "cpu" if self.low_vram else self.device

        # QwenEmotion (text -> emotion vector) loads on first use
        self.qwen_emo = None
        self._qwen_emo_disabled = False
        qwen = str(self.cfg.get("qwen_emo_path") or "").strip() or "qwen0.6bemo4-merge"
        self.qwen_emo_path = os.path.normpath(os.path.join(self.model_dir, qwen))

        gpt_config = to_container(self.cfg.gpt)
        gpt_config["use_accel"] = False
        self.gpt = UnifiedVoice(**gpt_config)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.load_device)
        if self.use_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=self.use_fp16)

        from transformers import SeamlessM4TFeatureExtractor

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(w2v_bert_path)
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat), w2v_bert_path)
        self.semantic_model = self.semantic_model.to(self.load_device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.load_device)
        self.semantic_std = self.semantic_std.to(self.load_device)

        import safetensors.torch

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        safetensors.torch.load_model(semantic_codec, semantic_codec_path)
        self.semantic_codec = semantic_codec.to(self.load_device)
        self.semantic_codec.eval()
        print('>> semantic_codec weights restored from: {}'.format(semantic_codec_path))

        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.load_device)
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.s2mel.eval()
        print(">> s2mel weights restored from:", s2mel_path)

        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_path, map_location="cpu", weights_only=True))
        self.campplus_model = campplus_model.to(self.load_device)
        self.campplus_model.eval()
        print(">> campplus_model weights restored from:", campplus_path)

        self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_path, use_cuda_kernel=False)
        self.bigvgan = self.bigvgan.to(self.load_device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", bigvgan_path)

        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)

        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix.strip()), weights_only=True)
        self.emo_matrix = emo_matrix.to(self.load_device)
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix.strip()), weights_only=True)
        self.spk_matrix = spk_matrix.to(self.load_device)

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        spect = self.cfg.s2mel['preprocess_params']['spect_params']
        mel_fn_args = {
            "n_fft": spect['n_fft'],
            "win_size": spect['win_length'],
            "hop_size": spect['hop_length'],
            "num_mels": spect['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": spect.get('fmin', 0),
            "fmax": None if spect.get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        # cached reference-voice conditioning
        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_mel = None

        # LC AV: called at the start of every text segment (Stop / progress); may raise to cancel
        self.on_segment = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    def to(self, device):
        """Move every part (plain tensors inside modules too) to device; low VRAM keeps the parts on the CPU."""
        device = str(device)
        self.load_device = "cpu" if self.low_vram else device
        for name in ["gpt", "semantic_model", "semantic_codec", "s2mel", "campplus_model", "bigvgan"]:
            move_module(getattr(self, name), self.load_device)
        if self.qwen_emo is not None:
            self.qwen_emo.to(device)
        self.semantic_mean = self.semantic_mean.to(self.load_device)
        self.semantic_std = self.semantic_std.to(self.load_device)
        self._move_matrices(self.load_device)
        self.cache_spk_cond = self.cache_s2mel_style = self.cache_s2mel_prompt = self.cache_spk_audio_prompt = None
        self.cache_emo_cond = self.cache_emo_audio_prompt = self.cache_mel = None
        self.device = device
        return self

    def _move_matrices(self, device):
        if hasattr(self, 'emo_matrix') and isinstance(self.emo_matrix, (tuple, list)):
            self.emo_matrix = tuple(t.to(device) for t in self.emo_matrix)
        if hasattr(self, 'spk_matrix') and isinstance(self.spk_matrix, (tuple, list)):
            self.spk_matrix = tuple(t.to(device) for t in self.spk_matrix)

    def _qwen(self):
        if self.qwen_emo is None and not self._qwen_emo_disabled:
            need = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
            missing = [f for f in need if not os.path.exists(os.path.join(self.qwen_emo_path, f))]
            if missing:
                print(f"QwenEmotion model missing {missing} in {self.qwen_emo_path}: no text emotion")
                self._qwen_emo_disabled = True
            else:
                self.qwen_emo = QwenEmotion(self.qwen_emo_path, preferred_device=self.device)
        return self.qwen_emo

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        input_features = input_features.to(self.device)
        attention_mask = attention_mask.to(self.device)
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        if not wavs or interval_silence <= 0:
            return wavs
        channel_size = wavs[0].size(0)
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)
        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)
        return wavs_list

    @staticmethod
    def _load_and_cut_audio(prompt, max_audio_length_seconds, sr=None):
        """LC AV: prompt is {"waveform": 1-D float32 mono tensor, "sample_rate": int}. Same as upstream's
        librosa.load(path, sr=sr, mono=True) + cut, except the resample to `sr` is torchaudio's sinc filter (ported),
        not librosa's soxr_hq."""
        audio = prompt["waveform"].detach().float().cpu().reshape(-1)
        in_sr = int(prompt["sample_rate"])
        if sr and sr != in_sr:
            audio = resample(audio[None], in_sr, sr)[0]
        else:
            sr = in_sr
        audio = audio.unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)
        if audio.shape[1] > max_audio_samples:
            audio = audio[:, :max_audio_samples]
        return audio, sr

    @staticmethod
    def _key(prompt):
        return None if prompt is None else prompt.get("key")

    def normalize_emo_vec(self, emo_vector, apply_bias=True):
        # apply biased emotion factors for better user experience,
        # by de-emphasizing emotions that can cause strange results
        if apply_bias:
            # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
            emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

        # the total emotion sum must be 0.8 or less
        emo_sum = sum(emo_vector)
        if emo_sum > 0.8:
            scale_factor = 0.8 / emo_sum
            emo_vector = [vec * scale_factor for vec in emo_vector]

        return emo_vector

    def infer(self, spk_audio_prompt, text, emo_audio_prompt=None, emo_alpha=1.0, emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200, verbose=False,
              max_text_tokens_per_segment=120, **generation_kwargs):
        """-> (22050, int16 numpy [samples, 1]) like upstream's output_path=None branch."""
        print(">> starting inference...")
        start_time = time.perf_counter()

        if use_emo_text:
            qwen = self._qwen()
            if qwen is None:
                print("QwenEmotion text analysis unavailable - continuing without text emotion")
                use_emo_text = False
            else:
                if emo_text is None:
                    emo_text = text  # use main text prompt
                try:
                    emo_dict = qwen.inference(emo_text)
                except Exception as exc:
                    self._qwen_emo_disabled = True
                    self.qwen_emo = None
                    use_emo_text = False
                    emo_vector = None
                    print("QwenEmotion text analysis failed; continuing without "
                          f"text emotion ({type(exc).__name__}: {exc})")
                else:
                    print(f"detected emotion vectors from text: {emo_dict}")
                    # convert ordered dict to list of vectors; the order is VERY important!
                    emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            emo_vector = [max(0.0, min(1.2, float(v))) for v in emo_vector]
            emo_vector = self.normalize_emo_vec(emo_vector)
            print(f"Using emotion vectors: {[round(x, 3) for x in emo_vector]}")
            # we have emotion vectors; they can't be blended via alpha mixing
            # in the main inference process later, so we must pre-calculate
            # their new strengths here based on the alpha instead!
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]

        if emo_audio_prompt is None:
            # no external emotion reference voice: the speaker's voice is the emotion reference
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        if spk_audio_prompt is None:
            raise ValueError("Speaker audio prompt is required but was None.")

        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != self._key(spk_audio_prompt):
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            if self.low_vram:
                self.semantic_model = self.semantic_model.to(self.device)
                self.semantic_mean = self.semantic_mean.to(self.device)
                self.semantic_std = self.semantic_std.to(self.device)
                self.semantic_codec = self.semantic_codec.to(self.device)
                self.campplus_model = self.campplus_model.to(self.device)
                self.s2mel = self.s2mel.to(self.device)

            audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15)
            audio_22k = resample(audio, sr, 22050)
            audio_16k = resample(audio, sr, 16000)

            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)

            _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
            ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
            feat = fbank(audio_16k.to(ref_mel.device), num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = self.campplus_model(feat.unsqueeze(0))

            prompt_condition = self.s2mel.models['length_regulator'](S_ref,
                                                                     ylens=ref_target_lengths,
                                                                     n_quantizers=3,
                                                                     f0=None)[0]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = self._key(spk_audio_prompt)
            self.cache_mel = ref_mel

            if self.low_vram:
                self.semantic_model = self.semantic_model.to("cpu")
                self.semantic_mean = self.semantic_mean.to("cpu")
                self.semantic_std = self.semantic_std.to("cpu")
                self.campplus_model = self.campplus_model.to("cpu")
                self.s2mel = self.s2mel.to("cpu")
                self.semantic_codec = self.semantic_codec.to("cpu")
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        if emo_vector is not None:
            if self.low_vram:
                self._move_matrices(self.device)
            weight_vector = torch.tensor(emo_vector).to(self.device)
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]
            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0)
            emovec_mat = emovec_mat.unsqueeze(0)
            if self.low_vram:
                self._move_matrices("cpu")

        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != self._key(emo_audio_prompt):
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            if self.low_vram:
                self.semantic_model = self.semantic_model.to(self.device)
                self.semantic_mean = self.semantic_mean.to(self.device)
                self.semantic_std = self.semantic_std.to(self.device)
                self.semantic_codec = self.semantic_codec.to(self.device)

            emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, sr=16000)
            emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
            emo_input_features = emo_inputs["input_features"].to(self.device)
            emo_attention_mask = emo_inputs["attention_mask"].to(self.device)
            emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

            if self.low_vram:
                self.semantic_model = self.semantic_model.to("cpu")
                self.semantic_mean = self.semantic_mean.to("cpu")
                self.semantic_std = self.semantic_std.to("cpu")
                self.semantic_codec = self.semantic_codec.to("cpu")
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = self._key(emo_audio_prompt)
        else:
            emo_cond_emb = self.cache_emo_cond

        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment)
        segments_count = len(segments)

        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if self.tokenizer.unk_token_id in text_token_ids:
            print(f"  >> Warning: input text contains {text_token_ids.count(self.tokenizer.unk_token_id)} unknown tokens (id={self.tokenizer.unk_token_id}):")
            print("     Tokens which can't be encoded: ", [t for t, id in zip(text_tokens_list, text_token_ids) if id == self.tokenizer.unk_token_id])

        if verbose:
            print("text_tokens_list:", text_tokens_list)
            print("segments count:", segments_count)
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        sampling_rate = 22050

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        for seg_idx, sent in enumerate(segments):
            if self.on_segment is not None:
                self.on_segment(seg_idx, segments_count)

            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)

            m_start_time = time.perf_counter()
            with torch.no_grad():
                if self.low_vram:
                    self.gpt = self.gpt.to(self.device)

                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    emovec = self.gpt.merge_emovec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        alpha=emo_alpha
                    )

                    if emo_vector is not None:
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec

                    codes, speech_conditioning_latent = self.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        emo_cond_emb,
                        cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        do_sample=do_sample,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=autoregressive_batch_size,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                        **generation_kwargs
                    )

                gpt_gen_time += time.perf_counter() - m_start_time

                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                code_lens = []
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_lens.append(len(code))
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                        code_len = len_ - 1
                    code_lens.append(code_len)
                codes = codes[:, :code_len]
                code_lens = torch.LongTensor(code_lens)
                code_lens = code_lens.to(self.device)

                m_start_time = time.perf_counter()
                use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = self.gpt(
                        speech_conditioning_latent,
                        text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                        codes,
                        torch.tensor([codes.shape[-1]], device=text_tokens.device),
                        emo_cond_emb,
                        cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        use_speed=use_speed,
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time

                if self.low_vram:
                    self.gpt = self.gpt.to("cpu")
                    if self.device.startswith("cuda"):
                        torch.cuda.empty_cache()

                dtype = None
                with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                    if self.low_vram:
                        self.s2mel = self.s2mel.to(self.device)
                        # freqs_cis is a plain attribute, not a buffer, so .to() doesn't move it
                        cfm = self.s2mel.models['cfm'] if 'cfm' in self.s2mel.models else None
                        if cfm and hasattr(cfm, 'estimator') and hasattr(cfm.estimator, 'transformer'):
                            if getattr(cfm.estimator.transformer, 'freqs_cis', None) is not None:
                                cfm.estimator.transformer.freqs_cis = cfm.estimator.transformer.freqs_cis.to(self.device)
                        self.semantic_codec = self.semantic_codec.to(self.device)
                        self.bigvgan = self.bigvgan.to(self.device)

                    m_start_time = time.perf_counter()
                    diffusion_steps = 25
                    inference_cfg_rate = 0.7
                    latent = self.s2mel.models['gpt_layer'](latent)
                    S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                    S_infer = S_infer.transpose(1, 2)
                    S_infer = S_infer + latent
                    target_lengths = (code_lens * 1.72).long()

                    cond = self.s2mel.models['length_regulator'](S_infer,
                                                                 ylens=target_lengths,
                                                                 n_quantizers=3,
                                                                 f0=None)[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)
                    vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                   torch.LongTensor([cat_condition.size(1)]).to(
                                                                       cond.device),
                                                                   ref_mel, style, None, diffusion_steps,
                                                                   inference_cfg_rate=inference_cfg_rate)
                    vc_target = vc_target[:, :, ref_mel.size(-1):]
                    s2mel_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                wavs.append(wav.cpu())

                if self.low_vram:
                    self.s2mel = self.s2mel.to("cpu")
                    self.semantic_codec = self.semantic_codec.to("cpu")
                    self.bigvgan = self.bigvgan.to("cpu")
                    if self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
        end_time = time.perf_counter()

        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} s, gpt_forward_time: {gpt_forward_time:.2f} s, "
              f"s2mel_time: {s2mel_time:.2f} s, bigvgan_time: {bigvgan_time:.2f} s")
        print(f">> Total inference time: {end_time - start_time:.2f} s for {wav_length:.2f} s of audio")
        wav_data = wav.cpu().type(torch.int16)
        return sampling_rate, wav_data.numpy().T


def find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()
    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index


class QwenEmotion:
    def __init__(self, model_dir, preferred_device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        # LC AV: one device, no accelerate device_map: load, then move
        qwen_device = self._select_device(preferred_device)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, dtype=torch.float16)
        self.model.to(qwen_device).eval()
        if hasattr(self.model, 'generation_config'):
            try:
                self.model.generation_config.use_cache = False
            except Exception:
                pass
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy",
            "愤怒": "angry",
            "悲伤": "sad",
            "恐惧": "afraid",
            "反感": "disgusted",
            # TODO: the "低落" (melancholic) emotion will always be mapped to
            # "悲伤" (sad) by QwenEmotion's text analysis. it doesn't know the
            # difference between those emotions even if user writes exact words.
            # SEE: `self.melancholic_words` for current workaround.
            "低落": "melancholic",
            "惊讶": "surprised",
            "自然": "calm",
        }
        self.desired_vector_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        self.melancholic_words = {
            # emotion text phrases that will force QwenEmotion's "悲伤" (sad) detection
            # to become "低落" (melancholic) instead, to fix limitations mentioned above.
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }
        self.max_score = 1.2
        self.min_score = 0.0

    @staticmethod
    def _select_device(preferred_device=None):
        preferred = str(preferred_device) if preferred_device is not None else None
        if not torch.cuda.is_available():
            return preferred or "cpu"
        try:
            free_memory = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
            return f"cuda:{max(range(len(free_memory)), key=free_memory.__getitem__)}"
        except Exception:
            return preferred if preferred and preferred.startswith("cuda") else "cuda:0"

    def to(self, device):
        device = str(device)
        if device.startswith("cuda"):
            device = self._select_device(device)
        self.model.to(device)
        return self

    def clamp_score(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"QwenEmotion returned a non-numeric emotion score {value!r}.") from exc
        return max(self.min_score, min(self.max_score, value))

    def convert(self, content):
        emotion_dict = {
            self.cn_key_to_en[cn_key]: self.clamp_score(content.get(cn_key, 0.0))
            for cn_key in self.desired_vector_order
        }
        # default to a calm/neutral voice if all emotion vectors were empty
        if all(val <= 0.0 for val in emotion_dict.values()):
            print(">> no emotions detected; using default calm/neutral voice")
            emotion_dict["calm"] = 1.0
        return emotion_dict

    def inference(self, text_input):
        messages = [
            {"role": "system", "content": f"{self.prompt}"},
            {"role": "user", "content": f"{text_input}"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_device = next(self.model.parameters()).device
        model_inputs = self.tokenizer([text], return_tensors="pt")
        model_inputs = {key: value.to(input_device) for key, value in model_inputs.items()}

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)

        try:
            content = json.loads(content)
        except json.decoder.JSONDecodeError:
            # invalid JSON; fallback to manual string parsing
            content = {
                m.group(1): float(m.group(2))
                for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
            }

        # workaround for QwenEmotion's inability to distinguish "悲伤" (sad) vs "低落" (melancholic).
        text_input_lower = text_input.lower()
        if any(word in text_input_lower for word in self.melancholic_words):
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)

        return self.convert(content)
