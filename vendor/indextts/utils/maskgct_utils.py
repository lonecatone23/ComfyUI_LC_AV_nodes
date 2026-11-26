# LC AV: trimmed to the two builders IndexTTS-2 inference uses. The MaskGCT S2A / acoustic-codec builders, the json5
# config loader and the librosa-based inference pipeline were removed (not used, and they pulled in librosa / json5).
# build_semantic_model takes the local w2v-bert-2.0 folder instead of calling the suite's downloader.
import torch
from transformers import Wav2Vec2BertModel

from .maskgct.models.codec.kmeans.repcodec_model import RepCodec


def build_semantic_model(path_, w2v_bert_path):
    semantic_model = Wav2Vec2BertModel.from_pretrained(w2v_bert_path, local_files_only=True)
    semantic_model.eval()
    stat_mean_var = torch.load(path_, weights_only=True)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    return semantic_model, semantic_mean, semantic_std


def build_semantic_codec(cfg):
    semantic_codec = RepCodec(cfg=cfg)
    semantic_codec.eval()
    return semantic_codec
