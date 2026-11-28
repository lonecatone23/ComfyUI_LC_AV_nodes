# Copyright (c) 2023 OpenAI. (authors: Whisper Team)
#               2024 Tsinghua Univ. (authors: Xingchen Song)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py
   Add rename_weights() & onnx2torch() & make_non_pad_mask() & mask_to_bias()
   Copy merge_tokenized_segments() from https://github.com/Mddct/s3tokenizer-long/blob/main/example.py
"""

# LC AV: only the helpers the vendored model uses (no onnx / torchaudio).
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    1 for non-padded part and 0 for padded part.

    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B,).

    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, max_T).

    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool-tensor to float-tensor for flash attention.

    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B, ?).

    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, ?).

    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
        >>> new_masks = s3tokenizer.mask_to_bias(masks, torch.float32)
        new_masks =
            [[-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
             [-0.0000e+00, -0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10],
             [-0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10, -1.0000e+10]]
    """
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)

    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask


def padding(data: List[torch.Tensor]):
    """ Padding the data into batch data

    Parameters
    ----------
        data: List[Tensor], shape of Tensor (128, T)

    Returns:
    -------
        feats [B, 128, T_max], feats lengths [B]
    """
    sample = data
    assert isinstance(sample, list)
    feats_lengths = torch.tensor([s.size(1) for s in sample],
                                 dtype=torch.int32)
    feats = [s.t() for s in sample]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)

    return padded_feats.transpose(1, 2), feats_lengths


def merge_tokenized_segments(tokenized_segments, overlap, token_rate):
    """
    Merges tokenized outputs by keeping the middle and dropping half of the overlapped tokens.

    Args:
    - tokenized_segments (List[List[int]]): List of tokenized sequences.
    - overlap (int): Overlapping duration in seconds (default: 4s).
    - token_rate (int): Number of tokens per second.

    Returns:
    - List[int]: A single merged token sequence.
    """
    merged_tokens = []
    overlap_tokens = (
        overlap //
        2) * token_rate  # Tokens corresponding to half of the overlap duration

    for i, tokens in enumerate(tokenized_segments):
        l = 0 if i == 0 else overlap_tokens
        r = -overlap_tokens if i != len(tokenized_segments) - 1 else len(
            tokens)
        # Keep only the middle part (drop overlap / 2 from both sides)
        merged_tokens.extend(tokens[l:r])

    return merged_tokens


def onnx2torch(*_a, **_k):
    raise RuntimeError("LC AV: ONNX conversion is not part of the vendored tokenizer.")
