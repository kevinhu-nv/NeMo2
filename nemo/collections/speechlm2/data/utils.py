# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import warnings
import torch
from torch.nn.utils.rnn import pad_sequence


def get_pad_id(tokenizer) -> int:
    pad_id = tokenizer.pad
    if pad_id is not None:
        return pad_id
    pad_id = tokenizer.unk_id
    if pad_id is not None:
        return pad_id
    warnings.warn(
        "The text tokenizer has no <pad> or <unk> tokens available, using ID 0 for padding (this may lead to silent bugs)."
    )
    return 0


def collate_and_pad_1d(data, pad_id=-1):
    """
    Collate and pad 1D sequences for batch processing.

    Args:
        data (List[List[int]] or List[torch.Tensor]): List of sequences.
        pad_id (int, optional): Padding value. Defaults to -1.

    Returns:
        torch.Tensor: Padded tensor of shape [batch_size, max_sequence_length]
    """
    if not data:
        return torch.tensor([])

    # Fast path: if all are already tensors, skip conversion
    if all(isinstance(seq, torch.Tensor) for seq in data):
        tensor_data = data
    else:
        # Convert to tensors using list comprehension (faster than loop)
        tensor_data = [seq if isinstance(seq, torch.Tensor) else torch.tensor(seq, dtype=torch.long) for seq in data]

    # Pad sequences
    padded = pad_sequence(tensor_data, batch_first=True, padding_value=pad_id)
    return padded


def collate_and_pad_2d(tensors, pad_id):
    """
    Collate and pad 2D tensors for batch processing.

    Args:
        tensors (List[torch.Tensor]): List of 2D tensors with varying shapes.
        pad_id (int): Padding value.

    Returns:
        torch.Tensor: Padded tensor of shape [batch_size, max_rows, max_cols]
    """
    if not tensors:
        return torch.tensor([])

    # Find max dimensions in a single pass
    batch_size = len(tensors)
    max_rows = 0
    max_cols = 0
    for t in tensors:
        max_rows = max(max_rows, t.shape[0])
        max_cols = max(max_cols, t.shape[1])

    # Create padded tensor on the same device as input tensors
    first_tensor = tensors[0]
    padded = torch.full(
        (batch_size, max_rows, max_cols),
        pad_id,
        dtype=first_tensor.dtype,
        device=first_tensor.device
    )

    # Fill with actual data (in-place copy is fast)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0], :t.shape[1]] = t

    return padded


def collate_and_pad(inputs, text_pad_id):
    """
    Collate and pad sequences.

    Args:
        inputs: List of tensors to collate
        text_pad_id: Padding ID for text

    Returns:
        Tuple of (padded_tokens, token_lengths)
    """
    if not inputs:
        return torch.tensor([]), torch.tensor([])

    # Compute lengths and pad in a more efficient way
    # Use tensor instead of list for better memory and GPU transfer
    token_lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)
    tokens = pad_sequence(inputs, batch_first=True, padding_value=text_pad_id)
    return tokens, token_lengths
