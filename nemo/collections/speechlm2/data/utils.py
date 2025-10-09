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
import logging
import torch
from lhotse import CutSet


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


def find_last_zero_to_one_transition(mask: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """
    Find the last position where mask transitions from 0 to 1 for each batch.
    
    Args:
        mask: Binary mask tensor of shape [B, T]
        lens: Length tensor of shape [B]
        
    Returns:
        Tensor of shape [B] with transition positions, defaults to lens[b] if no transition found
    """
    B, T = mask.shape
    transition_pos = lens.clone()
    for b in range(B):
        transition_pos[b] = lens[b].item()  # Default to end if no transition found
        for t in range(1, T):  # Start from 1 to check previous position
            if mask[b, t] == 1 and mask[b, t-1] == 0:
                transition_pos[b] = t
    return transition_pos


def create_agent_turn_mask_from_vad(source_audio: torch.Tensor, source_tokens: torch.Tensor, 
                                   source_token_lens: torch.Tensor, frame_length: float, 
                                   source_sample_rate: int, vad_model_path: str = None) -> torch.Tensor:
    """
    Create agent turn mask using VAD to identify user speech regions.
    
    Args:
        source_audio: Audio tensor of shape [B, T]
        source_tokens: Token tensor of shape [B, T]
        source_token_lens: Token length tensor of shape [B]
        frame_length: Duration of a single frame in seconds
        source_sample_rate: Sample rate of the audio
        vad_model_path: Path to local Silero VAD model (optional)
        
    Returns:
        Binary mask tensor of shape [B, T] where 1 = agent turn, 0 = user speech
    """
    # Load VAD model
    if vad_model_path:
        model, utils = torch.hub.load(vad_model_path, 'silero_vad', source='local', trust_repo=True, force_reload=False)
    else:
        model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True, force_reload=False)
    
    get_speech_timestamps, _, read_audio, _, _ = utils
    
    # Create mask with same shape as source_tokens [B, T]
    is_agent_turn = torch.ones_like(source_tokens)
    
    for batch_idx in range(source_audio.shape[0]):
        wav = source_audio[batch_idx, :]
        speech_timestamps = get_speech_timestamps(
            wav, model, sampling_rate=source_sample_rate,
            threshold=0.75, min_speech_duration_ms=1000, 
            min_silence_duration_ms=600, speech_pad_ms=80, 
            return_seconds=True
        )
        
        for segment in speech_timestamps:
            # Convert audio timestamps to token positions
            start_time = segment['start']
            end_time = segment['end']
            
            # Convert time to token frame indices
            start_frame = int(start_time / frame_length)
            end_frame = int(end_time / frame_length)
            
            # Ensure frame indices are within bounds
            start_frame = max(0, min(start_frame, source_tokens.shape[1] - 1))
            end_frame = max(0, min(end_frame, source_tokens.shape[1] - 1))
            
            # Mark user speech regions as 0 (not agent turn)
            is_agent_turn[batch_idx, start_frame:end_frame + 1] = 0.0
    
    return is_agent_turn
