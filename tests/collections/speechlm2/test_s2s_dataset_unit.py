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

import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut, dummy_recording

from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.speechlm2.data.s2s_dataset import DuplexS2SDataset, build_token_channel


class MockTokenizer:
    """Mock tokenizer for testing without downloading models"""
    
    def __init__(self):
        self.bos = 1
        self.eos = 2
        self.pad = 0
        self.vocab_size = 1000
        
    def text_to_ids(self, text):
        # Simple mock tokenization: convert each word to a number
        words = text.split()
        return [hash(word) % 1000 for word in words]


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def dataset(mock_tokenizer):
    return DuplexS2SDataset(
        mock_tokenizer,
        frame_length=0.08,
        source_sample_rate=16000,
        target_sample_rate=22050,
        input_roles=["user"],
        output_roles=["assistant"],
    )


@pytest.fixture
def training_cutset_batch():
    cut = dummy_cut(0, recording=dummy_recording(0, with_data=True))
    cut.target_audio = dummy_recording(1, with_data=True)
    cut.supervisions = [
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0,
            duration=0.1,
            text='hi',
            speaker="user",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.3,
            duration=0.1,
            text='hello',
            speaker="assistant",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.5,
            duration=0.1,
            text='ok',
            speaker="user",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.6,
            duration=0.4,
            text='okay',
            speaker="assistant",
        ),
    ]
    return CutSet([cut])


def test_build_token_channel_basic(mock_tokenizer):
    """Test build_token_channel function with basic input"""
    cut = dummy_cut(0, recording=dummy_recording(0, with_data=True))
    cut.supervisions = [
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.1,
            duration=0.2,
            text='hello world',
            speaker="assistant",
        )
    ]
    
    tokens = build_token_channel(
        cut=cut,
        tokenizer=mock_tokenizer,
        frame_length=0.08,
        roles={"assistant"},
        pad_id=0
    )
    
    # Check basic properties
    assert isinstance(tokens, torch.Tensor)
    assert tokens.dtype == torch.long
    
    # Calculate expected number of frames
    total_frames = int(cut.duration / 0.08)
    assert tokens.shape == (total_frames,)
    
    # Check that BOS token is at the start of speech
    speech_start_frame = int(0.1 / 0.08)
    assert tokens[speech_start_frame] == mock_tokenizer.bos
    
    # Check that EOS token is at the end of speech
    speech_end_frame = int(0.3 / 0.08)
    assert tokens[speech_end_frame] == mock_tokenizer.eos


def test_build_token_channel_multiple_speakers(mock_tokenizer):
    """Test build_token_channel with multiple speakers"""
    cut = dummy_cut(0, recording=dummy_recording(0, with_data=True))
    cut.supervisions = [
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.0,
            duration=0.1,
            text='hi',
            speaker="user",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.2,
            duration=0.1,
            text='hello',
            speaker="assistant",
        ),
    ]
    
    # Test for assistant role only
    tokens = build_token_channel(
        cut=cut,
        tokenizer=mock_tokenizer,
        frame_length=0.08,
        roles={"assistant"},
        pad_id=0
    )
    
    # Should only have assistant speech tokens
    speech_start_frame = int(0.2 / 0.08)
    speech_end_frame = int(0.3 / 0.08)
    
    assert tokens[speech_start_frame] == mock_tokenizer.bos
    assert tokens[speech_end_frame] == mock_tokenizer.eos
    
    # User speech should be padded
    user_start_frame = int(0.0 / 0.08)
    assert tokens[user_start_frame] == 0  # pad_id


def test_dataset_basic_functionality(dataset, training_cutset_batch):
    """Test DuplexS2SDataset basic functionality"""
    batch = dataset[training_cutset_batch]
    
    # Check required keys
    required_keys = [
        "source_audio", "target_audio", "source_audio_lens", "target_audio_lens",
        "target_tokens", "target_token_lens", "source_tokens", "source_token_lens",
        "target_texts", "sample_id", "formatter"
    ]
    
    for key in required_keys:
        assert key in batch, f"Missing key: {key}"
    
    # Check tensor types
    tensor_keys = [
        "source_audio", "target_audio", "source_audio_lens", "target_audio_lens",
        "target_tokens", "target_token_lens", "source_tokens", "source_token_lens"
    ]
    
    for key in tensor_keys:
        assert torch.is_tensor(batch[key]), f"{key} is not a tensor"
    
    # Check audio shapes
    assert batch["source_audio"].shape[0] == 1  # batch size
    assert batch["target_audio"].shape[0] == 1  # batch size
    
    # Check token shapes
    assert batch["target_tokens"].shape[0] == 1  # batch size
    assert batch["source_tokens"].shape[0] == 1  # batch size
    
    # Check target texts
    assert isinstance(batch["target_texts"], list)
    assert len(batch["target_texts"]) == 1
    assert "hello" in batch["target_texts"][0]  # assistant speech
    assert "okay" in batch["target_texts"][0]   # assistant speech


def test_dataset_token_alignment(dataset, training_cutset_batch):
    """Test that tokens are properly aligned with audio frames"""
    batch = dataset[training_cutset_batch]
    
    # Get the first (and only) sample
    target_tokens = batch["target_tokens"][0]  # Remove batch dimension
    source_tokens = batch["source_tokens"][0]  # Remove batch dimension
    
    # Check that tokens have the same length (same number of frames)
    assert target_tokens.shape == source_tokens.shape
    
    # Check that we have some non-pad tokens (actual speech)
    assert (target_tokens != 0).any(), "No non-pad tokens found in target"
    assert (source_tokens != 0).any(), "No non-pad tokens found in source"


def test_dataset_edge_cases(dataset):
    """Test edge cases like empty supervision or very short audio"""
    # Test with empty supervision
    cut = dummy_cut(0, recording=dummy_recording(0, with_data=True))
    cut.target_audio = dummy_recording(1, with_data=True)
    cut.supervisions = []  # No supervision segments
    
    cuts = CutSet([cut])
    batch = dataset[cuts]
    
    # Should still return valid tensors
    assert torch.is_tensor(batch["target_tokens"])
    assert torch.is_tensor(batch["source_tokens"])
    
    # All tokens should be pad tokens
    assert (batch["target_tokens"] == 0).all()
    assert (batch["source_tokens"] == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 