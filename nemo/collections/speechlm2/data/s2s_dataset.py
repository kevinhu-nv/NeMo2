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
import re

import torch
import torch.utils.data
import torchaudio

from lhotse import CutSet, MonoCut, Recording, Seconds, SupervisionSegment, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data import utils
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.utils import logging


class DuplexS2SDataset(torch.utils.data.Dataset):
    """
    A dataset for duplex speech-to-speech models that handles bidirectional conversations.

    This dataset processes Lhotse CutSet objects containing recordings with supervision segments
    from different speakers (roles). It creates aligned representations of audio and text for
    both source (input) and target (output) channels, preserving temporal alignment between
    audio frames and text tokens.

    Args:
        tokenizer (TokenizerSpec):
            Tokenizer for converting text to token IDs and vice versa. Must support BOS and EOS tokens.
            It's expected to support PAD token as well, otherwise we will use 0 as the pad token
            and emit a warning.

        frame_length (Seconds):
            Duration of a single frame in seconds. Used to calculate frame positions for token alignment.

        source_sample_rate (int):
            Sample rate for source audio (e.g., 16000 Hz).

        target_sample_rate (int):
            Sample rate for target audio (e.g., 22050 Hz).

        input_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as inputs. Defaults to ["user"].

        output_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as outputs. Defaults to ["agent"].

    Returns:
        A dictionary with the following keys:
            - source_audio: Tensor of source waveform samples [B, T]
            - source_audio_lens: Tensor of source audio lengths [B]
            - target_audio: Tensor of target waveform samples [B, T]
            - target_audio_lens: Tensor of target audio lengths [B]
            - target_tokens: Tensor of target text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - target_token_lens: Tensor of target token sequence lengths [B]
            - source_tokens: Tensor of source text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - source_token_lens: Tensor of source token sequence lengths [B]
            - target_texts: List of full target texts joined from output_roles supervisions [B]

    Notes:
        - The dataset ensures frame-level alignment between audio and text by inserting tokens at
          specific frame positions based on the timing of supervision segments.
        - PAD tokens (typically 0) are used to fill gaps where there's no text.
        - BOS tokens mark the beginning of each speech segment.
        - EOS tokens mark the end of each speech segment.
        - Text tokens from each speaker are placed at frame positions corresponding to their
          timestamp in the original recording, preserving the temporal relationship.
          This is a segment-level alignment only, not word-level alignment.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        source_sample_rate: int,
        target_sample_rate: int,
        input_roles: list[str] = None,
        output_roles: list[str] = None,
        word_align_position: str = 'left',
        use_vad_for_user_audio: bool = False,
        predict_user_text: bool = False
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        self.word_align_position = word_align_position
        self.use_vad_for_user_audio = use_vad_for_user_audio
        self.predict_user_text = predict_user_text
        
        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."
    
    def _create_minimal_batch(self) -> dict:
        """Create a minimal valid batch when all cuts are filtered out."""
        # Create minimal tensors with batch size 1
        device = torch.device('cpu')  # Default device
        
        return {
            "sample_id": ["empty_batch"],
            "source_audio": torch.zeros((1, 1000), dtype=torch.float32),  # 1 second of silence at 16kHz
            "source_audio_lens": torch.tensor([1000], dtype=torch.long),
            "agent_bos_vad": None,
            "target_audio": torch.zeros((1, 22050), dtype=torch.float32),  # 1 second of silence at 22.05kHz
            "target_audio_lens": torch.tensor([22050], dtype=torch.long),
            "target_tokens": torch.full((1, 50), self.tokenizer.pad_id, dtype=torch.long),
            "target_token_lens": torch.tensor([1], dtype=torch.long),
            "source_tokens": torch.full((1, 50), self.tokenizer.pad_id, dtype=torch.long),
            "source_token_lens": torch.tensor([1], dtype=torch.long),
            "source_texts": [""],
            "target_texts": [""],
            "all_texts": [""],
            "target_first_turn_audio": torch.zeros((1, 22050), dtype=torch.float32),
            "target_first_turn_audio_lens": torch.tensor([22050], dtype=torch.long),
            "formatter": ["s2s_duplex"],
        }

    def __getitem__(self, cuts: CutSet) -> dict:
        # cuts = cuts.transform_text(_strip_timestamps)

        if hasattr(cuts[0], 'formatter') and cuts[0].formatter == 'nemo_tarred_to_duplex':
            filtered_cuts = []
            skipped_cuts = []
            for cut in cuts:
                if any(s.text.strip() for s in cut.supervisions if s.speaker in self.input_roles):
                    filtered_cuts.append(cut)
                else:
                    skipped_cuts.append(cut.id)
            if skipped_cuts:
                logging.info(f"Skipped {len(skipped_cuts)} cuts with empty input text. Skipped cut ids: {', '.join(skipped_cuts)}")
            if not filtered_cuts:
                logging.warning(f"All cuts were filtered out! Original batch size: {len(cuts)}. Returning minimal valid batch to continue training.")
                return self._create_minimal_batch()
            cuts = CutSet.from_cuts(filtered_cuts)
            

        source_audio, source_audio_lens = collate_audio(cuts.resample(self.source_sample_rate))
        target_audio, target_audio_lens = collate_audio(
            cuts.resample(self.target_sample_rate), recording_field="target_audio"
        )
        target_tokens, target_token_lens = collate_token_channel(
            cuts, self.tokenizer, self.frame_length, roles=self.output_roles, bos_id=self.tokenizer.bos, eos_id=self.tokenizer.eos, remove_timestamps=True
        )
        source_tokens, source_token_lens = collate_token_channel(
            cuts, self.tokenizer, self.frame_length, roles=self.input_roles, bos_id=self.tokenizer.text_to_ids('^')[0], eos_id=self.tokenizer.text_to_ids('$')[0], word_align_position=self.word_align_position, remove_timestamps=not self.predict_user_text
        )

        agent_bos_vad = None
        if self.use_vad_for_user_audio:
            # Use VAD utility to create agent turn mask and find transition positions
            is_agent_turn = utils.create_agent_turn_mask_from_vad(
                source_audio, source_tokens, source_token_lens, 
                self.frame_length, self.source_sample_rate,
                vad_model_path='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/silero-vad'
            )
            agent_bos_vad = utils.find_last_zero_to_one_transition(is_agent_turn, source_token_lens)            

        # extract target speaker first turn audio to uses for speaker conditioning
        target_first_turn_audio, target_first_turn_audio_lens = collate_first_turn_audio(
            cuts.resample(self.target_sample_rate), roles=self.output_roles, recording_field="target_audio"
        )

        return {
            "sample_id": [str(cut.id) for cut in cuts],
            "source_audio": source_audio,
            "source_audio_lens": source_audio_lens,
            "agent_bos_vad": agent_bos_vad,
            "target_audio": target_audio,
            "target_audio_lens": target_audio_lens,
            "target_tokens": target_tokens,
            "target_token_lens": target_token_lens,
            "source_tokens": source_tokens,
            "source_token_lens": source_token_lens,
            "source_texts": [
                " ".join(_strip_timestamps(s.text) for s in cut.supervisions if s.speaker in self.input_roles) for cut in cuts
            ],
            "target_texts": [
                " ".join(_strip_timestamps(s.text) for s in cut.supervisions if s.speaker in self.output_roles) for cut in cuts
            ],
            "all_texts": [
                " ".join(_strip_timestamps(s.text) for s in cut.supervisions) for cut in cuts
            ],
            "target_first_turn_audio": target_first_turn_audio,
            "target_first_turn_audio_lens": target_first_turn_audio_lens,
            "formatter": [getattr(cut, "formatter", "s2s_duplex") for cut in cuts],
        }


def collate_first_turn_audio(
    cuts: CutSet,
    roles: set[str],
    recording_field: str = "target_audio",
) -> tuple[torch.Tensor, torch.Tensor]:
    first_turn_audios = []
    first_turn_audios_lens = []
    for cut in cuts:
        # Find supervisions that match the specified roles
        matching_supervisions = [s for s in cut.supervisions if s.speaker in roles]
        
        if not matching_supervisions:
            # Log warning and skip this cut if no matching supervisions found
            logging.warning(f"No supervisions found with roles {roles} for cut {cut.id}. Available speakers: {[s.speaker for s in cut.supervisions]}")
            continue
            
        first_supervision = matching_supervisions[0]
        truncated_audio = cut.truncate(offset=max(0, first_supervision.start), duration=first_supervision.duration).load_custom(recording_field)
        first_turn_audios.append(truncated_audio.squeeze(0))
        first_turn_audios_lens.append(truncated_audio.shape[-1])

    if not first_turn_audios:
        # If no valid audio was found, return empty tensors
        logging.error(f"No valid audio found for any cuts with roles {roles}")
        return torch.empty(0), torch.empty(0)

    return collate_vectors(first_turn_audios, padding_value=0), torch.tensor(first_turn_audios_lens)


def collate_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    bos_id: int = None,
    eos_id: int = None,
    word_align_position: str = 'left',
    remove_timestamps: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = get_pad_id(tokenizer)
    tokens = [
        build_token_channel(c, tokenizer=tokenizer, frame_length=frame_length, roles=roles, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, word_align_position=word_align_position, remove_timestamps=remove_timestamps)
        for c in cuts
    ]
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def build_token_channel(
        cut: Cut,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        roles: set[str],
        pad_id: int = -1,
        bos_id: int = None,
        eos_id: int = None,
        word_align_position: str = 'left',
        remove_timestamps: bool = False,
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id
    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            logging.info(f'text: {supervision.text}, role: {supervision.speaker}')

            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos >= len(tokens):  # Changed from > to >= for robustness
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {pos} is larger than or equal to the example's length {len(tokens)}. {diagnostic}"
                )
                continue
            eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
            available_frames_for_text = eospos - pos

            # Use different bos_id for user and agent
            text_ids = torch.as_tensor([bos_id] + _text_to_ids(supervision.text, tokenizer, available_frames_for_text=available_frames_for_text, word_align_position=word_align_position, remove_timestamps=remove_timestamps))


            if available_frames_for_text > 0 and len(text_ids) > available_frames_for_text:
                # Truncate text_ids to fit before the eos position.
                text_ids = text_ids[:available_frames_for_text]
            elif available_frames_for_text <= 0:
                # If there's no space for text (e.g., start >= end), use an empty sequence.
                text_ids = torch.tensor([], dtype=torch.long)

            endpos = pos + len(text_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - pos
                logging.warning(
                    f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
                )
                text_ids = text_ids[:trunc_len]
                endpos = pos + len(text_ids)  

            try:
                tokens[pos:endpos] = text_ids
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e

            if eospos < len(tokens) and eos_id is not None:
                # Only assign eos_id to target tokens, user eos is merged with agent bos
                tokens[eospos] = eos_id

            logging.info(f'text_ids w/ bos and eos: {tokens[pos:eospos+1]}')

    return tokens


def _strip_timestamps(
    text: str, _TIMESTAMP_PATTERN=re.compile(r"<\|\d+\|>"), _SPACE_PATTERN=re.compile(r"\s+")
) -> str:
    """
    Strips timestamp tokens from text, e.g. turns:
      '<|0|> Hey <|3|> <|3|> how <|5|> <|7|> are <|8|> <|8|> <|10|> you? <|12|>'
      into:
      'Hey how are you?'
    """
    # Regexp pattern args are cached compiled patterns (micro-optimization).
    text = _TIMESTAMP_PATTERN.sub("", text)  # strip timestamp tokens if present
    return _SPACE_PATTERN.sub(" ", text).strip()  # strip multi-whitespaces


def _text_to_ids(text: str, tokenizer: TokenizerSpec,
                 _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>",
                 available_frames_for_text=None,
                 word_align_position='left',
                 remove_timestamps=False):
    _TIMESTAMP_PATTERN = re.compile(_TIMESTAMP_PATTERN_STR)
    if _TIMESTAMP_PATTERN.search(text) and not remove_timestamps:
        text_ids = _text_with_timestamps_to_ids(text, tokenizer, _TIMESTAMP_PATTERN_STR, available_frames_for_text, word_align_position)
    else:
        text_ids = tokenizer.text_to_ids(text)
    return text_ids


def _text_with_timestamps_to_ids(text: str, tokenizer: TokenizerSpec,
                                 _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>",
                                 available_frames_for_text=None,
                                 word_align_position='left') -> list[int]:
    text_ids = []
    text_ids, start_times, end_times, word_lens = _extract_text_and_time_tokens(text, tokenizer, _TIMESTAMP_PATTERN_STR)
    text_ids_with_timestamps = _expand_text_with_timestamps_and_word_lengths(text_ids, word_lens, start_times, end_times, available_frames_for_text, frame_rate=0.08, pad_id=get_pad_id(tokenizer), word_align_position=word_align_position)
    logging.info(f'text_ids_with_timestamps: {text_ids_with_timestamps}')
    logging.info(f'text_ids: {text_ids}')
    logging.info(f'start_times: {start_times}')
    logging.info(f'end_times: {end_times}')
    logging.info(f'word_lens: {word_lens}')
    return text_ids_with_timestamps


def _extract_text_and_time_tokens(text, tokenizer: TokenizerSpec,
                                 _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>"):
    # Find all time tokens
    time_tokens = re.findall(_TIMESTAMP_PATTERN_STR, text)
    start_time = [int(time_tokens[i]) for i in range(0, len(time_tokens), 2)]
    end_time = [int(time_tokens[i]) for i in range(1, len(time_tokens), 2)]
    # Remove all time tokens to isolate words
    words = re.sub(_TIMESTAMP_PATTERN_STR, '', text).split()
    # Process each word, tokenize it, and calculate token lengths
    text_ids = []
    word_lens = []
    for i, word in enumerate(words):
        word_with_space = word if i == 0 else ' ' + word
        word_ids = tokenizer.text_to_ids(word_with_space)
        word_len = len(word_ids)
        text_ids.extend(word_ids)
        word_lens.append(word_len)
    return text_ids, start_time, end_time, word_lens


def _expand_text_with_timestamps_and_word_lengths(
        text_ids, word_lens, start_time, end_time, available_frames_for_text, frame_rate=0.08, pad_id=None, word_align_position='left'
    ):    
    """
    Expand word tokens according to start time tokens and word lengths for a batch of sequences.

    Args:
    - word_tokens: List of text ids w/o timestamps
    - word_lens: List of word lengths
    - start_time: List of start times
    - end_time: List of end times
    - available_frames_for_text: Maximum number of frames for text
    - frame_rate: Frame rate resolution
    - pad_id: Padding ID to use for empty positions in the tensor

    Returns:
    - text ids with word-level timestamps
    """

    def discretize_time(start_token, speech_frame_rate=0.08, timestamp_frame_rate=0.08):
        return int(start_token * timestamp_frame_rate / speech_frame_rate)

    if pad_id is None:
        raise ValueError("pad_id must be provided.")

    max_length = available_frames_for_text

    # Create the empty tensor with pad_id as the default value
    text_ids_with_timestamps = [pad_id] * max_length

    # Populate ids of each word starting at start_idx and ending at end_idx
    cur_word_idx = 0  # Start frame index of current word
    for word_idx, word_len in enumerate(word_lens):
        start_idx = discretize_time(start_time[word_idx], speech_frame_rate=frame_rate)
        end_idx = discretize_time(end_time[word_idx], speech_frame_rate=frame_rate)
        if word_align_position == 'left':
            end_idx = min(start_idx + word_len, end_idx)
        elif word_align_position == 'right':
            start_idx = max(start_idx, end_idx - word_len)
        else:
            raise ValueError(f"Unknown word_align_position: {word_align_position}")

        # Get ids of a single word
        word_ids = text_ids[cur_word_idx : cur_word_idx + word_len]

        # Populate a single word
        for i in range(start_idx, end_idx + 1):  # End inclusive at word level
            if i - start_idx < len(word_ids) and i < max_length:
                token_id = word_ids[i - start_idx]
                text_ids_with_timestamps[i] = token_id

        # Move to the next word in the concatenated word tokens
        cur_word_idx += word_len

    return text_ids_with_timestamps