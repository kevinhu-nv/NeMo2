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
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torchaudio

from lhotse import CutSet, MonoCut, Seconds, SupervisionSegment


class ForceAligner:
    """
    Force alignment utility using wav2vec2-based models for speech-to-text alignment.
    """
    
    def __init__(self, device: str = None, frame_length: float = 0.02):
        """
        Initialize the ForceAligner.
        
        Args:
            device: Device to run alignment on (default: auto-detect)
            frame_length: Frame length in seconds for timestamp conversion
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_length = frame_length
        
        # Initialize wav2vec2 model components
        self.wav2vec2_model = None
        self.wav2vec2_tokenizer = None
        self.wav2vec2_aligner = None
        self.wav2vec2_bundle = None
        
        self._load_wav2vec2_model()
    
    def _load_wav2vec2_model(self):
        """Load the wav2vec2 model and related components."""
        try:
            device = torch.device(self.device)
            logging.info(f"Loading wav2vec2 model for force alignment on device {device}")

            # Load wav2vec2 model and bundle (using MMS_FA for multilingual support)
            from torchaudio.pipelines import MMS_FA as bundle
            self.wav2vec2_bundle = bundle
            self.wav2vec2_model = bundle.get_model().to(device)
            self.wav2vec2_tokenizer = bundle.get_tokenizer()
            self.wav2vec2_aligner = bundle.get_aligner()
            
            # Set model to evaluation mode
            self.wav2vec2_model.eval()
            
            logging.info("Wav2vec2 model loaded successfully for force alignment")
        except Exception as e:
            logging.error(f"Failed to load wav2vec2 model for force alignment: {e}")
            self.wav2vec2_model = None
    
    def batch_force_align_user_audio(self, cuts: CutSet, source_sample_rate: int = 16000) -> None:
        """
        Perform batch force alignment on all user audio segments.
        
        Args:
            cuts: CutSet containing all cuts to process
            source_sample_rate: Source sample rate of the audio
        """
        if self.wav2vec2_model is None:
            logging.warning("Wav2vec2 model not available for force alignment, skipping batch alignment")
            return
        
        # Collect all user supervisions and their corresponding cuts
        user_supervisions = []
        user_cuts = []
        
        for cut in cuts:
            for supervision in cut.supervisions:
                if supervision.speaker.lower() == "user":
                    user_supervisions.append(supervision)
                    user_cuts.append(cut)
        
        if not user_supervisions:
            logging.info("No user supervisions found for force alignment")
            return
        
        logging.info(f"Performing force alignment on {len(user_supervisions)} user audio segments")
        
        # Prepare audio tensors and texts for batch processing
        audio_tensors = []
        texts = []
        
        for i, (supervision, cut) in enumerate(zip(user_supervisions, user_cuts)):
            # Extract user audio segment
            start_time = supervision.start
            duration = supervision.duration
            
            # Truncate the cut to get only the user segment
            user_cut = cut.truncate(offset=start_time, duration=duration)
            
            # Load audio and resample to model's expected sample rate
            audio = user_cut.load_audio()
            if audio.shape[0] > 1:  # Convert to mono if stereo
                audio = audio.mean(dim=0, keepdim=True)
            
            # Convert numpy array to torch tensor if needed
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            
            # Resample to wav2vec2's expected sample rate (16kHz)
            target_sample_rate = 16000
            if source_sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=source_sample_rate,
                    new_freq=target_sample_rate
                )
                audio = resampler(audio)
            
            # Add 0.64 seconds of trailing silence
            silence_duration = 0.64  # seconds
            silence_samples = int(silence_duration * target_sample_rate)
            silence = torch.zeros(1, silence_samples)
            audio = torch.cat([audio, silence], dim=1)
            
            # Store audio tensor and text for batch processing
            audio_tensors.append(audio)
            texts.append(self._strip_timestamps(supervision.text))
        
        # Use wav2vec2-based force alignment with in-memory audio tensors
        alignments_batch = self._wav2vec2_batch_align_tensors(audio_tensors, texts)
        
        # Process each alignment result and update supervision texts
        for i, alignment_result in enumerate(alignments_batch):
            if alignment_result is not None:
                # Convert alignment to timestamped text
                original_text = user_supervisions[i].text
                timestamped_text = self._convert_wav2vec2_alignment_to_timestamped_text(alignment_result, original_text)
                # Update the supervision text with force-aligned timestamps
                user_supervisions[i].text = timestamped_text
    
    def _wav2vec2_batch_align_tensors(self, audio_tensors: List[torch.Tensor], texts: List[str]) -> List[Optional[List[Dict[str, Any]]]]:
        """
        Perform batch force alignment using wav2vec2 with in-memory audio tensors.
        
        Args:
            audio_tensors: List of audio waveform tensors
            texts: List of text transcripts corresponding to each audio tensor
            
        Returns:
            List of alignment results for each audio tensor
        """
        alignments = []
        
        for audio_tensor, text in zip(audio_tensors, texts):
            try:
                # Perform alignment directly with the audio tensor
                alignment_result = self._wav2vec2_align(
                    audio_tensor, 
                    16000,  # wav2vec2 expects 16kHz
                    text
                )
                alignments.append(alignment_result)
                
            except Exception as e:
                logging.error(f"Failed to align audio tensor: {e}")
                alignments.append(None)
        
        return alignments
    
    def _wav2vec2_align(self, waveform: torch.Tensor, sample_rate: int, transcript: str) -> Optional[List[Dict[str, Any]]]:
        """
        Perform forced alignment using wav2vec2.
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of the audio
            transcript: Text transcript
            
        Returns:
            List of word segments with timing information
        """
        # Normalize transcript (following the documentation approach)
        normalized_transcript = self._normalize_transcript(transcript)
        
        # Split transcript into words for word-level alignment
        transcript_words = normalized_transcript.split()
        
        if not transcript_words:
            logging.warning(f"No valid words found in transcript: {transcript}")
            return None
        
        # Resample if needed (wav2vec2 expects 16kHz)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        
        # Move to device
        device = torch.device(self.device)
        waveform = waveform.to(device)
        
        # Get emission from wav2vec2 model
        with torch.no_grad():
            emission, _ = self.wav2vec2_model(waveform)
        
        # Tokenize transcript using the bundle's tokenizer
        tokens = self.wav2vec2_tokenizer(transcript_words)
        
        # Perform forced alignment using the bundle's aligner
        token_spans = self.wav2vec2_aligner(emission[0], tokens)
        
        if not token_spans:
            logging.warning(f"No alignment found for transcript: {transcript}")
            return None
        
        # Convert token spans to word segments
        word_segments = []
        ratio = waveform.size(1) / emission.size(1) / 16000  # Convert frames to seconds
        
        for word, spans in zip(transcript_words, token_spans):
            if spans:
                start_time = spans[0].start * ratio
                end_time = spans[-1].end * ratio
                # Calculate average score weighted by span length
                avg_score = sum(span.score * len(span) for span in spans) / sum(len(span) for span in spans)
                
                word_segments.append({
                    'word': word,
                    'start': start_time,
                    'end': end_time,
                    'score': avg_score
                })
        
        return word_segments
    
    def _normalize_transcript(self, transcript: str) -> str:
        """
        Normalize transcript following the documentation approach.
        This removes punctuation and converts to lowercase for the MMS_FA tokenizer.
        """
        # Convert to lowercase
        text = transcript.lower()
        
        # Replace apostrophes
        text = text.replace("'", "'")
        
        # Remove non-alphabetic characters except apostrophes and spaces
        text = re.sub(r"[^a-z' ]", " ", text)
        
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _convert_wav2vec2_alignment_to_timestamped_text(self, alignment_result: List[Dict[str, Any]], original_text: str) -> str:
        """
        Convert wav2vec2 alignment results to timestamped text format.
        
        Args:
            alignment_result: List of word segments with timing information
            original_text: Original text without timestamps
            
        Returns:
            Text with timestamp tokens in the format <|start_frame|>word<|end_frame|>
        """
        timestamped_words = []
        
        for word_seg in alignment_result:
            # Use the word from the alignment result as it represents what was actually aligned
            word = word_seg["word"]
            start_frame = int(word_seg["start"] / self.frame_length)
            end_frame = int(word_seg["end"] / self.frame_length)
            timestamped_words.append(f"<|{start_frame}|> {word} <|{end_frame}|>")
        
        return " ".join(timestamped_words)
    
    def _strip_timestamps(self, text: str) -> str:
        """
        Strip timestamp tokens from text.
        
        Args:
            text: Text that may contain timestamp tokens
            
        Returns:
            Text with timestamp tokens removed
        """
        # Remove timestamp tokens in the format <|frame_number|>
        text = re.sub(r'<\|[0-9]+\|>', '', text)
        
        # Clean up extra spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
