import torch
import yaml
from omegaconf import OmegaConf, DictConfig
import numpy as np
import librosa
import time
from transformers import DynamicCache
import re
import os
import sys
import argparse
import torchaudio
import functools
from typing import Optional
from nemo.utils import logging


# Monkeypatch NeMo logger to correctly report caller stack frame
# (otherwise just prints "nemo_logging:393" for almost everything
# - we should fix this in NeMo long-term)
def _patch_logger_stacklevel(func):
    @functools.wraps(func)
    def wrapper(msg, *args, **kwargs):
        kwargs.setdefault('stacklevel', 3)
        return func(msg, *args, **kwargs)
    return wrapper

logging.debug = _patch_logger_stacklevel(logging.debug)
logging.info = _patch_logger_stacklevel(logging.info)
logging.warning = _patch_logger_stacklevel(logging.warning)
logging.error = _patch_logger_stacklevel(logging.error)
logging.critical = _patch_logger_stacklevel(logging.critical)


# Update the sys path for your environment
BASE_DIR = "/lustre/fsw/portfolios/llmservice/users/cchen1/code/s2s_eartts"
CODE_DIR_NEW = f"{BASE_DIR}/NeMo"
sys.path.insert(0, CODE_DIR_NEW)

# Set environment variables
os.environ["HF_HOME"] = "/lustre/fsw/portfolios/llmservice/nanos/cchen1/hfcache"
os.environ["TORCH_HOME"] = "/lustre/fsw/portfolios/llmservice/nanos/cchen1/hfcache"
os.environ["NEMO_CACHE_DIR"] = "/lustre/fsw/portfolios/llmservice/nanos/cchen1/hfcache"

from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat

from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.audio.parts.utils.resampling import resample

# --- Configuration ---
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Streaming Parameters ---
SAMPLE_RATE = 16000
FRAME_SIZE_SEC = 0.08  # 80ms per frame
FRAME_SIZE_SAMPLES = int(SAMPLE_RATE * FRAME_SIZE_SEC)  # 1280 samples

TTS_SAMPLE_RATE = 22050


# Default hyper-parameters that can be overridden via `model_cfg`
DEFAULT_BUFFER_SIZE_FRAMES = 70
DEFAULT_NUM_FRAMES_PER_INFERENCE = 1
DEFAULT_CODEC_TOKEN_HISTORY_SIZE = 60



class RealtimeStreamingInference:
    """
    Realtime streaming inference that simulates microphone data capture.
    Uses a sliding window buffer and processes audio frame by frame.
    """
    
    def __init__(self, model_cfg: DictConfig):
        """
        Initialize the model for realtime streaming inference.
        
        Args:
            model_cfg (DictConfig): Configuration describing the model paths and runtime parameters.
        """
        if model_cfg is None:
            raise ValueError("model_cfg must be provided")
        if not isinstance(model_cfg, DictConfig):
            model_cfg = OmegaConf.create(model_cfg)

        self.model_cfg = model_cfg

        self.model_path = model_cfg.get("model_path")
        if not self.model_path:
            raise ValueError("`model_cfg.model_path` must be provided.")

        self.llm_checkpoint_path = model_cfg.get("llm_checkpoint_path")
        if not self.llm_checkpoint_path:
            raise ValueError("`model_cfg.llm_checkpoint_path` must be provided.")

        self.decode_audio = bool(model_cfg.get("decode_audio", True))
        self.codec_token_history_size = int(
            model_cfg.get("codec_token_history_size", DEFAULT_CODEC_TOKEN_HISTORY_SIZE)
        )

        self.speaker_reference = model_cfg.get("speaker_reference")
        if self.decode_audio and not self.speaker_reference:
            raise ValueError("`model_cfg.speaker_reference` must be provided when decode_audio is enabled.")

        compute_dtype = model_cfg.get("compute_dtype", "bfloat16")
        self.dtype = self._resolve_dtype(compute_dtype)

        self.device = self._resolve_device(
            device=model_cfg.get("device"),
            device_id=model_cfg.get("device_id"),
        )

        #logging.setLevel(logging.DEBUG)

        logging.info("=" * 70)
        logging.info("INITIALIZING REALTIME STREAMING INFERENCE")
        logging.info("=" * 70)
        logging.info(f"Frame size: {FRAME_SIZE_SEC}s ({FRAME_SIZE_SAMPLES} samples @ {SAMPLE_RATE}Hz)")
        logging.info(f"Device: {self.device}")
        logging.info(f"Compute dtype: {self.dtype}")
        logging.info(f"Decode audio: {self.decode_audio}")
        logging.info("=" * 70)
        
        # Cached TTS helpers populated during initialization/warmup
        self.first_context_subword_id = None
        self.generation_config = None
        self.first_tts_code_input = None
        self.first_tts_past_key_values_input = None


        self.model = None
        self.tokenizer = None
        
        self._initialize_model()

        logging.info(f"\n✅ RealtimeStreamingInference initialized successfully.")
    
    @staticmethod
    def _resolve_dtype(compute_dtype):
        if isinstance(compute_dtype, torch.dtype):
            return compute_dtype
        if compute_dtype is None:
            return torch.bfloat16
        if isinstance(compute_dtype, str):
            key = compute_dtype.lower()
            mapping = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "full": torch.float32,
            }
            if key in mapping:
                return mapping[key]
        raise ValueError(f"Unsupported compute_dtype: {compute_dtype}")

    @staticmethod
    def _resolve_device(device=None, device_id=None):
        if isinstance(device, torch.device):
            resolved_device = device
        else:
            if device is None:
                resolved_device = DEFAULT_DEVICE
            else:
                device_str = str(device)
                base = device_str
                if device_id is not None and device_str.startswith("cuda") and ":" not in device_str:
                    base = f"{device_str}:{device_id}"
                resolved_device = torch.device(base)
        return resolved_device

    def _samples_per_audio_output_frame(self):
        rate = getattr(self, "target_sample_rate", None)
        if rate is None:
            cfg_rate = None
            try:
                cfg_rate = self.model_cfg.get("tts_sample_rate", None)
            except Exception:
                cfg_rate = None
            if cfg_rate is None:
                try:
                    cfg_rate = self.model_cfg.get("output_sample_rate", None)
                except Exception:
                    cfg_rate = None
            if cfg_rate is not None:
                rate = float(cfg_rate)
        if rate is None:
            rate = TTS_SAMPLE_RATE
        samples = int(float(rate) * FRAME_SIZE_SEC)
        return samples
        
    def _load_and_merge_configs(self):
        """Load and merge configurations from both nano and eartts checkpoints."""
        logging.info("\n📋 Loading and merging configurations...")
        
        # Load nano's config (for LLM, perception)
        nano_config_file = os.path.join(self.llm_checkpoint_path, "config.json")
        logging.info(f"  Loading nano config: {nano_config_file}")
        with open(nano_config_file, 'r') as f:
            import json
            nano_cfg_dict = json.load(f)
        nano_cfg = DictConfig(nano_cfg_dict)
        
        # Load eartts's config (for TTS)
        eartts_config_file = os.path.join(self.model_path, "config.json")
        logging.info(f"  Loading eartts config: {eartts_config_file}")
        with open(eartts_config_file, 'r') as f:
            eartts_cfg_dict = json.load(f)
        eartts_cfg = DictConfig(eartts_cfg_dict)
        
        # Start with nano's config as base
        merged_cfg = nano_cfg
        
        # Override TTS-related parts with eartts's config
        logging.info("  Merging: Using nano's config for LLM/perception, eartts's for TTS")
        if 'model' in eartts_cfg and 'speech_generation' in eartts_cfg.model:
            merged_cfg.model.speech_generation = eartts_cfg.model.speech_generation
            logging.info("    ✓ TTS config from eartts")
        
        # Set speaker reference
        if 'model' not in merged_cfg:
            merged_cfg.model = {}
        merged_cfg.model.inference_speaker_reference = self.speaker_reference
        
        # Ensure data section has correct sample rates
        if 'data' not in merged_cfg:
            merged_cfg.data = eartts_cfg.data
        
        logging.info(f"  Final config:")
        logging.info(f"    - pretrained_llm: {merged_cfg.model.stt.model.pretrained_llm}")
        logging.info(f"    - perception.d_model: {merged_cfg.model.stt.model.perception.modality_adapter.d_model}")
        logging.info(f"    - speech_generation: {'present' if 'speech_generation' in merged_cfg.model else 'missing'}")
        
        return merged_cfg
        
    def _initialize_model(self):
        """Initialize the NemotronVoiceChat with hybrid loading."""
        from safetensors.torch import load_file
        from nemo.collections.speechlm2.parts.pretrained import set_model_dict_for_partial_init
        
        logging.info("\n🚀 Initializing model with hybrid loading strategy...")

        
        # Step 1: Load and merge configs
        cfg = self._load_and_merge_configs()
        
        # Step 2: DO NOT set pretrained_s2s_model - we'll load weights manually
        cfg.model.stt.model.pretrained_s2s_model = None
        cfg.model.speech_generation.model.pretrained_model = None

        # Convert to dict for model initialization
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Step 3: Initialize model structure
        logging.info("\n🏗️  Initializing model structure...")
        start_DuplexS2S_init = time.time()
        self.model = NemotronVoiceChat(cfg_dict)
        logging.info(f"🕒 Time taken to initialize NemotronVoiceChat: {time.time() - start_DuplexS2S_init} seconds")
        logging.info("  ✓ Model structure initialized")
        
        # Step 4: Load nano's checkpoint (LLM + perception)
        if self.llm_checkpoint_path is not None:
            logging.info(f"\n📦 Loading LLM + perception:")
            logging.info(f"  Path: {self.llm_checkpoint_path}")
            
            nano_state_dict = load_file(os.path.join(self.llm_checkpoint_path, "model.safetensors"))
            
            # Filter to non-TTS weights
            tts_keys = ['tts_model.', 'speech_generation.']
            nano_filtered = {k: v for k, v in nano_state_dict.items() 
                           if not any(k.startswith(prefix) for prefix in tts_keys)}

            logging.info(f"  Loading {len(nano_filtered)} parameters (excluding TTS)...")

            nano_filtered = set_model_dict_for_partial_init(nano_filtered, self.model.state_dict())
            missing, unexpected = self.model.load_state_dict(nano_filtered, strict=False)
            
            missing_non_tts = [k for k in missing if not any(k.startswith(prefix) for prefix in tts_keys)]
            unexpected_non_tts = [k for k in unexpected if not any(k.startswith(prefix) for prefix in tts_keys)]
            
            if missing_non_tts:
                logging.info(f"  ⚠️  {len(missing_non_tts)} non-TTS keys missing (might be OK)")
            if unexpected_non_tts:
                logging.info(f"  ⚠️  {len(unexpected_non_tts)} unexpected non-TTS keys")
        
        # Step 5: Load eartts's checkpoint (TTS only)
        if self.model_path is not None:
            logging.info(f"\n📦 Loading TTS checkpoint:")
            logging.info(f"  Path: {self.model_path}")
            
            eartts_state_dict = load_file(os.path.join(self.model_path, "model.safetensors"))

            # Filter to only TTS weights
            tts_keys_filter = ['tts_model.']
            eartts_tts_only = {k: v for k, v in eartts_state_dict.items() 
                                 if any(k.startswith(prefix) for prefix in tts_keys_filter)}
            
            logging.info(f"  Loading {len(eartts_tts_only)} TTS parameters...")

            start_tts_load_state_dict = time.time()
            missing, unexpected = self.model.load_state_dict(eartts_tts_only, strict=False)
            logging.info(f"🕒 Time taken to load TTS state dict: {time.time() - start_tts_load_state_dict} seconds")
            
            missing_tts = [k for k in missing if any(k.startswith(prefix) for prefix in tts_keys_filter)]
            unexpected_tts = [k for k in unexpected if any(k.startswith(prefix) for prefix in tts_keys_filter)]
            
            if missing_tts:
                logging.info(f"  ⚠️  {len(missing_tts)} TTS keys missing")
            if unexpected_tts:
                logging.info(f"  ⚠️  {len(unexpected_tts)} unexpected TTS keys")

            logging.info(f"  ✓ eartts checkpoint loaded (TTS only)")

        logging.info("\n✅ Hybrid loading completed!")

        # Setup model
        self.model.to(self.device)
        self.model.eval()
        
        # Convert only the S2S components to the configured dtype, not the TTS model
        logging.info(f"Converting S2S components to {self.dtype} (keeping TTS in float32)...")
        self.model.stt_model.llm = self.model.stt_model.llm.to(self.dtype)
        self.model.stt_model.lm_head = self.model.stt_model.lm_head.to(self.dtype)
        self.model.stt_model.embed_tokens = self.model.stt_model.embed_tokens.to(self.dtype)
        self.model.stt_model.asr_head = self.model.stt_model.asr_head.to(self.dtype)
        self.model.stt_model.embed_asr_tokens = self.model.stt_model.embed_asr_tokens.to(self.dtype)
        self.model.stt_model.perception = self.model.stt_model.perception.to(self.dtype)
        logging.info("✓ S2S components converted, TTS kept in float32")

        self.model.on_train_epoch_start()
        self.tokenizer = self.model.stt_model.tokenizer

        # Get TTS info
        if hasattr(self.model, 'tts_model'):
            self.target_fps = self.model.tts_model.target_fps
            self.target_sample_rate = self.model.tts_model.target_sample_rate
            logging.info(f"\nTTS model initialized: target_fps={self.target_fps}, sample_rate={self.target_sample_rate}")
            if self.decode_audio:
                self._prepare_tts_initial_state()
        else:
            logging.warning("Warning: TTS model not found in the model")
            
    def _get_bos_embedding(self):
        """Get beginning of sequence embedding."""
        text_bos = torch.full((1,), fill_value=self.model.stt_model.text_pad_id, device=self.device)
        input_embeds = self.model.stt_model.embed_tokens(text_bos)
        return input_embeds.to(dtype=self.dtype)

    def _get_asr_bos_embedding(self) -> torch.Tensor:
        """Get ASR BOS embedding for AR decoding."""
        text_bos = torch.full((1,), fill_value=self.model.stt_model.text_pad_id, device=self.device)
        input_embeds = self.model.stt_model.embed_asr_tokens(text_bos)
        return input_embeds.to(dtype=self.dtype)

    def _clone_cache(self, cache):
        """Deep clone cache structures to ensure complete isolation between streams."""
        if cache is None:
            return None
        if isinstance(cache, torch.Tensor):
            return cache.detach().clone()
        if isinstance(cache, (list, tuple)):
            return type(cache)(self._clone_cache(x) for x in cache)
        if isinstance(cache, dict):
            return {k: self._clone_cache(v) for k, v in cache.items()}
        # Handle complex objects (e.g., DynamicCache with __dict__ attributes)
        # Use deepcopy to ensure complete isolation between streams
        if hasattr(cache, '__dict__'):
            import copy
            return copy.deepcopy(cache)
        return cache

    def _prepare_tts_initial_state(self):
        if not self.decode_audio:
            return
        if not hasattr(self.model, 'tts_model'):
            return

        logging.info("\n🎯 Preparing TTS warmup state...")

        with fp32_precision():
            speaker_audio, speaker_sr = torchaudio.load(self.speaker_reference)
            speaker_audio = resample(speaker_audio, speaker_sr, self.model.tts_model.target_sample_rate)

        speaker_audio = speaker_audio.to(self.device)
        speaker_audio_lens = torch.tensor([speaker_audio.size(1)], device=self.device).long()

        #  init tts_model
        self.model.tts_model.set_init_inputs(
            speaker_audio=speaker_audio,
            speaker_audio_lens=speaker_audio_lens,
        )
        init_inputs = self.model.tts_model.get_init_inputs(B=1)

        self.generation_config = self.model.tts_model._get_generation_config(guidance_enabled=True)
        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": True})

        with torch.no_grad():
            outputs = self.model.tts_model.tts_model(**init_inputs)
            code = init_inputs["code"][:, -1:]
            # code, _, _ = self.model.tts_model.tts_model.generate_step(
            #     outputs.hidden_states[:, -1:], **self.generation_config
            # )

        self.first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)
        self.first_tts_code_input = code.detach().clone()
        self.first_tts_past_key_values_input = self._clone_cache(outputs.past_key_values)

        logging.info("✅ TTS warmup state prepared")

    def _update_audio_buffer(self, audio_buffer, buffer_fill_level, new_audio, buffer_size_samples):
        """
        Append incoming samples to the sliding-window buffer and produce the view used for inference.

        Parameters:
            audio_buffer (torch.Tensor): Tensor of shape `[1, buffer_size_samples]` holding the latest audio samples.
            buffer_fill_level (int): Number of valid samples currently stored in `audio_buffer`.
            new_audio (torch.Tensor): Incoming samples of shape `[1, slice_n_samples]` for the current step.
            buffer_size_samples (int): Total capacity of the buffer in samples.

        Returns:
            Tuple[torch.Tensor, int, torch.Tensor]:
                - Updated `audio_buffer` containing the newest samples (always capped to `buffer_size_samples`).
                - Updated `buffer_fill_level`, reflecting how many contiguous samples are valid.
                - `current_buffer`, a view over the valid portion of the buffer used for the model input.

        Notes:
            `audio_buffer` always retains the last `buffer_size_samples` samples even when overfilled,
            whereas `current_buffer` may be shorter during the initial warm-up phase when the buffer
            is not yet full.
        """
        if new_audio.shape[1] == 0:
            current_buffer = audio_buffer[:, :buffer_fill_level]
            return audio_buffer, buffer_fill_level, current_buffer

        remaining = new_audio

        if buffer_fill_level < buffer_size_samples and remaining.shape[1] > 0:
            warmup_take = min(buffer_size_samples - buffer_fill_level, remaining.shape[1])
            if warmup_take > 0:
                audio_buffer[:, buffer_fill_level:buffer_fill_level + warmup_take] = remaining[:, :warmup_take]
                buffer_fill_level += warmup_take
                remaining = remaining[:, warmup_take:]

        if remaining.shape[1] > 0:
            if remaining.shape[1] >= buffer_size_samples:
                audio_buffer = remaining[:, -buffer_size_samples:]
            else:
                audio_buffer = torch.cat([
                    audio_buffer[:, remaining.shape[1]:],
                    remaining
                ], dim=1)
            buffer_fill_level = buffer_size_samples
        current_buffer = audio_buffer if buffer_fill_level == buffer_size_samples else audio_buffer[:, :buffer_fill_level]
        return audio_buffer, buffer_fill_level, current_buffer


    def infer_one_step(self,
                       audio_input,
                       num_frames_per_inference,
                       frame_idx,
                       gen_text,
                       audio_toks_buffer,
                       input_embeds_history,
                       dynamic_cache,
                       embedding_position=-1,
                       past_key_values=None,
                       code=None,
                       subword_mask=None,
                       gen_asr_text=None):

        use_cache = dynamic_cache is not None
        batch_size = gen_text.shape[0]

        predicted_tokens = torch.empty((batch_size, num_frames_per_inference), dtype=gen_text.dtype, device=gen_text.device)
        asr_predicted_tokens = torch.empty((batch_size, num_frames_per_inference), dtype=gen_text.dtype, device=gen_text.device)

        # do "perception" step outside the for-loop 
        buffer_len = torch.tensor([audio_input.shape[1]], dtype=torch.long, device=self.device)

        source_encoded, _, _ = self.model.stt_model.perception(
            input_signal=audio_input,
            input_signal_length=buffer_len,
            return_encoder_emb=True,
        )

        source_encoded = source_encoded.to(self.dtype)
        total_encoded_frames = source_encoded.shape[1]

        if embedding_position < 0:
            newest_frame_index = total_encoded_frames + embedding_position
        else:
            newest_frame_index = embedding_position

        base_frame_index = newest_frame_index - (num_frames_per_inference - 1)
        base_frame_index = max(base_frame_index, 0)

        new_input_embeds = []
        for chunk_offset in range(num_frames_per_inference):
            current_frame_idx = frame_idx + chunk_offset
            current_frame_index = base_frame_index + chunk_offset
            current_frame_index = min(current_frame_index, total_encoded_frames - 1)
            current_frame_embedding = source_encoded[:, current_frame_index:current_frame_index + 1, :]
            
            current_input_emb = current_frame_embedding.clone()
            current_input_emb *= self.model.stt_model.cfg.get("duplex_nano_channel_weight", 1.0)

            if current_frame_idx == 0:
                current_input_emb += self._get_bos_embedding()
                current_input_emb += self._get_asr_bos_embedding()
            else:
                last_token_emb = self.model.stt_model.embed_tokens(gen_text[:, current_frame_idx - 1])
                current_input_emb += last_token_emb
                last_asr_token_emb = self.model.stt_model.embed_asr_tokens(gen_asr_text[:, current_frame_idx - 1])
                current_input_emb += last_asr_token_emb

            # import pdb; pdb.set_trace()

            if use_cache:
                if current_frame_idx == 0:
                    ans = self.model.stt_model(current_input_emb, cache=dynamic_cache)
                else:
                    ans = self.model.stt_model(current_input_emb, cache=dynamic_cache)
                dynamic_cache = ans["cache"]
            else:
                new_input_embeds.append(current_input_emb)
                full_input_embeds = torch.cat(input_embeds_history + new_input_embeds, dim=1)
                ans = self.model.stt_model(full_input_embeds, cache=None)

            predicted_token = ans["text_logits"][:, -1].argmax(dim=-1)
            asr_predicted_token = ans["asr_logits"][:, -1].argmax(dim=-1)

            # logging.info(f"current_frame_idx: {current_frame_idx}, text_token: {gen_text[:, current_frame_idx].item()}, asr_token: {gen_asr_text[:, current_frame_idx].item()}")
            # if current_frame_idx % 50 == 0:
                # import pdb; pdb.set_trace()

            gen_text[:, current_frame_idx] = predicted_token
            predicted_tokens[:, chunk_offset] = predicted_token
            
            gen_asr_text[:, current_frame_idx] = asr_predicted_token
            asr_predicted_tokens[:, chunk_offset] = asr_predicted_token
            
            # Apply forced turn taking based on ASR results
            self._maybe_apply_forced_turn_taking(current_frame_idx, gen_text, gen_asr_text)
            # Update predicted_tokens with any changes made by forced turn taking
            predicted_tokens[:, chunk_offset] = gen_text[:, current_frame_idx]
            
            if self.decode_audio:
                current_subword_id = gen_text[:, current_frame_idx].unsqueeze(-1)
                
                # do one step inference on Duplex TTS model
                if current_frame_idx == 0:
                    if self.first_context_subword_id is None:
                        raise RuntimeError("first_context_subword_id is not initialized. Ensure TTS warmup ran successfully.")
                    prev_subword_id = self.first_context_subword_id
                else:
                    prev_subword_id = gen_text[:, current_frame_idx-1].unsqueeze(-1)

                # create subword_mask
                current_subword_mask = subword_mask[:, current_frame_idx].unsqueeze(-1)

                if self.generation_config is None:
                    raise RuntimeError("generation_config is not initialized. Ensure TTS warmup ran successfully.")

                code, past_key_values = self.model.tts_model.infer_codes_one_step(
                    current_subword_id=current_subword_id,
                    prev_subword_id=prev_subword_id,
                    current_subword_mask=current_subword_mask,
                    prev_audio_tokens=code,
                    past_key_values=past_key_values,
                    guidance_enabled=True,
                    generation_config=self.generation_config,
                    ignore_eos_flag_stop=True,
                )

                # update audio_toks_buffer with new code
                # * we do this still inside the for-loop, and update one frame at a time
                # * note: audio_toks_buffer will be fed to the audio decoder 
                audio_toks_buffer = torch.cat([audio_toks_buffer[:, 1:], code], dim=1)

                # now that we've saved audio_toks_buffer for audio decoding purposes,
                # we can potentially overwrite the audio token with silence tokens (for feeding to the audio token predictor)
                if self.model.cfg.get('inference_force_speech_silence_on_eos', None):
                    silence_codes = self.model.tts_model.codec_silence_tokens.view(1, 1, -1).expand(code.shape)
                    code = torch.where(
                        current_subword_id.unsqueeze(-1) == self.model.tts_model.text_eos_id,
                        silence_codes,
                        code,
                    )

        # exit for-loop & do audio decoding non-autoregressively (if decode_audio is True)
        if self.decode_audio:
            samples_per_audio_output_frame = self._samples_per_audio_output_frame()
            start_time_decode = time.time()
            # logging.info(f"\n🔊 Decoding audio for {frame_idx}-th frame  ({num_frames_per_inference=})")

            len_audio_toks_buffer = torch.tensor([self.codec_token_history_size], dtype=torch.long, device=self.device)

            with fp32_precision(), torch.no_grad():
                decoded_audio, decoded_audio_len = self.model.tts_model.audio_codec.decode(
                    audio_toks_buffer,
                    len_audio_toks_buffer
                )
            
            # logging.info(f"   Decoded Audio full shape: {decoded_audio.shape}")
            # logging.info(f"   Decoded Audio length: {decoded_audio_len}")
            # logging.info(f"   samples_per_audio_output_frame={samples_per_audio_output_frame}")
            # logging.info(f"   num_frames_per_inference={num_frames_per_inference}")
            # logging.info(f"   Expected new samples to extract: {samples_per_audio_output_frame * num_frames_per_inference}")

            decoded_audio_new = decoded_audio[:, :, -samples_per_audio_output_frame * num_frames_per_inference:]
            # logging.info(f"   Extracted decoded_audio_new shape: {decoded_audio_new.shape}")
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            # logging.info(f"🕒   Time taken to decode audio: {time.time() - start_time_decode:.3f}s")

        else:
            audio_toks_buffer = None
            decoded_audio_new = None

        # convert new text tokens to string that we can show to the user
        predicted_text_strs = []
        # loop over batch dimension
        for predicted_tok_ids_b in predicted_tokens:
            predicted_tok_ids_b = predicted_tok_ids_b.tolist()
            predicted_toks_b = self.tokenizer.ids_to_tokens(predicted_tok_ids_b)

            # TODO: make more robust to tokenizer changes
            # replace "Ġ" with " " to restore proper word boundaries
            # replace '<SPECIAL_12>' with ""
            predicted_toks_b = [tok.replace('<SPECIAL_12>', "").replace('Ġ', ' ') for tok in predicted_toks_b]

            predicted_text_strs.append("".join(predicted_toks_b))

        # convert new ASR tokens to string
        asr_predicted_text_strs = []
        for asr_predicted_tok_ids_b in asr_predicted_tokens:
            asr_predicted_tok_ids_b = asr_predicted_tok_ids_b.tolist()
            asr_predicted_toks_b = self.tokenizer.ids_to_tokens(asr_predicted_tok_ids_b)

            # TODO: make more robust to tokenizer changes
            # replace "Ġ" with " " to restore proper word boundaries
            # replace '<SPECIAL_12>' with ""
            asr_predicted_toks_b = [tok.replace('<SPECIAL_12>', "").replace('Ġ', ' ') for tok in asr_predicted_toks_b]

            asr_predicted_text_strs.append("".join(asr_predicted_toks_b))

        return {
            'predicted_text_tokens': predicted_tokens,
            'asr_predicted_text_tokens': asr_predicted_tokens,
            'audio_toks_buffer': audio_toks_buffer,
            'decoded_audio_new': decoded_audio_new,
            'predicted_text_strs': predicted_text_strs,
            'asr_predicted_text_strs': asr_predicted_text_strs,
            'input_embeds_history': input_embeds_history + new_input_embeds if not use_cache else input_embeds_history,
            'dynamic_cache': dynamic_cache if use_cache else None,
            'past_key_values': past_key_values,
            'code': code,
        }

    def _maybe_apply_forced_turn_taking(self, t, gen_text, gen_asr):
        """Apply forced turn-taking rules based on ASR channel tokens."""
        if not self.model_cfg.get("force_turn_taking", True):
            return
        
        threshold = self.model_cfg.get("force_turn_taking_threshold", 40)
        pad_window_steps = self.model_cfg.get("force_turn_taking_pad_window", 25)
        
        B = gen_text.size(0)
        
        for batch_idx in range(B):
            lookback_start = max(0, t - threshold)
            agent_text_window = gen_text[batch_idx, lookback_start:t]
            current_asr_token = gen_asr[batch_idx, t]
            
            # ASR EOS or ~1 sec of pad tokens → insert agent BOS if not present in window
            # Skip if we don't have enough tokens at the beginning
            if t < pad_window_steps:
                continue
            
            pad_lookback_start = t - pad_window_steps
            asr_recent_tokens = gen_asr[batch_idx, pad_lookback_start:t]
            has_pad_window = (asr_recent_tokens == self.model.stt_model.text_pad_id).all() if len(asr_recent_tokens) > 0 else False
            
            # Require that the pad window starts after a non-pad token
            if has_pad_window and pad_lookback_start > 0:
                token_before_window = gen_asr[batch_idx, pad_lookback_start - 1]
                has_pad_window = (token_before_window != self.model.stt_model.text_pad_id)
            elif has_pad_window and pad_lookback_start == 0:
                # If the pad window starts at position 0, it doesn't meet the requirement
                has_pad_window = False
            
            if has_pad_window:
                if not (agent_text_window == self.model.stt_model.text_bos_id).any():
                    gen_text[batch_idx, t] = self.model.stt_model.text_bos_id
                    reason = "ASR EOS" if current_asr_token == self.tokenizer.eos else "pad window"
                    logging.info(f"🎤→🤖 Forced turn-taking at frame {t}: inserted agent BOS (reason: {reason})")
            
            # ASR BOS → insert agent EOS if not present in window
            elif current_asr_token == self.model.stt_model.user_bos_id:
                if not (agent_text_window == self.model.stt_model.text_eos_id).any():
                    gen_text[batch_idx, t] = self.model.stt_model.text_eos_id
                    logging.info(f"🤖→🎤 Forced turn-taking at frame {t}: inserted agent EOS (reason: user started speaking)")
    
    def _explore_padding_behavior(self, buffer_size_samples):
        """
        Explore whether the encoder uses left or right padding.
        This helps determine which frame's embedding to use.
        """
        logging.info("\n" + "=" * 70)
        logging.info(" EXPLORING ENCODER PADDING BEHAVIOR")
        logging.info("=" * 70)
        
        # Create test signals with distinctive patterns
        # Test 1: Full buffer
        full_signal = torch.randn(1, buffer_size_samples, dtype=self.dtype, device=self.device)
        full_len = torch.tensor([buffer_size_samples], dtype=torch.long, device=self.device)
        
        # Test 2: Half buffer (should be padded)
        half_size = buffer_size_samples // 2
        half_signal = torch.randn(1, half_size, dtype=self.dtype, device=self.device)
        # Pad to full buffer size
        half_signal_padded = torch.nn.functional.pad(half_signal, (0, buffer_size_samples - half_size), value=0.0)
        half_len = torch.tensor([half_size], dtype=torch.long, device=self.device)
        
        # Encode both
        with torch.no_grad():
            full_enc, _, _ = self.model.perception(full_signal, full_len, return_encoder_emb=True)
            half_enc, _, _ = self.model.perception(half_signal_padded, half_len, return_encoder_emb=True)
        
        logging.info(f"Full buffer encoding: {full_enc.shape}")
        logging.info(f"Half buffer encoding: {half_enc.shape}")
        
        # Analyze which positions have meaningful embeddings
        full_norm = torch.norm(full_enc, dim=-1)  # [B, T]
        half_norm = torch.norm(half_enc, dim=-1)  # [B, T]
        
        logging.info(f"\nNorm analysis (full buffer): min={full_norm.min():.4f}, max={full_norm.max():.4f}, mean={full_norm.mean():.4f}")
        logging.info(f"Norm analysis (half buffer): min={half_norm.min():.4f}, max={half_norm.max():.4f}, mean={half_norm.mean():.4f}")
        
        # Check first and last few frames
        logging.info(f"\nFirst 3 frame norms (half buffer): {half_norm[0, :3].tolist()}")
        logging.info(f"Last 3 frame norms (half buffer): {half_norm[0, -3:].tolist()}")
        
        # Heuristic: if last frames have significantly higher norm, likely right-padded (valid data at start)
        # if first frames have higher norm, likely left-padded (valid data at end)
        first_avg = half_norm[0, :half_enc.shape[1]//2].mean()
        last_avg = half_norm[0, half_enc.shape[1]//2:].mean()
        
        logging.info(f"\nFirst half average norm: {first_avg:.4f}")
        logging.info(f"Last half average norm: {last_avg:.4f}")
        
        if first_avg > last_avg * 1.2:  # First half significantly larger
            padding_type = "RIGHT"
            useful_position = -1  # Last position corresponds to newest data
            explanation = "Valid data at start, padding at end. New frame info at LAST position."
        elif last_avg > first_avg * 1.2:
            padding_type = "LEFT"
            useful_position = -1  # Still last position, but for different reason
            explanation = "Padding at start, valid data at end. New frame info at LAST position."
        else:
            padding_type = "UNKNOWN (similar norms)"
            useful_position = -1  # Default to last
            explanation = "Cannot determine clearly. Defaulting to LAST position."
        
        logging.info(f"\n{'='*70}")
        logging.info(f"CONCLUSION: Encoder appears to use {padding_type} padding")
        logging.info(f"Explanation: {explanation}")
        logging.info(f"Recommendation: Use embedding at position [{useful_position}] for newest frame")
        logging.info(f"{'='*70}\n")
        
        return useful_position
    
    @torch.no_grad()
    def inference_realtime_streaming(self, audio_path: str, num_frames_per_inference: int = None, explore_padding: bool = True, request_id: Optional[str] = None):
        """
        Perform realtime streaming inference simulating microphone capture.
        
        Args:
            audio_path: Path to input audio file (simulates microphone input)
            num_frames_per_inference: Number of frames to process per inference step (default: 1)
            explore_padding: Whether to explore encoder padding behavior first
            
        Returns:
            Dictionary with 'text', 'tokens_text', 'tokens_audio', 'audio', 'audio_len'
        """
        # Use provided value or default
        if num_frames_per_inference is None:
            num_frames_per_inference = DEFAULT_NUM_FRAMES_PER_INFERENCE
        if num_frames_per_inference < 1:
            raise ValueError("num_frames_per_inference must be at least 1")
        start_time = time.time()
        
        logging.info("\n" + "=" * 70)
        logging.info("STARTING REALTIME STREAMING INFERENCE")
        logging.info("=" * 70)
        
        buffer_size_frames = int(self.model_cfg.get("buffer_size_frames", DEFAULT_BUFFER_SIZE_FRAMES))
        buffer_size_samples = buffer_size_frames * FRAME_SIZE_SAMPLES
        if num_frames_per_inference > buffer_size_frames:
            raise ValueError(
                f"num_frames_per_inference ({num_frames_per_inference}) must be "
                f"less than or equal to buffer_size_frames ({buffer_size_frames})."
            )
        logging.info(f"Buffer size: {buffer_size_frames} frames ({buffer_size_frames * FRAME_SIZE_SEC}s)")
        logging.info(f"Frames per inference step: {num_frames_per_inference}")
        
        # Explore padding behavior if requested
        if explore_padding:
            embedding_position = self._explore_padding_behavior(buffer_size_samples)
        else:
            embedding_position = -1  # Default to last position
            logging.info(f"Using default embedding position: {embedding_position} (last frame)")

        # Load audio file (simulating microphone stream)
        logging.info(f"\n📁 Loading audio file: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        total_samples = len(audio_signal)
        total_duration = total_samples / SAMPLE_RATE
        
        logging.info(f"   Total duration: {total_duration:.2f}s")
        logging.info(f"   Total samples: {total_samples}")

        # derive num_inference_steps
        total_frames_maybe = int(np.ceil(total_samples / FRAME_SIZE_SAMPLES)) # "maybe" because we might need to add padding
        num_inference_steps = (total_frames_maybe // num_frames_per_inference)
        if total_frames_maybe % num_frames_per_inference != 0:
            num_inference_steps += 1
        total_frames = num_inference_steps * num_frames_per_inference

        # pad audio signal so that it is divisible by num_inference_steps
        padded_total_samples = num_inference_steps * num_frames_per_inference * FRAME_SIZE_SAMPLES
        if padded_total_samples > total_samples:
            audio_signal = np.pad(audio_signal, (0, padded_total_samples - total_samples), mode='constant')
            logging.info(f"   Padded to: {padded_total_samples} samples")
        logging.info(f" {num_frames_per_inference=} => {total_frames=}, {num_inference_steps=}")

        # convert audio signal to tensor
        audio_signal_tensor = torch.tensor(audio_signal, dtype=self.dtype, device=self.device).unsqueeze(0)

        # Check if Nemotron (no cache support)
        use_cache = 'Nemotron' not in self.model.stt_model.cfg.pretrained_llm
        logging.info(f"\n⚙️  Model: {self.model.stt_model.cfg.pretrained_llm}")
        logging.info(f"   Use cache: {use_cache}")

        # Initialize buffer and state
        audio_buffer = torch.zeros(1, buffer_size_samples, dtype=self.dtype, device=self.device)
        buffer_fill_level = 0  # How many samples currently in buffer

        # Initialize LLM cache
        if use_cache:
            llm_cache = DynamicCache()
        else:
            llm_cache = None
            input_embeds_history = []  # For no-cache mode

        # Initialize TTS
        code = None
        past_key_values = None
        subword_mask = None
        if self.decode_audio and hasattr(self.model, 'tts_model'):

            # init audio toks buffer with codec_token_history_size number of silence tokens
            # shape of audio_toks_buffer is: [1, codec_token_history_size, num_codes] (batch size = 1)
            audio_toks_buffer = self.model.tts_model.codec_silence_tokens.view(1, 1, -1).expand(
                -1, self.codec_token_history_size, -1
            ).to(self.device)

            if (
                self.first_context_subword_id is None
                or self.generation_config is None
                or self.first_tts_code_input is None
                or self.first_tts_past_key_values_input is None
            ):
                raise RuntimeError("TTS warmup state was not prepared during initialization.")

            past_key_values = self._clone_cache(self.first_tts_past_key_values_input)
            code = self.first_tts_code_input.detach().clone()
            subword_mask = torch.ones(1, total_frames, device=self.device, dtype=torch.bool)

            logging.info(f"✅ TTS initialized")

        gen_text = torch.full((1, total_frames), self.model.stt_model.text_pad_id, device=self.device, dtype=torch.long)
        gen_asr_text = torch.full((1, total_frames), self.model.stt_model.text_pad_id, device=self.device, dtype=torch.long)

        # initialize list to which we will append generated audio segments
        audio_segments = []

        logging.info("\n" + "=" * 70)
        logging.info("🎤 STARTING FRAME-BY-FRAME PROCESSING")
        logging.info("=" * 70)

        # frame_idx corresponds to index of the first frame passed to infer_one_step
        # (we need this distinction in the case that num_frames_per_inference > 1)
        frame_idx = 0
        while frame_idx < total_frames:
            slice_start = frame_idx * FRAME_SIZE_SAMPLES
            slice_n_samples = num_frames_per_inference * FRAME_SIZE_SAMPLES
            slice_end = slice_start + slice_n_samples
            new_audio = audio_signal_tensor[:, slice_start:slice_end]
            
            audio_buffer, buffer_fill_level, current_buffer = self._update_audio_buffer(
                audio_buffer, buffer_fill_level, new_audio, buffer_size_samples
            )
            
            result = self.infer_one_step(
                audio_input=current_buffer,
                num_frames_per_inference=num_frames_per_inference,
                frame_idx=frame_idx,
                gen_text=gen_text,
                audio_toks_buffer=audio_toks_buffer if self.decode_audio else None,
                input_embeds_history=input_embeds_history if not use_cache else [],
                dynamic_cache=llm_cache if use_cache else None,
                embedding_position=embedding_position,
                past_key_values=past_key_values if self.decode_audio else None,
                code=code if self.decode_audio else None,
                subword_mask=subword_mask if self.decode_audio else None,
                gen_asr_text=gen_asr_text,
            )

            # handle results from infer_one_step
            input_embeds_history = result['input_embeds_history']
            llm_cache = result['dynamic_cache']
            if self.decode_audio:
                audio_toks_buffer = result['audio_toks_buffer']
                decoded_audio_new = result['decoded_audio_new']
                if decoded_audio_new is not None:
                    audio_segments.append(decoded_audio_new)

                past_key_values = result['past_key_values']
                code = result['code']
            else:
                decoded_audio_new = None

            if frame_idx % 10 == 0 or frame_idx < 3 or gen_text[:, frame_idx].item() == self.model.stt_model.text_eos_id:
                token_str = self.tokenizer.ids_to_text([gen_text[0, frame_idx].item()])
                buffer_status = f"{buffer_fill_level}/{buffer_size_samples}" if buffer_fill_level < buffer_size_samples else "FULL"
                special_label = ""
                if gen_text[0, frame_idx].item() == self.model.stt_model.text_bos_id:
                    special_label = " [BOS]"
                elif gen_text[0, frame_idx].item() == self.model.stt_model.text_eos_id:
                    special_label = " [EOS]"
                elif gen_text[0, frame_idx].item() == self.model.stt_model.text_pad_id:
                    special_label = " [PAD]"
                logging.info(f"Frame {frame_idx:3d}/{total_frames} | Buffer: {buffer_status:20s} | Token: {gen_text[0, frame_idx].item():5d}{special_label} | '{token_str}'")
            
            # if gen_text[:, frame_idx].item() == self.model.stt_model.text_eos_id:
                # gen_text[:, frame_idx+1:] = self.model.stt_model.text_pad_id
                # total_frames = frame_idx + 1
                # break

            frame_idx += num_frames_per_inference

        import pdb; pdb.set_trace()
        # Find the index of the last non-pad token in the first turn and print timestamp
        # Here, turn boundary is assumed to be the first EOS or PAD after a burst of non-pad tokens
        # We'll use BOS (1), EOS (2), and PAD (12)
        pad_id = 12
        bos_id = 1
        eos_id = 2
        gen_text_b = gen_text[0].tolist()
        
        # Find the start of first turn (after first BOS)
        try:
            first_bos_idx = gen_text_b.index(bos_id)
        except ValueError:
            first_bos_idx = 0  # fallback if BOS is not found

        # Now collect indices between first BOS and the next EOS (or PAD, whichever comes first after text)
        in_first_turn = []
        in_turn = False
        for i in range(first_bos_idx, len(gen_text_b)):
            tok = gen_text_b[i]
            if tok == bos_id:
                in_turn = True
                continue  # Skip BOS itself
            if in_turn:
                if tok == eos_id or tok == pad_id:
                    # Stop collecting at first EOS or PAD after real tokens
                    break
                in_first_turn.append(i)

        # Now, find the last non-pad index in in_first_turn
        if in_first_turn:
            last_non_pad_idx = in_first_turn[-1]
            timestamp = last_non_pad_idx * 0.08
            print(f"[DEBUG] Last non-pad token index (first turn): {last_non_pad_idx}, timestamp: {timestamp:.2f} sec")
        else:
            print("[DEBUG] No tokens found for first turn, or all are PAD/EOS after BOS.")

        # Prepare results
        elapsed_time = time.time() - start_time
        logging.info("\n" + "=" * 70)
        logging.info("✅ STREAMING INFERENCE COMPLETED")
        logging.info("=" * 70)
        logging.info(f"Total time: {elapsed_time:.2f}s")
        logging.info(f"Audio duration: {total_duration:.2f}s")
        logging.info(f"RTF (Real-Time Factor): {elapsed_time / total_duration:.2f}x")
        logging.info(f"Processed frames: {total_frames}")

        # Trim to actual length
        # TODO: this is currently redundant since we iterate over all frames in the while loop
        gen_text = gen_text[:, :total_frames]
        gen_asr_text = gen_asr_text[:, :total_frames]

        # Decode text
        lengths = torch.tensor([total_frames], dtype=torch.long, device=self.device)
        text_output = tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id, eval_text_turn_taking=True)
        
        # Decode ASR text
        asr_text_output = tokens_to_str(gen_asr_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id, eval_text_turn_taking=True)

        logging.info(f"\n📝 Generated text: {text_output[0]}")
        logging.info(f"\n🎤 Generated ASR text: {asr_text_output[0]}")

        ans = {
            "text": text_output,
            "tokens_text": gen_text,
            "tokens_len": lengths,
            "audio": torch.cat(audio_segments, dim=-1) if audio_segments else None,
            "asr_text": asr_text_output,
            "asr_tokens": gen_asr_text,
            "input_audio": audio_signal_tensor,  # Store input audio for stereo output
            "input_sample_rate": SAMPLE_RATE,  # Store input sample rate
        }

        return ans


def main():
    parser = argparse.ArgumentParser(description="Realtime Streaming Inference (Microphone Simulation)")
    parser.add_argument("--model_path", type=str, 
                       default="/lustre/fsw/portfolios/llmservice/users/cchen1/code/s2s_eartts/Duplex_S2S_Nanov2_30_set_hf/",
                       help="Path to eartts's checkpoint with TTS (HF format)")
    parser.add_argument("--llm_checkpoint_path", type=str,
                       default="/lustre/fsw/portfolios/llmservice/users/cchen1/code/s2s_eartts/nano-9b-model/checkpoints_hf_24002",
                       help="Path to YOUR checkpoint with correct LLM/perception (HF format)")
    parser.add_argument("--audio_path", type=str, required=True,
                       help="Path to input audio file")
    parser.add_argument("--speaker_reference", type=str,
                       default="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-full-duplex/inference_references/Emma_S3_A1_SC7_singleturntarget_21_channel_1_audio_in.wav",
                       help="Path to speaker reference audio file")
    parser.add_argument("--buffer_size_frames", type=int, default=70,
                       help="Size of audio buffer in frames (each frame = 80ms)")
    parser.add_argument("--num_frames_per_inference", type=int, default=DEFAULT_NUM_FRAMES_PER_INFERENCE,
                       help="Number of frames per inference step (default: 1)")
    parser.add_argument("--output_text", type=str, default="output_text_streaming.txt",
                       help="Output text file path")
    parser.add_argument("--output_asr_text", type=str, default="output_asr_text_streaming.txt",
                       help="Output ASR text file path")
    parser.add_argument("--output_audio", type=str, default="generated_audio_streaming.wav",
                       help="Output audio file path")
    parser.add_argument("--output_stereo_audio", type=str, default="generated_audio_streaming_stereo.wav",
                       help="Output stereo audio file path (left=input, right=output)")
    parser.add_argument("--decode_audio", action="store_true",
                       help="Whether to decode audio")
    parser.add_argument("--explore_padding", action="store_true", default=False,
                       help="Explore encoder padding behavior (recommended for first run)")
    parser.add_argument("--combine_inp_out_audio", action="store_true",
                    help="Whether to decode audio")
    
    # vLLM arguments
    parser.add_argument("--engine_type", type=str, default="native", choices=["native", "vllm"],
                       help="Engine type for inference (default: native)")
    parser.add_argument("--vllm_model_path", type=str, default=None,
                       help="Path to vLLM-compatible model checkpoint if the path not exists, it will be auto-converted")
    parser.add_argument("--vllm_max_model_len", type=int, default=10240,
                       help="Maximum sequence length for vLLM (default: 10240)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.4,
                       help="GPU memory utilization for vLLM (default: 0.4)")
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16",
                       help="Data type for vLLM (default: bfloat16)")

    args = parser.parse_args()
    
    try:
        
        model_cfg_dict = {
            "model_path": args.model_path,
            "llm_checkpoint_path": args.llm_checkpoint_path,
            "speaker_reference": args.speaker_reference,
            "buffer_size_frames": args.buffer_size_frames,
            "decode_audio": bool(args.decode_audio),
        }
        model_cfg = OmegaConf.create(model_cfg_dict)

        model = RealtimeStreamingInference(model_cfg=model_cfg)
        
        # Run inference
        results = model.inference_realtime_streaming(
            args.audio_path, 
            num_frames_per_inference=args.num_frames_per_inference,
            explore_padding=args.explore_padding,
        )

        # Save outputs
        logging.info("\n" + "=" * 70)
        logging.info("💾 SAVING OUTPUTS")
        logging.info("=" * 70)
        
        # Save text
        with open(args.output_text, 'w') as f:
            f.write(results['text'][0].replace("<|", "\n<|"))
        logging.info(f"✅ Text output saved: {args.output_text}")

        # Save ASR text
        if 'asr_text' in results:
            with open(args.output_asr_text, 'w') as f:
                f.write(results['asr_text'][0])
            logging.info(f"✅ ASR text output saved: {args.output_asr_text}")
        
        # Save audio if available
        if args.decode_audio and 'audio' in results:
            audio_np = results['audio'].float().cpu().numpy()
            
            import soundfile as sf
            sf.write(args.output_audio, audio_np, model.target_sample_rate)
            logging.info(f"✅ Audio output saved: {args.output_audio}")
            
            # Verify
            import subprocess
            size_output = subprocess.check_output(['du', '-h', args.output_audio]).decode().split()[0]
            logging.info(f"   File size: {size_output}")
            
            # Save stereo audio (input + output)
            if 'input_audio' in results:
                logging.info(f"\n🎵 Creating stereo audio (left=input, right=output)...")
                
                input_audio = results['input_audio'].float().cpu().numpy().flatten()
                output_audio = audio_np.flatten()
                
                input_sr = results['input_sample_rate']
                output_sr = model.target_sample_rate
                
                # Resample input to match output sample rate if needed
                if input_sr != output_sr:
                    logging.info(f"   Resampling input from {input_sr}Hz to {output_sr}Hz...")
                    input_audio_resampled = librosa.resample(input_audio, orig_sr=input_sr, target_sr=output_sr)
                else:
                    input_audio_resampled = input_audio
                
                # Align lengths (pad or trim to match the shorter/longer one)
                input_len = len(input_audio_resampled)
                output_len = len(output_audio)
                target_len = min(input_len, output_len)
                
                logging.info(f"   Input length: {input_len} samples")
                logging.info(f"   Output length: {output_len} samples")
                logging.info(f"   Aligned length: {target_len} samples")
                
                # Trim both to the same length
                input_channel = input_audio_resampled[:target_len]
                output_channel = output_audio[:target_len]
                
                # Create stereo array: shape (samples, 2) where column 0 is left, column 1 is right
                stereo_audio = np.stack([input_channel, output_channel], axis=1)
                
                sf.write(args.output_stereo_audio, stereo_audio, output_sr)
                logging.info(f"✅ Stereo audio output saved: {args.output_stereo_audio}")
                
                size_stereo = subprocess.check_output(['du', '-h', args.output_stereo_audio]).decode().split()[0]
                logging.info(f"   File size: {size_stereo}")
                logging.info(f"   Format: 2 channels, {output_sr}Hz, {target_len} samples ({target_len/output_sr:.2f}s)")
        
        logging.info("=" * 70)
        logging.info("✅ ALL DONE!")
        logging.info("=" * 70)
        
    except Exception as e:
        logging.error(f"❌ ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

