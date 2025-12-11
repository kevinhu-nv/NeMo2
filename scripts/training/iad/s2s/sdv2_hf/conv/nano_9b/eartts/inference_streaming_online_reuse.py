"""
Streaming STT inference script that REUSES the online_inference implementation
from duplex_stt_model.py instead of rewriting the logic.

Key differences from inference_streaming_realtime_niva_json.py:
1. Uses model.online_inference() directly instead of custom infer_one_step()
2. Doesn't manually manage audio buffers - handled by the model
3. Doesn't manually call perception encoder - handled internally
4. STT ONLY - no TTS audio generation (simpler, focused on text prediction)
5. Simplified model loading - just load config and checkpoint directly
6. Window size configured via model.cfg.online_window_size

This demonstrates the proper way to use the existing streaming inference infrastructure.

Reference: Based on s2s_duplex_stt_streaming_infer.py example
"""

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
from jiwer import wer


# Monkeypatch NeMo logger to correctly report caller stack frame
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
CODE_DIR_PRETRAIN = "/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/NeMo"
sys.path.insert(0, CODE_DIR_PRETRAIN)

# Set environment variables
os.environ["HF_HOME"] = "/lustre/fsw/portfolios/llmservice/nanos/cchen1/hfcache"
os.environ["TORCH_HOME"] = "/lustre/fsw/portfolios/llmservice/nanos/cchen1/hfcache"
os.environ["NEMO_CACHE_DIR"] = "/lustre/fsw/portfolios/llmservice/nanos/cchen1/hfcache"

# Import the duplex STT model with streaming inference support
from nemo.collections.speechlm2.models.duplex_stt_model import DuplexSTTModel, tokens_to_str

# --- Configuration ---
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Streaming Parameters ---
SAMPLE_RATE = 16000
FRAME_SIZE_SEC = 0.08  # 80ms per frame
FRAME_SIZE_SAMPLES = int(SAMPLE_RATE * FRAME_SIZE_SEC)  # 1280 samples

# Default hyper-parameters
DEFAULT_BUFFER_SIZE_FRAMES = 70  # Used as window size for streaming inference


class StreamingSTTInference:
    """
    Streaming STT inference that REUSES the online_inference() method
    from DuplexSTTModel instead of rewriting the logic.
    
    This is the CORRECT way to do streaming inference - by leveraging
    the existing, well-tested implementation in the model itself.
    
    Architecture:
    - DuplexSTTModel: Handles STT with online_inference()
    - This class: Manages model loading, I/O, and batch processing
    - STT ONLY: Focus on text prediction only (no TTS audio generation)
    """
    
    def __init__(self, model_cfg: DictConfig):
        """
        Initialize the model for online streaming inference.
        
        Args:
            model_cfg (DictConfig): Configuration describing the model paths and runtime parameters.
        """
        if model_cfg is None:
            raise ValueError("model_cfg must be provided")
        if not isinstance(model_cfg, DictConfig):
            model_cfg = OmegaConf.create(model_cfg)

        self.model_cfg = model_cfg

        self.llm_checkpoint_path = model_cfg.get("llm_checkpoint_path")
        if not self.llm_checkpoint_path:
            raise ValueError("`model_cfg.llm_checkpoint_path` must be provided.")

        compute_dtype = model_cfg.get("compute_dtype", "bfloat16")
        self.dtype = self._resolve_dtype(compute_dtype)

        self.device = self._resolve_device(
            device=model_cfg.get("device"),
            device_id=model_cfg.get("device_id"),
        )

        logging.info("=" * 70)
        logging.info("INITIALIZING STREAMING STT INFERENCE (REUSING MODEL METHOD)")
        logging.info("=" * 70)
        logging.info(f"Frame size: {FRAME_SIZE_SEC}s ({FRAME_SIZE_SAMPLES} samples @ {SAMPLE_RATE}Hz)")
        logging.info(f"Device: {self.device}")
        logging.info(f"Compute dtype: {self.dtype}")
        logging.info(f"STT ONLY (no TTS audio generation)")
        logging.info("=" * 70)

        self.stt_model = None
        self.tokenizer = None
        
        self._initialize_model()

        logging.info(f"\n✅ StreamingSTTInference initialized successfully.")
    
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

    def _initialize_model(self):
        """Initialize the DuplexSTTModel (simplified approach based on reference)."""
        from safetensors.torch import load_file
        
        logging.info("\n🚀 Initializing DuplexSTTModel...")
        logging.info(f"  Loading from: {self.llm_checkpoint_path}")
        
        # Load config
        config_file = os.path.join(self.llm_checkpoint_path, "config.json")
        with open(config_file, 'r') as f:
            import json
            cfg_dict = json.load(f)
        
        cfg = DictConfig(cfg_dict)
        cfg.model.pretrained_weights = False
        
        # Set online window size from model config
        buffer_size_frames = int(self.model_cfg.get("buffer_size_frames", DEFAULT_BUFFER_SIZE_FRAMES))
        cfg.model.online_window_size = buffer_size_frames
        
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Initialize model structure
        logging.info("\n🏗️  Initializing model structure...")
        start_init = time.time()
        self.stt_model = DuplexSTTModel(cfg_dict)
        logging.info(f"🕒 Time taken to initialize: {time.time() - start_init:.2f}s")
        logging.info("  ✓ Model structure initialized")
        
        # Load checkpoint weights
        logging.info(f"\n📦 Loading checkpoint weights...")
        state_dict = load_file(os.path.join(self.llm_checkpoint_path, "model.safetensors"))
        missing, unexpected = self.stt_model.load_state_dict(state_dict, strict=False)
        
        if missing:
            logging.info(f"  ⚠️  {len(missing)} keys missing")
        if unexpected:
            logging.info(f"  ⚠️  {len(unexpected)} unexpected keys")
        
        logging.info(f"  ✓ Checkpoint loaded")
        
        # Setup model
        self.stt_model.to(self.device)
        self.stt_model.eval()
        
        # Convert components to configured dtype
        logging.info(f"Converting model components to {self.dtype}...")
        self.stt_model.llm = self.stt_model.llm.to(self.dtype)
        self.stt_model.lm_head = self.stt_model.lm_head.to(self.dtype)
        self.stt_model.embed_tokens = self.stt_model.embed_tokens.to(self.dtype)
        self.stt_model.perception = self.stt_model.perception.to(self.dtype)
        self.stt_model.asr_head = self.stt_model.asr_head.to(self.dtype)
        self.stt_model.embed_asr_tokens = self.stt_model.embed_asr_tokens.to(self.dtype)
        logging.info("✓ Model components converted")
        
        self.stt_model.on_train_epoch_start()
        self.tokenizer = self.stt_model.tokenizer
        
        logging.info(f"\nModel info:")
        logging.info(f"  - LLM: {self.stt_model.cfg.pretrained_llm}")
        logging.info(f"  - Predict user text (ASR): {self.stt_model.predict_user_text}")

    @torch.no_grad()
    def inference_streaming(self, audio_path: str, request_id: Optional[str] = None):
        """
        Perform streaming STT inference using the model's built-in online_inference() method.
        
        This is the KEY difference from the original script:
        - We call stt_model.online_inference() instead of manually implementing the loop
        - The model handles all the complexity: windowing, encoding, caching, etc.
        - Window size is configured via model.cfg.online_window_size
        - STT ONLY - no TTS audio generation
        
        Args:
            audio_path: Path to input audio file (simulates microphone input)
            request_id: Optional request ID for tracking
            
        Returns:
            Dictionary with 'text', 'src_text', 'tokens_text', 'tokens_text_src', 'tokens_len'
        """
        start_time = time.time()
        
        logging.info("\n" + "=" * 70)
        logging.info("STARTING STREAMING STT INFERENCE (USING MODEL METHOD)")
        logging.info("=" * 70)
        
        buffer_size_frames = int(self.model_cfg.get("buffer_size_frames", DEFAULT_BUFFER_SIZE_FRAMES))
        logging.info(f"Buffer size: {buffer_size_frames} frames ({buffer_size_frames * FRAME_SIZE_SEC}s)")

        # Load audio file
        logging.info(f"\n📁 Loading audio file: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        total_samples = len(audio_signal)
        total_duration = total_samples / SAMPLE_RATE
        
        logging.info(f"   Total duration: {total_duration:.2f}s")
        logging.info(f"   Total samples: {total_samples}")

        # Convert to tensor
        audio_signal_tensor = torch.tensor(audio_signal, dtype=self.dtype, device=self.device).unsqueeze(0)
        audio_signal_lens = torch.tensor([total_samples], dtype=torch.long, device=self.device)

        logging.info(f"\n⚙️  Model: {self.stt_model.cfg.pretrained_llm}")
        logging.info(f"   Window size: {buffer_size_frames} frames (configured in model.online_window_size)")

        logging.info("\n" + "=" * 70)
        logging.info("🎤 CALLING MODEL.ONLINE_INFERENCE()")
        logging.info("=" * 70)

        # ===================================================================
        # KEY DIFFERENCE: We call the model's online_inference() method
        # instead of manually implementing the streaming loop
        # ===================================================================
        inference_start = time.time()
        
        results = self.stt_model.online_inference(
            input_signal=audio_signal_tensor,
            input_signal_lens=audio_signal_lens,
            decode_audio=False,
        )
        inference_time = time.time() - inference_start
        
        logging.info(f"\n✅ online_inference() completed in {inference_time:.2f}s")
        logging.info(f"   Generated text (agent): {results['text'][0]}")
        if results.get('src_text') and results['src_text'] is not None:
            logging.info(f"   Generated ASR text (user): {results['src_text'][0]}")

        # Final metrics
        elapsed_time = time.time() - start_time
        logging.info("\n" + "=" * 70)
        logging.info("✅ STREAMING STT INFERENCE COMPLETED")
        logging.info("=" * 70)
        logging.info(f"Total time: {elapsed_time:.2f}s")
        logging.info(f"Audio duration: {total_duration:.2f}s")
        logging.info(f"RTF (Real-Time Factor): {elapsed_time / total_duration:.2f}x")
        logging.info(f"Processed frames: {results['tokens_len'].item()}")

        return results



def main():
    parser = argparse.ArgumentParser(description="Streaming STT Inference (Reusing Model Method)")
    parser.add_argument("--llm_checkpoint_path", type=str, required=True,
                       help="Path to checkpoint with LLM/perception (HF format)")
    parser.add_argument("--input_json", type=str, required=True,
                       help="Path to input JSON file (JSONL format) with records containing 'audio_filepath' field")
    parser.add_argument("--output_json", type=str, default="output_results.json",
                       help="Path to output JSON file with predictions (JSONL format)")
    parser.add_argument("--buffer_size_frames", type=int, default=70,
                       help="Size of audio window in frames (each frame = 80ms)")
    
    args = parser.parse_args()
    
    try:
        # Load input JSON file
        import json
        logging.info(f"📖 Loading input JSON: {args.input_json}")
        with open(args.input_json, 'r') as f:
            input_records = [json.loads(line) for line in f]
        
        logging.info(f"Found {len(input_records)} records to process")
        
        model_cfg_dict = {
            "llm_checkpoint_path": args.llm_checkpoint_path,
            "buffer_size_frames": args.buffer_size_frames,
        }
        model_cfg = OmegaConf.create(model_cfg_dict)

        model = StreamingSTTInference(model_cfg=model_cfg)
        
        # Open output file for incremental writing
        logging.info(f"📝 Output will be saved incrementally to: {args.output_json}")
        output_file = open(args.output_json, 'w', encoding='utf-8')
        
        # Process each record
        output_records = []
        wer_scores = []  # Track WER for each record
        try:
            for idx, record in enumerate(input_records):
                logging.info("\n" + "=" * 70)
                logging.info(f"📝 Processing record {idx + 1}/{len(input_records)}")
                logging.info("=" * 70)
                
                audio_path = record.get('audio_filepath')
                ground_truth_text = record.get('text', '')
                
                if not audio_path:
                    logging.warning(f"⚠️  Record {idx} missing audio_filepath, skipping...")
                    continue
                
                if not os.path.exists(audio_path):
                    logging.warning(f"⚠️  Audio file not found: {audio_path}, skipping...")
                    continue
                
                logging.info(f"   Audio: {audio_path}")
                logging.info(f"   Ground truth: {ground_truth_text}")
                
                # Run inference using the model's online_inference method
                results = model.inference_streaming(audio_path)
                
                # Calculate WER
                pred_src_text = results['src_text'][0] if results.get('src_text') else ''
                pred_src_text = pred_src_text.strip().replace('^', '')
                if ground_truth_text and pred_src_text:
                    wer_score = wer(ground_truth_text, pred_src_text)
                else:
                    wer_score = None  # Cannot calculate WER if either text is missing
                
                if wer_score is not None:
                    wer_scores.append(wer_score)
                    logging.info(f"\033[92m   WER: {wer_score * 100:.2f}%\033[0m")
                
                # Prepare output record
                output_record = {
                    'audio_filepath': audio_path,
                    'text': ground_truth_text,  # Ground truth user text
                    'pred_src_text': pred_src_text,  # Predicted user ASR
                    'pred_text': results['text'][0],  # Predicted agent text
                    'wer': wer_score,  # Word Error Rate
                }
                
                output_records.append(output_record)
                
                # Write record immediately to file
                json.dump(output_record, output_file, ensure_ascii=False)
                output_file.write('\n')
                output_file.flush()  # Ensure it's written to disk immediately
                
                logging.info(f"✅ Record {idx + 1} completed and saved")
        
        finally:
            # Always close output file, even if there's an error
            output_file.close()
        
        # Summary
        logging.info("\n" + "=" * 70)
        logging.info("💾 ALL RESULTS SAVED")
        logging.info("=" * 70)
        logging.info(f"✅ Results saved to: {args.output_json}")
        logging.info(f"   Processed {len(output_records)}/{len(input_records)} records successfully")
        
        # Calculate and print average WER
        if wer_scores:
            avg_wer = sum(wer_scores) / len(wer_scores)
            logging.info("\n" + "=" * 70)
            logging.info("📊 WER STATISTICS")
            logging.info("=" * 70)
            logging.info(f"   Records with WER: {len(wer_scores)}")
            logging.info(f"   Average WER: {avg_wer * 100:.2f}%")
            logging.info(f"   Min WER: {min(wer_scores) * 100:.2f}%")
            logging.info(f"   Max WER: {max(wer_scores) * 100:.2f}%")
        else:
            logging.info("\n⚠️  No WER scores calculated (missing ground truth or predictions)")
        
        logging.info("\n" + "=" * 70)
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

