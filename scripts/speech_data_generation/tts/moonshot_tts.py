#!/usr/bin/env python3
"""
TTS script using Moonshot's Kimi-Audio model.
Kimi-Audio is a high-quality TTS model from Moonshot AI.
"""

import os
import torch
import argparse
import glob
import random
import re
from pathlib import Path
from nemo.collections import asr as nemo_asr
import transformers
import torchaudio
import numpy as np
import soundfile as sf

def transcribe_audio_file(asr_model, audio_path):
    """Transcribe an audio file using NeMo ASR model."""
    try:
        asr_outputs = asr_model.transcribe([audio_path])
        if asr_outputs and len(asr_outputs) > 0:
            result = asr_outputs[0]
            text = result.text if hasattr(result, 'text') else ""
            return text.strip()
        return ""
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""

def create_pipeline(model_id, token=""):
    """Create LLM pipeline for text generation."""
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )
    return pipeline

def create_prompt(question, system_prompt=None):
    """Create a prompt for the LLM."""
    DEFAULT_SYSTEM_PROMPT = "Based on the question, directly answer by one short sentence using spoken words. The answer should be at most 1 sentences with at most 20 words."
    sys_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
    prompt = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"{question}"},
    ]
    return prompt

def run_llm_pipeline(llm_pipeline, user_transcription):
    """Generate agent response using LLM based on user transcription."""
    if llm_pipeline is None:
        return ""
    
    try:
        prompt = create_prompt(user_transcription)
        outputs = llm_pipeline(
            prompt,
            max_new_tokens=512,
            return_full_text=False
        )
        
        if outputs and len(outputs) > 0:
            text = outputs[0]['generated_text']
            cleaned_text = text.strip()
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            return cleaned_text
        else:
            return ""
    except Exception as e:
        print(f"Error generating agent response: {e}")
        return ""

def get_wav_files_and_speakers(wav_dir):
    """Get wav files and randomly assign speakers."""
    # Get all subdirectories (which are the cut IDs)
    subdirs = [d for d in os.listdir(wav_dir) if os.path.isdir(os.path.join(wav_dir, d))]

    # Sort numerically for numeric directories, keep non-numeric ones at the end
    def sort_key(d):
        try:
            return (0, int(d))  # Numeric directories first, sorted by integer value
        except ValueError:
            return (1, d)  # Non-numeric directories after, sorted alphabetically
    
    subdirs.sort(key=sort_key)

    items = []
    for subdir_name in subdirs:
        wav_path = os.path.join(wav_dir, subdir_name, "input.wav")
        if os.path.exists(wav_path):
            items.append((wav_path, subdir_name))
        else:
            print(f"Warning: No wav file found at {wav_path}")
    
    return items

def setup_moonshot_tts():
    """Setup Moonshot Kimi-Audio TTS model."""
    try:
        from kimia_infer.api.kimia import KimiAudio
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Moonshot's open-source audio model
        model = KimiAudio(
            model_path="moonshotai/Kimi-Audio-7B-Instruct",
            load_detokenizer=True
        )
        
        print("Moonshot Kimi-Audio model loaded successfully.")
        return model, device
    except ImportError:
        print("Kimi-Audio not found. Please install it first:")
        print("pip install kimia-infer")
        return None, None
    except Exception as e:
        print(f"Error loading Moonshot TTS: {e}")
        return None, None

def generate_tts_audio(moonshot_model, device, text, output_path, seed_audio_path, audio_temperature=0.8, audio_top_k=10):
    """Generate TTS audio using Moonshot Kimi-Audio with seed audio."""
    print('seed_audio_path:', seed_audio_path)
    try:
        with torch.no_grad():
            # Prepare messages for Kimi-Audio - audio first, then text
            messages = [
                {"role":"user","message_type":"audio", "content": seed_audio_path},
                {"role":"user","message_type":"text",  "content": "Read the next message aloud exactly. Do not add or omit any words."},
                {"role":"user","message_type":"text",  "content": text},
            ]
            # messages = [
            #     {"role": "user", "message_type": "audio", "content": seed_audio_path},
            #     {"role": "user", "message_type": "text", "content": f"Please read aloud: {text}"},
            # ]
            
            # Set generation parameters
            params = dict(
                audio_temperature=audio_temperature, 
                audio_top_k=audio_top_k,
                text_temperature=0.0, 
                text_top_k=5,
                audio_repetition_penalty=1.0, 
                audio_repetition_window_size=64,
                text_repetition_penalty=1.0,  
                text_repetition_window_size=16,
            )
            
            # Generate audio and text
            wav, generated_text = moonshot_model.generate(messages, output_type="both", **params)
            
            # Convert to numpy and get sample rate
            wave = wav.detach().cpu().view(-1).numpy()
            sr = 24000  # Kimi-Audio outputs at 24kHz
            
            # Resample to 16kHz if needed
            if sr != 16000:
                audio_tensor = torch.from_numpy(wave).unsqueeze(0)  # Add batch dimension
                resampled = torchaudio.functional.resample(audio_tensor, sr, 16000)
                wave = resampled.squeeze().numpy()
                sr = 16000
            
            # Save audio
            sf.write(output_path, wave, sr)
            
            print(f"Generated text: {generated_text}")
            
        return True
    except Exception as e:
        print(f"Error generating TTS audio: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio using Moonshot Kimi-Audio with transcribed text from wav files")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory containing input wav files to transcribe")
    parser.add_argument("--out_dir", type=str, default="/lustre/fsw/portfolios/convai/users/kevinhu/debug", help="Output directory for generated audio")
    parser.add_argument("--asr_model", type=str, default="nvidia/parakeet-tdt-0.6b-v2", help="NeMo ASR model for transcription")
    parser.add_argument("--transcribe", action="store_true", help="Enable transcription of input wav files")
    parser.add_argument("--fallback_text", type=str, default="Hello, this is a test.", help="Fallback text if transcription fails")
    parser.add_argument("--paraphrase_user_text", action="store_true", help="Use LLM to paraphrase user transcription text.")
    parser.add_argument('--llm_model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="LLM model to use for generating agent responses.")
    parser.add_argument("--audio_temperature", type=float, default=0.8, help="Audio generation temperature for Moonshot TTS")
    parser.add_argument("--audio_top_k", type=int, default=10, help="Audio generation top-k for Moonshot TTS")
    args = parser.parse_args()
    
    # Get wav files
    items = get_wav_files_and_speakers(args.wav_dir)
    
    if not items:
        print("No valid wav files found.")
        return
    
    print(f"Found {len(items)} wav files to process")
    
    # Initialize ASR model if transcription is enabled
    asr_model = None
    if args.transcribe:
        print(f"Loading ASR model: {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.asr_model).cuda()
        print("ASR model loaded successfully.")
    
    # Initialize LLM pipeline if paraphrasing is enabled
    llm_pipeline = None
    if args.paraphrase_user_text:
        print(f"Loading LLM model: {args.llm_model}")
        llm_pipeline = create_pipeline(args.llm_model)
        print("LLM model loaded successfully.")
    
    # Setup Moonshot TTS
    moonshot_model, device = setup_moonshot_tts()
    if moonshot_model is None:
        return
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Process each item
    for i, (wav_file, subdir_name) in enumerate(items):
        print(f"\nProcessing {i+1}/{len(items)}: {subdir_name}")
        
        # Transcribe if enabled
        if args.transcribe and asr_model:
            text = transcribe_audio_file(asr_model, wav_file)
            if not text:
                text = args.fallback_text
                print(f"Using fallback text for {subdir_name}")
            print(f"Transcribed: {text}")
        else:
            text = args.fallback_text
            print(f"Using fallback text: {text}")
        
        # Paraphrase if enabled
        if args.paraphrase_user_text and llm_pipeline and text:
            llm_prompt = create_prompt(text, system_prompt="Paraphrase the following sentence, keeping the meaning but changing the wording. Keep it as a single sentence and natural for spoken English.")
            try:
                paraphrased_text = run_llm_pipeline(llm_pipeline, llm_prompt)
                if paraphrased_text:
                    print(f"Original: {text}")
                    print(f"Paraphrased: {paraphrased_text}")
                    text = paraphrased_text
            except Exception as e:
                print(f"Error paraphrasing text: {e}")
        
        # Create output subdirectory
        subdir_path = os.path.join(args.out_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)
        
        # Generate TTS audio
        output_path = os.path.join(subdir_path, "input.wav")
        print(f"Generating TTS audio...")
        
        success = generate_tts_audio(
            moonshot_model, 
            device, 
            text, 
            output_path,
            wav_file,  # Use the current wav_file as seed audio
            audio_temperature=args.audio_temperature,
            audio_top_k=args.audio_top_k
        )
        
        if success:
            print(f"Saved -> {output_path}")
            print(f"  Text: {text}")
        else:
            print(f"Failed to generate audio for {subdir_name}")
    
    print(f"\nGenerated {len(items)} audio files in {args.out_dir}")

if __name__ == "__main__":
    main()
