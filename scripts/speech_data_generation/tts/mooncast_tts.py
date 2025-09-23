#!/usr/bin/env python3
"""
TTS script using MoonCast model.
MoonCast is a high-quality, zero-shot podcast generation system for natural speech synthesis.
"""

import os
import torch
import argparse
import glob
import random
import re
from pathlib import Path
import whisper
import transformers
import torchaudio
import numpy as np
import soundfile as sf

def transcribe_audio_file(asr_model, audio_path):
    """Transcribe an audio file using Whisper model."""
    try:
        result = asr_model.transcribe(audio_path)
        if result and 'text' in result:
            return result['text'].strip()
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

def setup_mooncast_tts():
    """Setup MoonCast TTS model."""
    try:
        from inference import Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load MoonCast model
        model = Model()
        print("MoonCast TTS model loaded successfully.")
        return model, device
    except ImportError:
        print("MoonCast TTS not found. Please install it first:")
        print("git clone https://github.com/jzq2000/MoonCast.git")
        print("cd MoonCast && pip install -r requirements.txt")
        print("python download_pretrain.py")
        return None, None
    except Exception as e:
        print(f"Error loading MoonCast TTS: {e}")
        return None, None

def generate_tts_audio(mooncast_model, device, text, output_path, seed_audio_path, audio_temperature=0.8, audio_top_k=10):
    """Generate TTS audio using MoonCast with seed audio."""
    print('seed_audio_path:', seed_audio_path)
    try:
        import base64
        
        # MoonCast uses dialogue format with base64 encoded output
        req = {"dialogue": [{"role": "0", "text": text}]}
        
        # Generate audio using MoonCast inference
        audio_base64 = mooncast_model.inference(req)
        
        # Decode base64 audio data
        audio_data = base64.b64decode(audio_base64)
        
        # Save as MP3 first, then convert to WAV
        temp_mp3_path = output_path.replace('.wav', '_temp.mp3')
        with open(temp_mp3_path, "wb") as f:
            f.write(audio_data)
        
        # Convert MP3 to WAV using torchaudio
        try:
            waveform, sample_rate = torchaudio.load(temp_mp3_path)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampled = torchaudio.functional.resample(waveform, sample_rate, 16000)
                waveform = resampled
                sample_rate = 16000
            
            # Save as WAV
            sf.write(output_path, waveform.squeeze().numpy(), sample_rate)
            
            # Clean up temp file
            os.remove(temp_mp3_path)
            
        except Exception as e:
            print(f"Error converting MP3 to WAV: {e}")
            # Fallback: just rename the MP3 file
            os.rename(temp_mp3_path, output_path.replace('.wav', '.mp3'))
            return False
        
        print(f"Generated audio for text: {text}")
        
        return True
    except Exception as e:
        print(f"Error generating TTS audio: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio using MoonCast with transcribed text from wav files")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory containing input wav files to transcribe")
    parser.add_argument("--out_dir", type=str, default="/lustre/fsw/portfolios/convai/users/kevinhu/debug", help="Output directory for generated audio")
    parser.add_argument("--asr_model", type=str, default="base", help="Whisper model size for transcription (tiny, base, small, medium, large)")
    parser.add_argument("--transcribe", action="store_true", help="Enable transcription of input wav files")
    parser.add_argument("--fallback_text", type=str, default="Hello, this is a test.", help="Fallback text if transcription fails")
    parser.add_argument("--paraphrase_user_text", action="store_true", help="Use LLM to paraphrase user transcription text.")
    parser.add_argument('--llm_model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="LLM model to use for generating agent responses.")
    parser.add_argument("--audio_temperature", type=float, default=0.8, help="Audio generation temperature for MoonCast TTS")
    parser.add_argument("--audio_top_k", type=int, default=10, help="Audio generation top-k for MoonCast TTS")
    
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
        print(f"Loading Whisper model: {args.asr_model}")
        asr_model = whisper.load_model(args.asr_model)
        print("Whisper model loaded successfully.")
    
    # Initialize LLM pipeline if paraphrasing is enabled
    llm_pipeline = None
    if args.paraphrase_user_text:
        print(f"Loading LLM model: {args.llm_model}")
        llm_pipeline = create_pipeline(args.llm_model)
        print("LLM model loaded successfully.")
    
    # Setup MoonCast TTS
    mooncast_model, device = setup_mooncast_tts()
    if mooncast_model is None:
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
            mooncast_model, 
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
