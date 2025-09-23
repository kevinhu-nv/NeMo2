#!/usr/bin/env python3
"""
TTS script using Bark TTS model.
Bark is a popular, easy-to-install TTS model that generates very natural speech.
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
    """Get wav files and randomly assign speakers from available voice presets."""
    # Bark TTS voice presets
    available_voices = [
        "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3", "v2/en_speaker_4",
        "v2/en_speaker_5", "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9"
    ]
    
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
            # Randomly select a voice from available voices
            selected_voice = random.choice(available_voices)
            items.append((wav_path, selected_voice, subdir_name))
        else:
            print(f"Warning: No wav file found at {wav_path}")
    
    return items

def setup_bark():
    """Setup Bark TTS model."""
    try:
        # Import bark after installation
        from bark import SAMPLE_RATE, generate_audio, preload_models
        from bark.generation import load_codec_model, generate_text_semantic
        from bark.api import semantic_to_waveform
        
        print("Loading Bark TTS models...")
        # Preload models for faster generation
        preload_models()
        print("Bark TTS model loaded successfully.")
        return True
    except ImportError:
        print("Bark TTS not found. Please install it first:")
        print("pip install bark")
        return False
    except Exception as e:
        print(f"Error loading Bark TTS: {e}")
        return False

def generate_tts_audio(text, voice, output_path):
    """Generate TTS audio using Bark TTS."""
    try:
        from bark import SAMPLE_RATE, generate_audio
        
        # Generate audio
        audio_array = generate_audio(text, history_prompt=voice)
        
        # Save audio
        import soundfile as sf
        sf.write(output_path, audio_array, SAMPLE_RATE)
        
        # Resample to 16kHz if needed
        if SAMPLE_RATE != 16000:
            import torchaudio
            audio, sr = torchaudio.load(output_path)
            resampled = torchaudio.functional.resample(audio, sr, 16000)
            torchaudio.save(output_path, resampled, 16000)
        
        return True
    except Exception as e:
        print(f"Error generating TTS audio: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio using Bark TTS with transcribed text from wav files")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory containing input wav files to transcribe")
    parser.add_argument("--out_dir", type=str, default="/lustre/fsw/portfolios/convai/users/kevinhu/debug", help="Output directory for generated audio")
    parser.add_argument("--asr_model", type=str, default="nvidia/parakeet-tdt-0.6b-v2", help="NeMo ASR model for transcription")
    parser.add_argument("--transcribe", action="store_true", help="Enable transcription of input wav files")
    parser.add_argument("--fallback_text", type=str, default="Hello, this is a test.", help="Fallback text if transcription fails")
    parser.add_argument("--paraphrase_user_text", action="store_true", help="Use LLM to paraphrase user transcription text.")
    parser.add_argument('--llm_model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="LLM model to use for generating agent responses.")
    parser.add_argument("--voice", type=str, default=None, help="Specific voice to use (if not provided, will randomly select)")
    
    args = parser.parse_args()
    
    # Get wav files and speakers
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
    
    # Setup Bark TTS
    if not setup_bark():
        return
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Process each item
    for i, (wav_file, speaker, subdir_name) in enumerate(items):
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
        
        # Use specific voice if provided, otherwise use the randomly selected one
        voice_to_use = args.voice if args.voice else speaker
        print(f"Using voice: {voice_to_use}")
        
        # Create output subdirectory
        subdir_path = os.path.join(args.out_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)
        
        # Generate TTS audio
        output_path = os.path.join(subdir_path, "input.wav")
        print(f"Generating TTS audio...")
        
        success = generate_tts_audio(text, voice_to_use, output_path)
        
        if success:
            print(f"Saved -> {output_path}")
            print(f"  Text: {text}")
            print(f"  Voice: {voice_to_use}")
        else:
            print(f"Failed to generate audio for {subdir_name}")
    
    print(f"\nGenerated {len(items)} audio files in {args.out_dir}")

if __name__ == "__main__":
    main()