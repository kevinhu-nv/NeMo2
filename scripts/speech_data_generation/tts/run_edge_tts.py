#!/usr/bin/env python3
"""
TTS script using Edge TTS.
Edge TTS is Microsoft's cloud-based text-to-speech service.
"""

import os
import asyncio
import argparse
import glob
import random
import re
from pathlib import Path
from nemo.collections import asr as nemo_asr
import transformers
import torchaudio
import numpy as np
import edge_tts

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

def get_available_voices():
    """Get list of available Edge TTS voices."""
    # Popular English voices for variety
    voices = [
        "en-US-JennyNeural",
        "en-US-GuyNeural", 
        "en-US-AriaNeural",
        "en-US-DavisNeural",
        "en-US-JaneNeural",
        "en-US-JasonNeural",
        "en-US-NancyNeural",
        "en-US-SaraNeural",
        "en-US-TonyNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
        "en-AU-NatashaNeural",
        "en-AU-WilliamNeural",
        "en-CA-ClaraNeural",
        "en-CA-LiamNeural",
        "en-IN-NeerjaNeural",
        "en-IN-PrabhatNeural",
        "en-IE-EmilyNeural",
        "en-IE-ConnorNeural",
        "en-NZ-MollyNeural",
        "en-NZ-MitchellNeural",
        "en-ZA-LeahNeural",
        "en-ZA-LukeNeural"
    ]
    return voices

def get_random_voice():
    """Get a random voice from available voices."""
    voices = get_available_voices()
    return random.choice(voices)

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

async def generate_tts_audio(text, output_path, voice="en-US-JennyNeural", rate="+0%", volume="+0%"):
    """Generate TTS audio using Edge TTS."""
    try:
        # Create Edge TTS communicate object
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate, volume=volume)
        
        # Save as MP3 first
        temp_mp3_path = output_path.replace('.wav', '_temp.mp3')
        await communicate.save(temp_mp3_path)
        
        # Convert MP3 to WAV using torchaudio
        try:
            waveform, sample_rate = torchaudio.load(temp_mp3_path)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampled = torchaudio.functional.resample(waveform, sample_rate, 16000)
                waveform = resampled
                sample_rate = 16000
            
            # Save as WAV
            import soundfile as sf
            sf.write(output_path, waveform.squeeze().numpy(), sample_rate)
            
            # Clean up temp file
            os.remove(temp_mp3_path)
            
        except Exception as e:
            print(f"Error converting MP3 to WAV: {e}")
            # Fallback: just rename the MP3 file
            os.rename(temp_mp3_path, output_path.replace('.wav', '.mp3'))
            return False

        return True
    except Exception as e:
        print(f"Error generating TTS audio: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio using Edge TTS with transcribed text from wav files")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory containing input wav files to transcribe")
    parser.add_argument("--out_dir", type=str, default="/lustre/fsw/portfolios/convai/users/kevinhu/debug", help="Output directory for generated audio")
    parser.add_argument("--asr_model", type=str, default="nvidia/parakeet-tdt-0.6b-v2", help="NeMo ASR model for transcription")
    parser.add_argument("--transcribe", action="store_true", help="Enable transcription of input wav files")
    parser.add_argument("--fallback_text", type=str, default="Hello, this is a test.", help="Fallback text if transcription fails")
    parser.add_argument("--paraphrase_user_text", action="store_true", help="Use LLM to paraphrase user transcription text.")
    parser.add_argument('--llm_model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="LLM model to use for generating agent responses.")
    parser.add_argument("--voice", type=str, default="en-US-JennyNeural", help="Edge TTS voice to use")
    parser.add_argument("--randomize_voice", action="store_true", help="Randomize voice for each audio file")
    parser.add_argument("--rate", type=str, default="+0%", help="Speech rate adjustment")
    parser.add_argument("--volume", type=str, default="+0%", help="Volume adjustment")
    
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
        
        # Select voice (random or specified)
        if args.randomize_voice:
            selected_voice = get_random_voice()
            print(f"Using random voice: {selected_voice}")
        else:
            selected_voice = args.voice
        
        success = asyncio.run(generate_tts_audio(text, output_path, selected_voice, args.rate, args.volume))
        
        if success:
            print(f"Saved -> {output_path}")
            print(f"  Text: {text}")
        else:
            print(f"Failed to generate audio for {subdir_name}")
    
    print(f"\nGenerated {len(items)} audio files in {args.out_dir}")

if __name__ == "__main__":
    main()
