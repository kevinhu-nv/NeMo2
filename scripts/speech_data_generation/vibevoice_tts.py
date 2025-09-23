import os, torch
import argparse
import glob
import random
from pathlib import Path
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from nemo.collections import asr as nemo_asr

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

def get_wav_files_and_speakers(wav_dir, voices_dir):
    """Get wav files and randomly assign speakers from available voice files."""
    available_voices = [
        "en-Alice_woman",
        "en-Carter_man", 
        "en-Frank_man",
        "en-Mary_woman_bgm",
        "en-Maya_woman"
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
            # Store the original subdirectory name along with wav_path and voice
            items.append((wav_path, selected_voice, subdir_name))
        else:
            print(f"Warning: No wav file found at {wav_path}")
    
    return items

def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio using VibeVoice with transcribed text from wav files")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory containing input wav files to transcribe")
    parser.add_argument("--voices_dir", type=str, default="/workspace/VibeVoice/demo/voices", help="Directory containing voice samples")
    parser.add_argument("--out_dir", type=str, default="/lustre/fsw/portfolios/convai/users/kevinhu/debug", help="Output directory for generated audio")
    parser.add_argument("--model", type=str, default="microsoft/VibeVoice-1.5B", help="VibeVoice model to use")
    parser.add_argument("--asr_model", type=str, default="nvidia/parakeet-tdt-0.6b-v2", help="NeMo ASR model for transcription")
    parser.add_argument("--transcribe", action="store_true", help="Enable transcription of input wav files")
    parser.add_argument("--fallback_text", type=str, default="Hello, this is a test.", help="Fallback text if transcription fails")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing audio files (reduce if you get tensor size errors)")
    
    args = parser.parse_args()
    
    # Get wav files and speakers
    items = get_wav_files_and_speakers(args.wav_dir, args.voices_dir)
    
    if not items:
        print("No valid wav files found with corresponding voice samples.")
        return
    
    print(f"Found {len(items)} wav files to process")
    
    # Initialize ASR model if transcription is enabled
    asr_model = None
    if args.transcribe:
        print(f"Loading ASR model: {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.asr_model).cuda()
        print("ASR model loaded successfully.")
    
    # Transcribe wav files if enabled
    transcribed_items = []
    for wav_file, speaker, subdir_name in items:
        if args.transcribe and asr_model:
            text = transcribe_audio_file(asr_model, wav_file)
            if not text:
                text = args.fallback_text
                print(f"Using fallback text for {subdir_name}")
            print(f"Transcribed {subdir_name}: {text}")
        else:
            text = args.fallback_text
            print(f"Using fallback text for {subdir_name}: {text}")
        
        transcribed_items.append((text, speaker, subdir_name))
    
    # Use the transcribed items for TTS generation
    items = transcribed_items
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading VibeVoice model: {args.model}")
    proc = VibeVoiceProcessor.from_pretrained(args.model)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    ).eval()
    model.set_ddpm_inference_steps(num_steps=14)
    
    # Process in batches to avoid tensor size issues
    batch_size = args.batch_size
    total_items = len(items)
    
    print(f"Processing {total_items} audio files in batches of {batch_size}...")
    
    for batch_start in range(0, total_items, batch_size):
        batch_end = min(batch_start + batch_size, total_items)
        batch_items = items[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}: items {batch_start+1}-{batch_end}")
        
        # Prepare texts and voice samples for this batch
        texts = [f"Speaker 1: {s}" for s, _spk, _subdir in batch_items]
        voice_samples = [[os.path.join(args.voices_dir, spk + ".wav")] for _s, spk, _subdir in batch_items]
        
        print("Generating TTS audio for batch...")
        inputs = proc(
            text=texts,
            voice_samples=voice_samples,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        
        try:
            outs = model.generate(
                **inputs,
                cfg_scale=1.3,
                tokenizer=proc.tokenizer,
                generation_config={"do_sample": False},
            )
        except RuntimeError as e:
            if "canUse32BitIndexMath" in str(e):
                print(f"Error: Tensor size too large for batch size {batch_size}")
                print("Try reducing --batch_size to 1 or 2")
                print(f"Original error: {e}")
                return
            else:
                raise e
        
        # Save generated audio files for this batch
        for i, (s, spk, subdir_name) in enumerate(batch_items):
            # Create subdirectory using the original subdirectory name
            subdir_path = os.path.join(args.out_dir, subdir_name)
            os.makedirs(subdir_path, exist_ok=True)
            
            # Save as input.wav in the subdirectory
            out_path = os.path.join(subdir_path, "input.wav")
            # Resample to 16kHz before saving
            import torchaudio
            audio = outs.speech_outputs[i]
            orig_sr = getattr(outs, "sampling_rate", 24000)  # fallback to 24kHz if not present
            if orig_sr != 16000:
                audio = torchaudio.functional.resample(audio, orig_sr, 16000)
            proc.save_audio(audio, out_path, sampling_rate=16000)
            print(f"Saved -> {out_path}")
            print(f"  Text: {s}")
            print(f"  Speaker: {spk}")
            print(f"  Subdir: {subdir_name}")
    
    print(f"\nGenerated {total_items} audio files in {args.out_dir}")

if __name__ == "__main__":
    main()
