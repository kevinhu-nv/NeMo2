import argparse
import copy
import csv
import glob
import json
import os
import random
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
import re

import librosa
import numpy as np
import soundfile as sf
import torch
from lhotse import AudioSource, CutSet, Recording, SupervisionSegment
from lhotse.array import Array, TemporalArray
from lhotse.audio import RecordingSet, save_audio
from lhotse.cut.base import Cut
from lhotse.features.base import Features, FeatureSet
from lhotse.shar.readers.lazy import LazySharIterator
from lhotse.shar.writers import AudioTarWriter
from nemo.collections import asr as nemo_asr
from tqdm import tqdm
import transformers

# Import utils for LLM pipeline creation
try:
    import utils
except ImportError:
    print("Warning: utils module not found. LLM functionality will be disabled.")
    utils = None

def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)

def create_pipeline(model_id, token=""):
    # pip install --upgrade transformers
    # import pdb; pdb.set_trace()
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )
    return pipeline

def create_prompt(question, syste_prompt=None):
    DEFAULT_SYSTEM_PROMPT = "Based on the question, directly answer by one short sentence using spoken words. The answer should be at most 1 sentences with at most 20 words."
    sys_prompt = DEFAULT_SYSTEM_PROMPT if syste_prompt is None else syste_prompt
    prompt = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"{question}"},
    ]
    return prompt

def run_llm_pipeline(llm_pipeline, user_transcription):
    """Generate agent response using Meta-Llama-3.1-8B-Instruct based on user transcription."""
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

def transcribe_audio_segment(asr_model_obj, audio_path):
    """Transcribe an audio segment using NeMo ASR model and return text with end timestamp."""
    asr_outputs = asr_model_obj.transcribe([audio_path], timestamps=True)    
    
    if not asr_outputs:
        return "", 0.0
    
    # Get the transcription result
    result = asr_outputs[0]
    text = result.text if hasattr(result, 'text') else ""
    
    # Extract end timestamp of the last word based on the provided snippet structure
    end_timestamp = 0.0
    if hasattr(result, 'timestamp') and result.timestamp and 'word' in result.timestamp:
        word_timestamps = result.timestamp["word"]
        if word_timestamps and len(word_timestamps) > 0:
            # Get the end timestamp of the last word
            last_word = word_timestamps[-1]
            if isinstance(last_word, dict) and 'end' in last_word:
                end_timestamp = last_word['end']
            else:
                print(f"Warning: Unexpected word timestamp format: {last_word}")
    elif hasattr(result, 'end_time'):
        # Fallback to overall end time if word timestamps not available
        end_timestamp = result.end_time
    else:
        print("Warning: No timestamp information available in ASR result")
    
    return text, end_timestamp

def update_shar_with_new_recording(
        in_dir, shar_index, out_shar_dir, new_wav_dir, transcribe=False, asr_model="nvidia/parakeet-tdt-0.6b-v2", 
        use_llm=False, llm_model="meta-llama/Meta-Llama-3.1-8B-Instruct", turn_silence_sec=0.32, use_end_timestamp=False,
        paraphrase_user_text=False, start_index=0, end_index=None, target_sample_rate=16000):
    cuts_name = [f"{in_dir}/cuts.{shar_index:06d}.jsonl.gz"]

    cuts = LazySharIterator(
        {
            "cuts": cuts_name,
            "recording": [name.replace("cuts", "recording").replace("jsonl.gz", "tar") for name in cuts_name],
            "target_audio": [name.replace("cuts", "target_audio").replace("jsonl.gz", "tar") for name in cuts_name],
        }
    )

    updated_cuts = []
    
    # Get all audio files (wav or m4a) directly from the directory
    audio_files = [f for f in os.listdir(new_wav_dir) if (f.endswith('.wav') or f.endswith('.m4a')) and os.path.isfile(os.path.join(new_wav_dir, f))]
    
    # Sort audio files alphabetically
    audio_files.sort()
    
    # Filter audio_files based on start_index and end_index
    if end_index is None:
        end_index = len(audio_files)
    # Ensure indices are within bounds
    start_index = max(0, start_index)
    end_index = min(len(audio_files), end_index)
    # Filter audio_files to only include the specified range
    filtered_audio_files = audio_files[start_index:end_index]
    
    print(f"Processing audio files {start_index} to {end_index-1} (total: {len(filtered_audio_files)} out of {len(audio_files)})")
    
    audio_file_index = 0

    # Initialize ASR model if transcription is enabled
    asr_model_obj = None
    if transcribe:
        print(f"Loading ASR model: {asr_model}")
        asr_model_obj = nemo_asr.models.ASRModel.from_pretrained(model_name=asr_model).cuda()
        print("ASR model loaded successfully.")

    # Initialize LLM pipeline if LLM generation is enabled
    llm_pipeline = None
    if use_llm:
        print(f"Loading LLM model: {llm_model}")
        llm_pipeline = create_pipeline(llm_model)
        print("LLM model loaded successfully.")

    for cut in tqdm(cuts):
        if audio_file_index >= len(filtered_audio_files):
            print("No more audio files available.")
            break

        audio_filename = filtered_audio_files[audio_file_index]
        audio_file_index += 1
        
        # Construct path to the audio file (supports both .wav and .m4a)
        audio_path = os.path.join(new_wav_dir, audio_filename)
        
        # Extract ID from filename (remove file extension)
        cut_id = os.path.splitext(audio_filename)[0]
        
        if not os.path.exists(audio_path):
            print(f"Warning: {audio_path} does not exist, skipping...")
            continue

        # Load the new audio and calculate its duration (supports wav and m4a)
        user_audio, sample_rate = librosa.load(audio_path, sr=target_sample_rate)
        new_duration = len(user_audio) / sample_rate

        # Estimate target audio duration and create silence padding
        # First transcribe audio if enabled
        user_transcription = ""
        user_end_timestamp = 0.0
        if transcribe and asr_model_obj is not None:
            try:
                user_transcription, user_end_timestamp = transcribe_audio_segment(asr_model_obj, audio_path)
                print(f"User transcription for {cut_id}: {user_transcription}")
                print(f"User end timestamp: {user_end_timestamp}")
            except Exception as e:
                print(f"Error transcribing {audio_path}: {e}")
                user_transcription = ""
                user_end_timestamp = 0.0

        # Use the LLM to generate a similar sentence to the user_transcription, instead of using it directly.
        if paraphrase_user_text and use_llm and llm_pipeline is not None and user_transcription:
            llm_prompt = create_prompt(user_transcription, syste_prompt="Paraphrase the following sentence, keeping the meaning but changing the wording. Keep it as a single sentence and natural for spoken English.")
            try:
                llm_user_transcription = run_llm_pipeline(llm_pipeline, llm_prompt)
                print(f"LLM-generated similar sentence for {cut_id}: {llm_user_transcription}")
            except Exception as e:
                print(f"Error generating similar sentence with LLM: {e}")
                llm_user_transcription = user_transcription
            print(f'user_transcription: {user_transcription}')
            print(f'llm_user_transcription: {llm_user_transcription}')
            user_transcription = llm_user_transcription

        # Generate agent response using LLM if enabled
        agent_response = ""
        if use_llm and llm_pipeline is not None and user_transcription:
            try:
                agent_response = run_llm_pipeline(llm_pipeline, user_transcription)
                print(f"Agent response for {cut_id}: {agent_response}")
            except Exception as e:
                print(f"Error generating agent response: {e}")
                agent_response = ""

        # Estimate agent response duration
        fixed_duration = new_duration
        if agent_response:
            # Prefer word count for speech, fallback to char count if needed
            words = agent_response.split()
            words_per_sec = 3.0  # average speaking rate
            min_duration = 0.5   # minimum duration for very short responses
            est_duration = max(len(words) / words_per_sec, min_duration)
        else:
            est_duration = fixed_duration

        # Create target audio as silence with length based on est_duration
        target_num_samples = int(est_duration * sample_rate)
        target_audio = np.zeros(target_num_samples, dtype=user_audio.dtype)

        # Concatenate to duplex format
        if use_end_timestamp:
            user_audio = user_audio[:int(user_end_timestamp * sample_rate)]
        silence_padding = np.zeros(int(turn_silence_sec * sample_rate))
        full_user_audio = np.concatenate([user_audio, silence_padding, np.zeros_like(target_audio)])
        full_agent_audio = np.concatenate([np.zeros_like(user_audio), silence_padding, target_audio])

        assert len(full_user_audio) == len(full_agent_audio), "User and agent audio must have the same length"

        # Save concatenated audio into recording and target_audio
        user_stream = BytesIO()
        agent_stream = BytesIO()
        save_audio(dest=user_stream, src=full_user_audio, sampling_rate=sample_rate, format="wav")
        save_audio(dest=agent_stream, src=full_agent_audio, sampling_rate=sample_rate, format="wav")
        user_stream.seek(0)
        agent_stream.seek(0)
        cut.recording = Recording.from_bytes(user_stream.getvalue(), f"{cut_id}_user")
        cut.target_audio = Recording.from_bytes(agent_stream.getvalue(), f"{cut_id}_agent")

        # Update cut with new ID and recording
        cut.id = cut_id
        cut.duration = len(full_user_audio) / sample_rate
        
        # Update supervisions
        cut.supervisions[0].start = 0.0
        
        # Use end timestamp for precise timing if available and enabled
        if use_end_timestamp and user_end_timestamp > 0:
            user_duration = user_end_timestamp
            print(f"Using end timestamp for user duration: {user_duration}")
        else:
            user_duration = len(user_audio) / sample_rate
            
        cut.supervisions[0].duration = user_duration
        cut.supervisions[0].text = user_transcription

        cut.supervisions[1].start = cut.supervisions[0].start + cut.supervisions[0].duration + turn_silence_sec
        cut.supervisions[1].duration = len(target_audio) / sample_rate
        cut.supervisions[1].text = agent_response
        
        cut.supervisions = cut.supervisions[:2]

        updated_cuts.append(cut)

    if updated_cuts:
        # Save updated cuts to a new SHAR
        out_shar_dir = Path(out_shar_dir)
        out_shar_dir.mkdir(parents=True, exist_ok=True)

        updated_cuts = CutSet(cuts=updated_cuts)
        updated_cuts.to_shar(out_shar_dir, fields={"recording": "flac", "target_audio": "flac"}, shard_size=len(updated_cuts))

        print(f"Updated SHAR files saved to {out_shar_dir}")
    else:
        print("No cuts were updated.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True, help="Input directory containing SHAR files.")
    parser.add_argument('--shar_index', type=int, required=True, help="Index of the SHAR to update.")
    parser.add_argument('--out_shar_dir', type=str, required=True, help="Output directory for updated SHAR files.")
    parser.add_argument('--new_wav_dir', type=str, required=True, help="Directory containing new audio files.")
    parser.add_argument('--transcribe', action='store_true', help="Enable transcription of audio files.")
    parser.add_argument('--asr_model', type=str, default="nvidia/parakeet-tdt-0.6b-v2", help="NeMo ASR model to use for transcription.")
    parser.add_argument('--use_llm', action='store_true', help="Enable LLM generation of agent responses.")
    parser.add_argument('--llm_model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="LLM model to use for generating agent responses.")
    parser.add_argument('--paraphrase_user_text', action='store_true', help="Use LLM to paraphrase user transcription text.")
    parser.add_argument('--use_end_timestamp', action='store_true', help="Use end timestamp of last word for precise timing.")
    parser.add_argument('--start_index', type=int, default=0, help="Start index for processing subdirectories (0-based).")
    parser.add_argument('--end_index', type=int, default=None, help="End index for processing subdirectories (exclusive, 0-based). If None, process until the end.")
    parser.add_argument('--target_sample_rate', type=int, default=16000, help="Target sample rate for resampling input wav files (default: 16000 Hz).")

    args = parser.parse_args()

    update_shar_with_new_recording(
        in_dir=args.in_dir,
        shar_index=args.shar_index,
        out_shar_dir=args.out_shar_dir,
        new_wav_dir=args.new_wav_dir,
        transcribe=args.transcribe,
        asr_model=args.asr_model,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        use_end_timestamp=args.use_end_timestamp,
        paraphrase_user_text=args.paraphrase_user_text,
        start_index=args.start_index,
        end_index=args.end_index,
        target_sample_rate=args.target_sample_rate
    )

if __name__ == "__main__":
    main()
