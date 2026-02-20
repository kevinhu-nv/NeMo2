import json
import argparse
import os
from pathlib import Path
import tarfile
import gzip
import shutil
from tqdm import tqdm
from lhotse import CutSet, RecordingSet, Recording, SupervisionSegment
from lhotse.shar.readers.lazy import LazySharIterator
from lhotse.shar.writers import AudioTarWriter
import numpy as np
from lhotse.audio import RecordingSet, save_audio
from io import BytesIO
import copy
import re


def is_bad_entry(entry):
    return entry['text'] == ''


def create_single_turn_jsonl(root_dir, input_manifest, out_dir, sampling_rate, turn_silence_sec, input_manifest_path=None):
    
    # Use direct path if provided, otherwise combine root_dir and input_manifest
    if input_manifest_path:
        manifest_path = Path(input_manifest_path)
    else:
        manifest_path = Path(root_dir, input_manifest)
    print(f"Processing manifest: {manifest_path}")
    
    # Load manifest entries
    entries = {}
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            record = json.loads(line_stripped)
            # Store the full path if root_dir is provided
            audio_path = record['audio_filepath']
            if root_dir:
                audio_path = str(Path(root_dir) / audio_path)
            entries[audio_path] = record

    # Create recordings and cuts
    user_recordings = []
    
    # Create manifest subdirectory first
    out_dir = Path(out_dir) / '.'.join(input_manifest.split("/")[-1].split(".")[:-1])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(out_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for wav files inside manifest directory
    temp_dir = manifest_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for audio_path, entry in tqdm(entries.items(), desc="Creating recordings"):
        try:
            # Create recording directly from the wav file path
            recording = Recording.from_file(audio_path)
            recording.id = audio_path
            user_recordings.append(recording)
            count += 1
        except Exception as e:
            print(f"Warning: Could not read audio file {audio_path}: {str(e)}")
            continue

    print(f"Created {len(user_recordings)} recordings")
    
    # Create cuts
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(user_recordings))
    
    # Attach text and metadata
    new_cuts = []
    for cut in tqdm(cuts, desc="Attaching metadata"):

        orig_sampleing_rate = cut.recording.sampling_rate
        num_silence_samples = int(turn_silence_sec * orig_sampleing_rate)
        silence_padding = np.zeros((1, num_silence_samples))

        entry = entries[cut.recording.id]

        if is_bad_entry(entry):
            continue
        
        # Add supervision
        new_cut = copy.deepcopy(cut)
        cut_id = os.path.basename(cut.id).split(".")[0]
        new_cut.id = cut_id
        new_cut.supervisions = [
            SupervisionSegment(
                id=cut_id,
                recording_id=cut_id,
                start=0,
                duration=0,  # will update below  
                text=entry["text"],
                speaker="user",
                language="EN",
            )
        ]

        # Create zero-filled agent audio with same length as user audio
        user_audio = cut.recording.load_audio()
        user_audio = np.concatenate([user_audio, silence_padding], axis=1)
        agent_audio = np.zeros_like(user_audio)

        new_cut.supervisions[0].duration = agent_audio.shape[1] / orig_sampleing_rate
        new_cut.duration = new_cut.supervisions[0].duration
        new_cut.start = 0.0
    
        # breakpoint()
        temp_agent_wav = temp_dir / f"{cut_id}_agent.wav"
        save_audio(temp_agent_wav, agent_audio, sampling_rate)
        target_recording = Recording.from_file(temp_agent_wav)
        target_recording.id = f"{cut_id}_agent"
        new_cut.target_audio = target_recording

        user_stream = BytesIO()
        agent_stream = BytesIO()
        save_audio(dest=user_stream, src=user_audio, sampling_rate=sampling_rate, format="wav")
        save_audio(dest=agent_stream, src=agent_audio, sampling_rate=sampling_rate, format="wav")
        user_stream.seek(0)
        agent_stream.seek(0)
        new_cut.recording = Recording.from_bytes(user_stream.getvalue(), f"{cut_id}_user")
        new_cut.target_audio = Recording.from_bytes(agent_stream.getvalue(), f"{cut_id}_agent")
        new_cuts.append(new_cut)

    # Create shar files
    print("Creating shar files...")    
    cuts = CutSet(cuts=new_cuts)
    shard_size = 1024
    # Export cuts to shar format with single shard
    exported = cuts.to_shar(
        out_dir, 
        fields={
            "recording": "wav",  # Changed from flac to wav
            "target_audio": "wav"  # Changed from flac to wav
        }, 
        num_jobs=1, 
        shard_size=shard_size
    )
    print(f"Created shar files: {exported}")

    # Clean up temporary directory
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

    ##################
    # sanity check
    cuts_name = [f"{out_dir}/cuts.000000.jsonl.gz"]

    def replace_cuts_name(cuts_name, new_name=None, new_root=None):
        cuts_name = [name.replace("jsonl.gz", "tar") for name in cuts_name]
        if new_root is not None:
            cuts_name = [os.path.join(new_root, os.path.basename(name)) for name in cuts_name]
        if new_name is not None:
            cuts_name = [name.replace("cuts", new_name) for name in cuts_name]
        return cuts_name

    cuts = LazySharIterator(
        {
            "cuts": cuts_name,
            "recording": replace_cuts_name(cuts_name, new_name="recording"),
            "target_audio": replace_cuts_name(cuts_name, new_name="target_audio"),
        }
    )
    num_cuts = len(cuts)
    n = 0
    for j, cut in enumerate(cuts):
        print(f'Processing {j}/{num_cuts} cuts...')
        print("supervision[0].text:", cut.supervisions[0].text)
        # print("supervision[1].text:", cut.supervisions[1].text)
        print("loading recording...")
        cut.recording.load_audio()
        print("DONE")
        print("loading target_audio...")
        cut.target_audio.load_audio()
        print("DONE")
        if n > 10:
            break
        n += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/asr_evaluator/datasets/HF-audio',
                        required=False,
                        help='Root directory to prepend to audio_filepath in manifest (optional)')
    parser.add_argument("--input_manifest", type=str, 
                        default='open-asr-leaderboarddatasets-test-only-librispeech-test.clean.json',
                        required=False,
                        help="Path to your input JSON lines (each with id, user, assistant).")
    parser.add_argument("--input_manifest_path", type=str, 
                        default=None,
                        required=False,
                        help="Direct path to input manifest file. If provided, this takes precedence over root_dir + input_manifest.")
    parser.add_argument("--out_dir", type=str,
                        default='/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/ASR/',
                        required=False,
                        help="Directory to write the SHAR files (cuts.*.jsonl.gz, recording.*.tar, target_audio.*.tar).")
    parser.add_argument("--sampling_rate", type=int, default=16000,
                        help="Sampling rate of the wav file.")
    parser.add_argument('--turn_silence_sec', type=float, default=0.8, help='10 tokens of silence in seconds')
    args = parser.parse_args()

    create_single_turn_jsonl(args.root_dir, args.input_manifest, args.out_dir, args.sampling_rate, args.turn_silence_sec, args.input_manifest_path)

if __name__ == "__main__":
    main()
