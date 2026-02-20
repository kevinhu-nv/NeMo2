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
    pattern = r"<\|.*?\|>"
    match = re.search(pattern, entry["answer"])
    return match is None


def create_single_turn_jsonl(input_manifest, source_tar_dir, out_dir, sampling_rate, turn_silence_sec=0.8):
    """
    Reads JSON from 'input_manifest', extracts metadata from TAR files, and saves to a JSONL.GZ file.
    """
    manifest_index = int(Path(input_manifest).stem.split('_')[-1])
    
    # Load manifest entries
    entries = {}
    with open(input_manifest, 'r', encoding='utf-8') as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            record = json.loads(line_stripped)
            entries[record['audio_filepath'].replace(".wav", "")] = record

    # Create recordings and cuts
    user_recordings = []
    
    # Create manifest subdirectory first
    manifest_dir = Path(out_dir) / f"manifest_{manifest_index:06d}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for wav files inside manifest directory
    temp_dir = manifest_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for turn_id, entry in tqdm(entries.items(), desc="Creating recordings"):
        # Create recording from the wav file in source tar
        with tarfile.open(f"{source_tar_dir}/audio_{manifest_index}.tar", "r") as source_tar:
            try:
                audio_member = source_tar.getmember(f"{turn_id}.wav")
                # Extract to temporary file
                temp_wav = temp_dir / f"{turn_id}.wav"
                with open(temp_wav, 'wb') as f:
                    f.write(source_tar.extractfile(audio_member).read())
                
                # Create recording
                recording = Recording.from_file(temp_wav)
                user_recordings.append(recording)
                count += 1
                # if count > 1023:
                #     break
            except KeyError:
                print(f"Warning: Could not find audio file {turn_id}.wav in source tar")
                continue

    print(f"Created {len(user_recordings)} recordings")
    
    # Create cuts
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(user_recordings))
    
    # Attach text and metadata
    new_cuts = []
    for cut in tqdm(cuts, desc="Attaching metadata"):

        num_silence_samples = int(turn_silence_sec * sampling_rate)
        silence_padding = np.zeros((1, num_silence_samples))

        entry = entries[cut.recording.id]

        if is_bad_entry(entry):
            continue
        
        # Add supervision
        new_cut = copy.deepcopy(cut)
        new_cut.supervisions = [
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=0,
                duration=entry["duration"] + turn_silence_sec,
                text=entry["answer"],
                speaker="user",
                language="EN",
            )
        ]

        # Create zero-filled agent audio with same length as user audio
        user_audio = cut.recording.load_audio()
        user_audio = np.concatenate([user_audio, silence_padding], axis=1)
        agent_audio = np.zeros_like(user_audio)

        new_cut.duration = entry["duration"] + turn_silence_sec
        new_cut.start = 0.0
    
        temp_agent_wav = temp_dir / f"{cut.id}_agent.wav"
        save_audio(temp_agent_wav, agent_audio, sampling_rate)
        target_recording = Recording.from_file(temp_agent_wav)
        target_recording.id = f"{cut.id}_agent"  # Ensure unique ID
        new_cut.target_audio = target_recording

        user_stream = BytesIO()
        agent_stream = BytesIO()
        save_audio(dest=user_stream, src=user_audio, sampling_rate=sampling_rate, format="wav")
        save_audio(dest=agent_stream, src=agent_audio, sampling_rate=sampling_rate, format="wav")
        user_stream.seek(0)
        agent_stream.seek(0)
        new_cut.recording = Recording.from_bytes(user_stream.getvalue(), f"{cut.id}_user")
        new_cut.target_audio = Recording.from_bytes(agent_stream.getvalue(), f"{cut.id}_agent")
        new_cuts.append(new_cut)

    # Create shar files
    print("Creating shar files...")
    out_dir = Path(out_dir) / f"manifest_{manifest_index:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
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
    parser.add_argument("--input_manifest_dir", type=str, 
                        default='/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/MMLPC/en/tarred_train/timestamp/',
                        required=False,
                        help="Path to your input JSON lines (each with id, user, assistant).")
    parser.add_argument("--source_tar_dir", type=str, 
                        default='/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/MMLPC/en/tarred_train',
                        required=False,
                        help="Directory containing recording.xxxxxx.tar and target_audio.xxxxxx.tar")
    parser.add_argument("--out_dir", type=str,
                        default='/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/ASR/MMLPC/en/tarred_train/timestamp/',
                        required=False,
                        help="Directory to write the SHAR files (cuts.*.jsonl.gz, recording.*.tar, target_audio.*.tar).")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="Index of the shard to process, e.g., 0 for manifest_000000.jsonl.")
    parser.add_argument("--sampling_rate", type=int, default=16000,
                        help="Sampling rate of the wav file.")
    parser.add_argument('--turn_silence_sec', type=float, default=0.8, help='10 tokens of silence in seconds')
    args = parser.parse_args()

    manifest_file = Path(args.input_manifest_dir) / f"manifest_{args.shard_index}.json"
    print(f"Processing manifest: {manifest_file}")
    create_single_turn_jsonl(manifest_file, args.source_tar_dir, args.out_dir, args.sampling_rate, args.turn_silence_sec)

if __name__ == "__main__":
    main()
