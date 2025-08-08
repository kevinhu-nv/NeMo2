import argparse
import copy
import csv
import glob
import json
import os
import random
import shutil
from io import BytesIO
from pathlib import Path
import re

import librosa
import numpy as np
import soundfile as sf
from lhotse import AudioSource, CutSet, Recording, SupervisionSegment
from lhotse.array import Array, TemporalArray
from lhotse.audio import RecordingSet, save_audio
from lhotse.cut.base import Cut
from lhotse.features.base import Features, FeatureSet
from lhotse.shar.readers.lazy import LazySharIterator
from lhotse.shar.writers import AudioTarWriter
from tqdm import tqdm

def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)

def update_shar_with_new_recording(in_dir, shar_index, out_shar_dir, new_wav_dir):
    cuts_name = [f"{in_dir}/cuts.{shar_index:06d}.jsonl.gz"]

    cuts = LazySharIterator(
        {
            "cuts": cuts_name,
            "recording": [name.replace("cuts", "recording").replace("jsonl.gz", "tar") for name in cuts_name],
            "target_audio": [name.replace("cuts", "target_audio").replace("jsonl.gz", "tar") for name in cuts_name],
        }
    )

    updated_cuts = []
    
    # Get all subdirectories (which are the cut IDs)
    subdirs = [d for d in os.listdir(new_wav_dir) if os.path.isdir(os.path.join(new_wav_dir, d))]
    subdirs.sort()  # Sort to ensure consistent ordering
    subdir_index = 0

    for cut in tqdm(cuts):
        if subdir_index >= len(subdirs):
            print("No more audio directories available.")
            break

        subdir_name = subdirs[subdir_index]
        subdir_index += 1
        
        # Construct path to input.wav in the subdirectory
        new_wav_path = os.path.join(new_wav_dir, subdir_name, "input.wav")
        
        if not os.path.exists(new_wav_path):
            print(f"Warning: {new_wav_path} does not exist, skipping...")
            continue

        # Load the new audio and calculate its duration
        new_audio, sample_rate = librosa.load(new_wav_path, sr=None)
        new_duration = len(new_audio) / sample_rate

        # Update recording with new cut ID (directory name)
        new_recording = Recording(
            id=subdir_name,  # Use directory name as cut ID
            sources=[AudioSource(type="file", channels=[0], source=new_wav_path)],
            sampling_rate=sample_rate,
            num_samples=len(new_audio),
            duration=new_duration,
            channel_ids=[0]
        )
        
        # Update cut with new ID and recording
        cut.id = subdir_name
        cut.recording = new_recording
        cut.duration = new_duration

        # Update target_audio to be the same as recording
        cut.target_audio = new_recording

        # Update supervisions to any 4 turns
        assert len(cut.supervisions) >= 2, "Expected at least two supervision segments."
        fixed_duration = new_duration / 4
        cut.supervisions[0].duration = fixed_duration
        cut.supervisions[0].text = ""
        cut.supervisions[1].start = cut.supervisions[0].start + cut.supervisions[0].duration
        cut.supervisions[1].duration = fixed_duration
        cut.supervisions[1].text = ""
        cut.supervisions[2].start = cut.supervisions[1].start + cut.supervisions[1].duration
        cut.supervisions[2].duration = fixed_duration
        cut.supervisions[2].text = ""
        cut.supervisions[3].start = cut.supervisions[2].start + cut.supervisions[2].duration
        cut.supervisions[3].duration = fixed_duration
        cut.supervisions[3].text = ""

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

    args = parser.parse_args()

    update_shar_with_new_recording(
        in_dir=args.in_dir,
        shar_index=args.shar_index,
        out_shar_dir=args.out_shar_dir,
        new_wav_dir=args.new_wav_dir
    )

if __name__ == "__main__":
    main()
