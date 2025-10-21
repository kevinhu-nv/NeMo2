#!/usr/bin/env python3
"""
Script to merge multiple shar files into a single shar file.

Usage:
    python merge_shar_files.py --shar_dirs dir1 dir2 dir3 --output_dir merged_shar --shard_size 1000

Or use a file containing paths:
    python merge_shar_files.py --shar_list_file paths.txt --output_dir merged_shar
"""

import argparse
import glob
import os
from pathlib import Path
from typing import List

from lhotse import CutSet
from lhotse.shar.readers.lazy import LazySharIterator
from tqdm import tqdm


def find_shar_files(directory: str) -> dict:
    """
    Find all shar file components (cuts, recording, target_audio) in a directory.
    
    Returns a dict with keys 'cuts', 'recording', 'target_audio' containing lists of file paths.
    """
    directory = Path(directory)
    
    # Find all cuts files
    cuts_files = sorted(glob.glob(str(directory / "cuts.*.jsonl.gz")))
    
    if not cuts_files:
        print(f"Warning: No cuts files found in {directory}")
        return None
    
    # Derive corresponding recording and target_audio files
    recording_files = [
        f.replace("cuts", "recording").replace("jsonl.gz", "tar") 
        for f in cuts_files
    ]
    target_audio_files = [
        f.replace("cuts", "target_audio").replace("jsonl.gz", "tar") 
        for f in cuts_files
    ]
    
    # Verify all files exist
    for file_list, file_type in [
        (cuts_files, "cuts"),
        (recording_files, "recording"),
        (target_audio_files, "target_audio")
    ]:
        for f in file_list:
            if not os.path.exists(f):
                print(f"Warning: {file_type} file not found: {f}")
    
    return {
        "cuts": cuts_files,
        "recording": recording_files,
        "target_audio": target_audio_files
    }


def merge_shar_files(
    shar_dirs: List[str],
    output_dir: str,
    shard_size: int = None,
    fields: dict = None
):
    """
    Merge multiple shar directories into a single shar output.
    
    Args:
        shar_dirs: List of directory paths containing shar files to merge
        output_dir: Output directory for the merged shar files
        shard_size: Number of cuts per shard. If None, all cuts go into one shard.
        fields: Dict specifying how to store each field (e.g., {"recording": "flac", "target_audio": "flac"})
    """
    if fields is None:
        fields = {"recording": "flac", "target_audio": "flac"}
    
    all_cuts = []
    
    print(f"Merging {len(shar_dirs)} shar directories...")
    
    for shar_dir in tqdm(shar_dirs, desc="Processing shar directories"):
        print(f"\nProcessing: {shar_dir}")
        
        # Find shar files in this directory
        shar_files = find_shar_files(shar_dir)
        
        if shar_files is None:
            print(f"Skipping {shar_dir} - no valid shar files found")
            continue
        
        # Create lazy iterator for this shar
        try:
            cuts_iter = LazySharIterator(shar_files)
            
            # Collect all cuts from this shar
            shar_cuts = []
            for cut in cuts_iter:
                shar_cuts.append(cut)
            
            print(f"  Found {len(shar_cuts)} cuts in {shar_dir}")
            all_cuts.extend(shar_cuts)
            
        except Exception as e:
            print(f"Error processing {shar_dir}: {e}")
            continue
    
    if not all_cuts:
        print("Error: No cuts found to merge!")
        return
    
    print(f"\nTotal cuts collected: {len(all_cuts)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to CutSet and save as shar
    print(f"\nWriting merged shar files to {output_dir}...")
    merged_cutset = CutSet(cuts=all_cuts)
    
    # If shard_size is not specified, use the total number of cuts
    if shard_size is None:
        shard_size = len(all_cuts)
    
    merged_cutset.to_shar(
        output_path,
        fields=fields,
        shard_size=shard_size
    )
    
    print(f"Successfully merged {len(all_cuts)} cuts into {output_dir}")
    print(f"Shard size: {shard_size}")


def read_shar_list_from_file(filepath: str) -> List[str]:
    """
    Read shar directory paths from a text file (one path per line).
    """
    with open(filepath, 'r') as f:
        paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple shar files into a single shar file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge shar files from multiple directories
  python merge_shar_files.py --shar_dirs dir1 dir2 dir3 --output_dir merged_shar
  
  # Merge with custom shard size
  python merge_shar_files.py --shar_dirs dir1 dir2 --output_dir merged_shar --shard_size 500
  
  # Read shar paths from a file
  python merge_shar_files.py --shar_list_file paths.txt --output_dir merged_shar
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--shar_dirs',
        nargs='+',
        help="List of directories containing shar files to merge"
    )
    input_group.add_argument(
        '--shar_list_file',
        type=str,
        help="Path to text file containing shar directory paths (one per line)"
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Output directory for merged shar files"
    )
    parser.add_argument(
        '--shard_size',
        type=int,
        default=None,
        help="Number of cuts per shard. If not specified, all cuts go into one shard."
    )
    parser.add_argument(
        '--recording_format',
        type=str,
        default='flac',
        choices=['flac', 'wav', 'opus'],
        help="Audio format for recording field (default: flac)"
    )
    parser.add_argument(
        '--target_audio_format',
        type=str,
        default='flac',
        choices=['flac', 'wav', 'opus'],
        help="Audio format for target_audio field (default: flac)"
    )
    
    args = parser.parse_args()
    
    # Get list of shar directories
    if args.shar_list_file:
        shar_dirs = read_shar_list_from_file(args.shar_list_file)
        print(f"Loaded {len(shar_dirs)} shar paths from {args.shar_list_file}")
    else:
        shar_dirs = args.shar_dirs
    
    # Validate that directories exist
    valid_shar_dirs = []
    for shar_dir in shar_dirs:
        if os.path.isdir(shar_dir):
            valid_shar_dirs.append(shar_dir)
        else:
            print(f"Warning: Directory not found: {shar_dir}")
    
    if not valid_shar_dirs:
        print("Error: No valid shar directories found!")
        return
    
    print(f"Processing {len(valid_shar_dirs)} valid shar directories")
    
    # Set up fields dict
    fields = {
        "recording": args.recording_format,
        "target_audio": args.target_audio_format
    }
    
    # Merge shar files
    merge_shar_files(
        shar_dirs=valid_shar_dirs,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        fields=fields
    )


if __name__ == "__main__":
    main()

