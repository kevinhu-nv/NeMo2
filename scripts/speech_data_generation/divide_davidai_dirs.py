#!/usr/bin/env python3
import argparse
import glob
import json
import os
from pathlib import Path

def divide_davidai_dirs(davidai_base_path: str, num_buckets: int, output_dir: str):
    # Get all directories containing transcription.json
    all_seglst_path = glob.glob(f'{davidai_base_path}/*/transcription.json')
    dir_paths = [os.path.dirname(path) for path in all_seglst_path]
    
    # Calculate dirs per bucket
    dirs_per_bucket = len(dir_paths) // num_buckets
    remainder = len(dir_paths) % num_buckets
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Divide directories into buckets
    start_idx = 0
    for i in range(num_buckets):
        # Add one extra dir to early buckets if there's a remainder
        current_bucket_size = dirs_per_bucket + (1 if i < remainder else 0)
        end_idx = start_idx + current_bucket_size
        
        bucket_dirs = dir_paths[start_idx:end_idx]
        
        # Write bucket to file
        output_file = os.path.join(output_dir, f'davidai_dirs_bucket_{i}.txt')
        with open(output_file, 'w') as f:
            for dir_path in bucket_dirs:
                f.write(f"{dir_path}\n")
        
        start_idx = end_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Divide DavidAI directories into buckets')
    parser.add_argument('--davidai_path', type=str, required=True,
                        help='Base path to DavidAI directory (containing subdirs with transcription.json)')
    parser.add_argument('--num_buckets', type=int, required=True,
                        help='Number of buckets to divide into')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the bucket files')
    
    args = parser.parse_args()
    divide_davidai_dirs(args.davidai_path, args.num_buckets, args.output_dir)
