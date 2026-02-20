import copy
import os
import json
import glob
import re
import sys
import argparse
from tqdm import tqdm
from lhotse import CutSet
from lhotse.shar.readers.lazy import LazySharIterator
from pathlib import Path

def read_ctm_file(ctm_filepath, include_end=True):
    """Read a CTM file and return a concatenated string of the words with their start times."""
    transcript = []
    max_integer = None
    try:
        with open(ctm_filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    word_start = float(parts[2])
                    duration = float(parts[3])
                    word = parts[4]

                    adjusted_integer = round(word_start / 0.08)
                    adjusted_word_start = f"<|{adjusted_integer}|>"
                    adjusted_integer_end = round((word_start + duration) / 0.08)
                    adjusted_word_end = f"<|{adjusted_integer_end}|>"

                    if max_integer is None or adjusted_integer > max_integer:
                        max_integer = adjusted_integer

                    if include_end:
                        transcript.append(f"{adjusted_word_start} {word} {adjusted_word_end}")
                    else:
                        transcript.append(f"{adjusted_word_start} {word}")
    except FileNotFoundError:
        print(f"CTM file not found: {ctm_filepath}")
    return ' '.join(transcript), max_integer

def replace_cuts_name(cuts_name, new_name=None, new_root=None):
    cuts_name = [name.replace("jsonl.gz", "tar") for name in cuts_name]
    if new_root is not None:
        cuts_name = [os.path.join(new_root, os.path.basename(name)) for name in cuts_name]
    if new_name is not None:
        cuts_name = [name.replace("cuts", new_name) for name in cuts_name]
    return cuts_name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_shar_dir", default="/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/Mixtral8x22b_MMLPC_en/manifest_0")
    parser.add_argument("--ctm_dir", default="/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/Mixtral8x22b_MMLPC_en/manifest_0")
    # parser.add_argument("--output_manifest_dir", default="/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/Mixtral8x22b_MMLPC_en/manifest_0/")
    parser.add_argument("--num_shard", type=int, default=1, help="Output shard number to save outputs.")
    # parser.add_argument("--shard_size", type=int, default=1024, help="Output shard number to save outputs.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    in_shar_dir = args.in_shar_dir   
    cuts_files = sorted(glob.glob(f"{in_shar_dir}/cuts.*.jsonl.gz"))
    
    out_dir = f'{args.ctm_dir}/source_align'
    new_cuts = []
    for cuts_file in cuts_files:
        fname = os.path.basename(cuts_file)
        ctm_dir = f'{args.ctm_dir}/nfa_manifest.jsonl_align/ctm/words'

        cuts = LazySharIterator(
            {
                "cuts": [cuts_file],
                "recording": replace_cuts_name([cuts_file], new_name="recording"),
                "target_audio": replace_cuts_name([cuts_file], new_name="target_audio"),
            }
        )

        for j, cut in enumerate(tqdm(cuts, desc=f"Processing {os.path.basename(cuts_file)}")):
            new_cut = copy.deepcopy(cut)
            skip_cut = False
            for i in range(0, len(cut.supervisions), 2):
                ctm_filepath = os.path.join(ctm_dir, f"{cut.id}_user{i//2}.ctm")
                # Update the text with timestamped text from CTM file
                time_text, _ = read_ctm_file(ctm_filepath)
                if not time_text.strip():
                    print(f"Skipping cut {cut.id} due to empty timed_text")
                    skip_cut = True
                    break
                new_cut.supervisions[i].text = time_text
            
            if not skip_cut:
                new_cuts.append(new_cut)
        
    ################################
    # Output to a new shar
    cuts = CutSet(cuts=new_cuts)
    num_total = len(cuts)
    shard_size = int((num_total+1) / args.num_shard)
    if num_total % shard_size != 0:
        shard_size += 1
    print(f"shard_size {shard_size} num_shards {args.num_shard}")    
    
    print(f"...Making Shars")
    out_shar_dir = Path(out_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
        
    exported = cuts.to_shar(
        out_shar_dir, fields={"recording": "flac", "target_audio": "flac"}, num_jobs=1, shard_size=shard_size
    )
    print(f"...Shars created in {out_shar_dir}")

    ###################
    # Sanity check
    cuts_name = [f"{out_shar_dir}/cuts.000000.jsonl.gz"]    
    cuts = LazySharIterator(
        {
            "cuts": cuts_name,
            "recording": replace_cuts_name(cuts_name, new_name="recording"),
            "target_audio": replace_cuts_name(cuts_name, new_name="target_audio"),
        }
    )
    num_cuts = len(cuts)
    n = 0
    print(f'Sanity check cuts {cuts_name} ')
    for j, cut in enumerate(cuts):
        print(f'Processing {j}/{num_cuts} cuts...')
        for i in range(len(cut.supervisions)):
            print(f"supervision[{i}].text:", cut.supervisions[i].text)
        print("loading recording...")
        cut.recording.load_audio()
        print("DONE")
        print("loading target_audio...")
        cut.target_audio.load_audio()
        print("DONE")
        if n > 10:
            break
        n += 1

if __name__ == "__main__":
    main()