import copy
import os
import json
import glob
import re
import sys
import argparse
from tqdm import tqdm
from lhotse.shar.readers.lazy import LazySharIterator
from pathlib import Path
from natsort import natsorted

# Sanity check all the shars and find problematic ones

def replace_cuts_name(cuts_name, new_name=None, new_root=None):
    cuts_name = [name.replace("jsonl.gz", "tar") for name in cuts_name]
    if new_root is not None:
        cuts_name = [os.path.join(new_root, os.path.basename(name)) for name in cuts_name]
    if new_name is not None:
        cuts_name = [name.replace("cuts", new_name) for name in cuts_name]
    return cuts_name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_shar_dir", default="/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/kevinhu/ASR/MMLPC/en/tarred_train/")
    parser.add_argument("--start_manifest_number", type=int, default=0, help="Starting manifest number.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    in_shar_dir = args.in_shar_dir   
    cuts_files = natsorted(glob.glob(f"{in_shar_dir}/*/cuts.*.jsonl.gz", recursive=True))
    
    pattern = r"<\|.*?\|>"
    m_number_pattern = re.compile(r"/manifest_(\d+)/")

    for cuts_file in cuts_files:
        match = m_number_pattern.search(cuts_file)
        manifest_number = int(match.group(1))  
        if manifest_number < args.start_manifest_number:
            continue

        cuts = LazySharIterator(
            {
                "cuts": [cuts_file],
                "recording": replace_cuts_name([cuts_file], new_name="recording"),
                "target_audio": replace_cuts_name([cuts_file], new_name="target_audio"),
            }
        )

        no_alignment = False
        for j, cut in enumerate(tqdm(cuts, desc=f"Processing {cuts_file}")):
            for supervision in cut.supervisions:
                # Make sure user text is not empty and has alignment
                if supervision.speaker == 'user':
                    match = re.search(pattern, supervision.text)
                    if not match:
                        # import pdb; pdb.set_trace()
                        print(f'\033[92mcuts file has no alignment: {cuts_file}, id: {cut.id}\033[0m')
                        print(f'supervision.text: {supervision.text}')
                        no_alignment = True
                        break
            if no_alignment:
                break

if __name__ == "__main__":
    main()