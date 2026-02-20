import copy
import os
import json
import glob
import re
import sys
import argparse
from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from lhotse import AudioSource, CutSet, Recording, SupervisionSegment
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
    # parser.add_argument("--in_shar_dir", default="/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/Mixtral8x22b_MMLPC_en")
    parser.add_argument("--in_shar_dir", default="/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/triviaqa_train/shar_duplex/source_align")
    # parser.add_argument("--in_shar_dir", default="/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/ultrachat_v2/shar_duplex/source_align")
    # parser.add_argument("--in_shar_dir", default="/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/ultrachat/shar_duplex/source_align")
    # parser.add_argument("--in_shar_dir", default="/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex//topic_v2/Meta-Llama-3.1-70B-Instruct/shar_duplex/source_align")
    parser.add_argument("--start_manifest_number", type=int, default=0, help="Starting manifest number.")
    # parser.add_argument("--ctm_dir", default="/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/Mixtral8x22b_MMLPC_en/manifest_0")
    # parser.add_argument("--output_manifest_dir", default="/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/Mixtral8x22b_MMLPC_en/manifest_0/")
    # parser.add_argument("--num_shard", type=int, default=1, help="Output shard number to save outputs.")
    # parser.add_argument("--shard_size", type=int, default=1024, help="Output shard number to save outputs.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    in_shar_dir = args.in_shar_dir   
    cuts_files = natsorted(glob.glob(f"{in_shar_dir}/*/source_align/cuts.*.jsonl.gz", recursive=True))
    
    pattern = r"<\|.*?\|>"
    m_number_pattern = re.compile(r"/manifest_(\d+)/")

    # import pdb; pdb.set_trace()
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
                    if supervision.text == '':
                        print(f'\033[92mEmpty user text: {cuts_file}, id: {cut.id}\033[0m')
                        no_alignment = True
                        break
                    match = re.search(pattern, supervision.text)
                    if not match:
                        # import pdb; pdb.set_trace()
                        print(f'\033[92mcuts file has no alignment: {cuts_file}, id: {cut.id}\033[0m')
                        no_alignment = True
                        break
            if no_alignment:
                break

if __name__ == "__main__":
    main()