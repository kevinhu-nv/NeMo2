import json
import argparse
import os
from pathlib import Path
import tarfile
import gzip
import shutil
from tqdm import tqdm
from lhotse import CutSet
from lhotse.shar.readers.lazy import LazySharIterator

def read_json_from_tar(tar_path, json_name):
    """
    Reads a JSON file from a TAR archive.
    """
    with tarfile.open(tar_path, "r") as tar:
        member = tar.getmember(json_name)
        with tar.extractfile(member) as f:
            return json.load(f)

def extract_ordered_ids_from_tar(tar_path):
    """
    Extracts the order of IDs from the tar file based on the presence of .json files.
    """
    ordered_ids = []
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".json"):
                ordered_ids.append(member.name.replace(".json", ""))
    return ordered_ids            

def extract_and_write_filtered_tar(tar_path, output_tar_path, filtered_ids):
    """
    Extracts only the entries with IDs in `filtered_ids` from the source TAR and writes them to a new TAR file.
    """
    with tarfile.open(tar_path, "r") as tar:
        with tarfile.open(output_tar_path, "w") as out_tar:
            for member in tar.getmembers():
                member_id = member.name.rsplit(".", 1)[0]
                if member_id in filtered_ids:
                    out_tar.addfile(member, tar.extractfile(member))

def create_single_turn_jsonl(input_manifest, target_tar_dir, source_tar_dir, out_dir):
    """
    Reads JSON from 'input_manifest', extracts metadata from TAR files, and saves to a JSONL.GZ file.
    """

    manifest_index = Path(input_manifest).stem.split('_')[-1]
    ordered_ids = extract_ordered_ids_from_tar(f"{source_tar_dir}/recording.{manifest_index}.tar")

    entries = {}
    with open(input_manifest, 'r', encoding='utf-8') as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            record = json.loads(line_stripped)
            # Use the 'id' field as the key
            if 'id' in record:
                entries[record['id']] = record
            else:
                raise KeyError("The JSON record does not contain an 'id' field.")

    ordered_ids = [oid for oid in ordered_ids if oid in entries]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"cuts.{manifest_index}.jsonl.gz"

    with gzip.open(jsonl_path, 'wt', encoding='utf-8') as gz_file:
        for turn_id in tqdm(ordered_ids, desc=f"Processing {manifest_index}"):
            entry = entries[turn_id]

            user_meta = read_json_from_tar(f"{source_tar_dir}/recording.{manifest_index}.tar", f"{turn_id}.json")
            assistant_meta = read_json_from_tar(f"{target_tar_dir}/target_recording.{manifest_index}.tar", f"{turn_id}.json")

            # import pdb; pdb.set_trace()

            json_object = {
                "id": turn_id,
                "start": 0,
                "duration": user_meta["duration"] + assistant_meta["duration"],
                "channel": 0,
                "supervisions": [
                    {
                        "id": f"{turn_id}",
                        "recording_id": turn_id,
                        "start": 0,
                        "duration": user_meta["duration"],
                        "channel": 0,
                        "text": entry["user"],
                        "language": "EN",
                        "speaker": "user"
                    },
                    {
                        "id": f"{turn_id}",
                        "recording_id": turn_id,
                        "start": user_meta["duration"],
                        "duration": assistant_meta["duration"],
                        "channel": 0,
                        "text": entry["assistant"],
                        "language": "EN",
                        "speaker": "agent"
                    }
                ],
                "recording": {
                    "id": f"{turn_id}",
                    "sources": [
                        {
                            "type": "shar",
                            "channels": user_meta["channel_ids"],
                            "source": f"recording.{manifest_index}.tar?{turn_id}.flac"
                        }
                    ],
                    "sampling_rate": user_meta["sampling_rate"],
                    "num_samples": user_meta["num_samples"],
                    "duration": user_meta["duration"],
                    "channel_ids": user_meta["channel_ids"]
                },
                "custom": {
                    "target_audio": {
                        "id": f"{turn_id}",
                        "sources": [
                            {
                                "type": "shar",
                                "channels": assistant_meta["channel_ids"],
                                "source": f"target_audio.{manifest_index}.tar?{turn_id}.flac"
                            }
                        ],
                        "sampling_rate": assistant_meta["sampling_rate"],
                        "num_samples": assistant_meta["num_samples"],
                        "duration": assistant_meta["duration"],
                        "channel_ids": assistant_meta["channel_ids"]
                    },
                    "shard_origin": "",
                    "shar_epoch": 0,
                },
                "type": "MonoCut"
            }

            # import pdb; pdb.set_trace()

            gz_file.write(json.dumps(json_object) + '\n')
    print(f"Done creating JSONL.GZ file in: {jsonl_path}")

    filtered_recording_tar = out_dir / f"recording.{manifest_index}.tar"
    filtered_target_tar = out_dir / f"target_audio.{manifest_index}.tar"
    extract_and_write_filtered_tar(f"{source_tar_dir}/recording.{manifest_index}.tar", filtered_recording_tar, ordered_ids)
    extract_and_write_filtered_tar(f"{target_tar_dir}/target_recording.{manifest_index}.tar", filtered_target_tar, ordered_ids)
    
    ##################
    # sanity check
    cuts_name = [f"{out_dir}/cuts.{manifest_index}.jsonl.gz"]

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
            # "question_recording": replace_cuts_name(
            #     cuts_name, new_name="question_recording", new_root=in_dir_question
            # ),
        }
    )
    num_cuts = len(cuts)
    n = 0
    for j, cut in enumerate(cuts):
        print(f'Processing {j}/{num_cuts} cuts...')
        print("supervision[0].text:", cut.supervisions[0].text)
        print("supervision[1].text:", cut.supervisions[1].text)
        print("loading recording...")
        cut.recording.load_audio()
        print("DONE")
        print("loading target_audio...")
        cut.target_audio.load_audio()
        print("DONE")
        if n > 20:
            break
        n += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest_dir", type=str, 
                        default='/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/ultrachat',
                        required=False,
                        help="Path to your input JSON lines (each with id, user, assistant).")
    parser.add_argument("--target_tar_dir", type=str, 
                        default='/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/s2s_synthetic_data/ultrachat/ultrachat',
                        required=False,
                        help="Directory containing recording.xxxxxx.tar and target_audio.xxxxxx.tar")
    parser.add_argument("--source_tar_dir", type=str, 
                        default='/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/s2s_synthetic_data/ultrachat/ultrachat',
                        required=False,
                        help="Directory containing recording.xxxxxx.tar and target_audio.xxxxxx.tar")
    parser.add_argument("--out_dir", type=str,
                        default='/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/ultrachat/shar/ultrachat',
                        required=False,
                        help="Directory to write the SHAR files (cuts.*.jsonl.gz, recording.*.tar, target_audio.*.tar).")
    parser.add_argument("--shard_index", type=int, required=True,
                        help="Index of the shard to process, e.g., 0 for manifest_000000.jsonl.")
    args = parser.parse_args()

    manifest_file = Path(args.input_manifest_dir) / f"manifest_{args.shard_index:06d}.jsonl"
    print(f"Processing manifest: {manifest_file}")
    create_single_turn_jsonl(manifest_file, args.target_tar_dir, args.source_tar_dir, args.out_dir)


if __name__ == "__main__":
    main()
