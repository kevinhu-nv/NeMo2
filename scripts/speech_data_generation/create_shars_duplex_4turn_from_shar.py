import argparse
from collections import defaultdict
import copy
import csv
import glob
import json
import os
import random
import shutil
from tqdm import tqdm
from io import BytesIO
from pathlib import Path
import re

### from nemo.collections.tts.models import AudioCodecModel
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
from matplotlib import pyplot as plt
from tqdm import tqdm

#  python -m pdb -c continue /lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/create_shars_duplex_multi_from_single.py --manifest /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/tmp/msmarco_train_normalized.conversation_style_manifest_normalized_with_correctpath_with_evaluations.json.200 --out_shar_dir /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/tmp/msmarco_train_normalized.b.duplex.200/shars --num_shard 1


def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)


def create_shar_from_manifest(in_dir, shar_index, out_shar_dir, num_shard=10, turn_silence_sec=0.32, num_turn=2):
    cuts_name = [f"{in_dir}/cuts.{shar_index:06d}.jsonl.gz"]

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
    new_cuts = []
    prev_cut = None

    def get_audio(recording, sample_rate):
        return recording.resample(sample_rate).load_audio()

    def has_empty_text(cut_group):
        for cut in cut_group:
            for supervision in cut.supervisions:
                if supervision.text == "":
                    return True
        return False

    def create_cut_from_single_cut(cut, new_cut, total_dur):
        agent_recording = cut.target_audio
        sample_rate = agent_recording.sampling_rate
        user_recording = cut.recording
        cur_user_audio = get_audio(user_recording, sample_rate)
        # cur_user_question_audio = cut.question_recording
        user_duration = user_recording.duration + turn_silence_sec
        agent_duration = agent_recording.duration

        cur_agent_audio = agent_recording.load_audio()

        user_audio_list.extend([cur_user_audio, silence_padding, np.zeros_like(cur_agent_audio)])
        agent_audio_list.extend([np.zeros_like(cur_user_audio), silence_padding, cur_agent_audio])

        new_cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=total_dur,
                duration=user_duration,
                text=cut.supervisions[0].text,
                speaker="user",
                language="EN",
            ),
        )
        new_cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=total_dur + user_duration,
                duration=agent_duration,
                text=cut.supervisions[1].text,
                speaker="agent",
                language="EN",
            ),
        )
        total_dur += user_duration + agent_duration
        return total_dur


    print(f'Reading cuts {cuts_name}...')
    grouped_cuts = defaultdict(list)
    for cut in cuts:
        prefix = cut.id.split('_')[0]
        grouped_cuts[prefix].append(cut)
    print(f'{len(cuts)} cuts read.')

    print('Grouping cuts...')
    # Sort each group by suffix (e.g., _0, _1, _2, _3)
    for prefix in grouped_cuts:
        grouped_cuts[prefix].sort(key=lambda x: int(x.id.split('_')[1]))

    # Process each group
    new_cuts = []
    for prefix, cut_group in tqdm(grouped_cuts.items(), desc="Processing cut groups"):
        total_dur = 0
        user_audio_list = []
        agent_audio_list = []
        sample_rate = cut_group[0].target_audio.sampling_rate  # Assume all cuts in group have the same sampling rate
        new_cut = copy.deepcopy(cut_group[0])
        new_group_id = f'{shar_index}_{new_cut.id}'
        # new_group_id = new_cut.id
        new_cut.id = new_group_id
        new_cut.supervisions = []

        if has_empty_text(cut_group):
            print(f'Empty text detected in cut group {cut_group[0].id}. Skipping.')
            continue

        try:
            for cut in cut_group:
                silence_padding = np.zeros((1, int(turn_silence_sec * sample_rate)))
                total_dur = create_cut_from_single_cut(cut, new_cut, total_dur)
        except:
            print(f'Processing cut {cut_group[0]}.id failed. Skipping.')
            continue

        # Append trailing silence
        user_audio_list.append(silence_padding)
        agent_audio_list.append(silence_padding)

        user_audio = np.concatenate(user_audio_list, axis=1)
        agent_audio = np.concatenate(agent_audio_list, axis=1)

        new_cut.duration = total_dur + turn_silence_sec
        new_cut.duration_no_sil = total_dur
        new_cut.start = 0.0

        # Save concatenated audio
        user_stream = BytesIO()
        agent_stream = BytesIO()
        save_audio(dest=user_stream, src=user_audio, sampling_rate=sample_rate, format="wav")
        save_audio(dest=agent_stream, src=agent_audio, sampling_rate=sample_rate, format="wav")
        user_stream.seek(0)
        agent_stream.seek(0)
        new_cut.recording = Recording.from_bytes(user_stream.getvalue(), f"{cut_group[0].id}_user")
        new_cut.target_audio = Recording.from_bytes(agent_stream.getvalue(), f"{cut_group[0].id}_agent")
        new_cuts.append(new_cut)

    cuts = CutSet(cuts=new_cuts)
    num_total_group = len(grouped_cuts)
    shard_size = int((len(grouped_cuts)+1) / num_shard)
    if num_total_group % shard_size != 0:
        shard_size += 1
    print(f"shard_size {shard_size} num_shards {num_shard}")

    ################################
    # Output to a temporary directory
    print(f"...Making Shars {shar_index:06d}")
    out_shar_dir = Path(f"{out_shar_dir}/manifest_{shar_index:06d}")
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    # assert len(user_recordings) % shard_size != 0, "Lhotse breaks if feat_list is a multiple of shard_size"
           
    exported = cuts.to_shar(
        out_shar_dir, fields={"recording": "flac", "target_audio": "flac"}, num_jobs=1, shard_size=shard_size
    )
    print(f"...Shars created in {out_shar_dir}")

    ###################
    # Sanity check
    cuts_name = [f"{out_shar_dir}/cuts.000000.jsonl.gz"]
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
    print(f'Sanity check cuts {cuts_name} ')
    for j, cut in enumerate(cuts):
        print(f'Processing {j}/{num_cuts} cuts...')
        for i in range(8):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_dir',
        type=str,
        default="/home/kevinhu/s2s/duplex/shar/",
    )
    parser.add_argument(
        '--shar_index',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--out_shar_dir',
        type=str,
        default="/home/kevinhu/s2s/duplex/shar_duplex/",
    )
    parser.add_argument(
        '--num_shard',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--turn_silence_sec',
        type=float,
        default=0.32,
    )
    parser.add_argument(
        '--num_turn',
        type=int,
        default=2,
    )

    args = parser.parse_args()
    print(f"out_shar_dir {args.out_shar_dir}")
    print(f"num_shard {args.num_shard}")

    create_shar_from_manifest(
        in_dir=args.in_dir,
        shar_index=args.shar_index,
        out_shar_dir=args.out_shar_dir,
        num_shard=args.num_shard,
        turn_silence_sec=args.turn_silence_sec,
        num_turn=args.num_turn,
    )


if __name__ == "__main__":
    main()
