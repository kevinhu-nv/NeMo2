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

import librosa
import numpy as np
import soundfile as sf
from lhotse import AudioSource, CutSet, Recording, SupervisionSegment, MonoCut
from lhotse.array import Array, TemporalArray
from lhotse.audio import RecordingSet, save_audio
from lhotse.cut.base import Cut
from lhotse.features.base import Features, FeatureSet
from lhotse.shar.readers.lazy import LazySharIterator
from lhotse.shar.writers import AudioTarWriter
from matplotlib import pyplot as plt
from tqdm import tqdm

from nemo.utils import logging

import os
import glob
import json

import re
import os
import json
import concurrent.futures
from tqdm import tqdm
import argparse
import logging
from collections import defaultdict
import jiwer
import string as strfrm
from jiwer.transforms import RemoveKaldiNonWords
from nltk.tokenize import SyllableTokenizer

from lhotse import CutSet
from lhotse.supervision import AlignmentItem, SupervisionSet, SupervisionSegment
from lhotse.cut import MonoCut

from cytoolz import groupby

def make_sups_from_seglines(uniq_id, seglst_lines, round_digits=3):
    sups, rttm_lines = [], []
    for idx, line in enumerate(seglst_lines):
        spk, start, end = line['speaker'], float(line['start']), float(line['end'])
        words = line['text']
        dur = round(end - start, round_digits)
        rttm_line_str = f'SPEAKER {uniq_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>'
        rttm_lines.append(rttm_line_str)
        if len(words.strip()) == 0:
            sups.append(SupervisionSegment(id=f'sup{idx}', recording_id=uniq_id, start=start, duration=dur, channel=0, text="<unk>", speaker=spk))
        elif len(words.strip()) > 0:
            sups.append(SupervisionSegment(id=f'sup{idx}', recording_id=uniq_id, start=start, duration=dur, channel=0, text=words, speaker=spk))
    return sups, rttm_lines



def read_seglst(uniq_id, seglst_filepath: str, round_digits=3):
    """
    Read the seglst file and return the seglst info
    Args:
        seglst_filepath: path to the seglst file
        seglst format: 
        [
            {
                "session_id": "Bed008",
                "words": "alright so i'm i should read all of these numbers",
                "speaker": "me045",
                "start_time": "53.814",
                "end_time": "56.753"
            }
        ]
    Returns:
        cut: lhotse MonoCut
    """
    with open(seglst_filepath, 'r') as f:
        seglst_lines = json.loads(f.read())
    sups, rttm_lines = make_sups_from_seglines(uniq_id, seglst_lines)
    sups = SupervisionSet.from_segments(sups)
    return sups, rttm_lines

def group_segments(
        args,
        recording_id: str,
        sups: SupervisionSet,
        round_digits: int = 3
    ):

    sid = 0
    cuts = []
    
    while sid < len(sups) - 1:
        min_length = random.randint(args.min_length, args.max_length)
        use_seg_hop = True
        cut_sups = []
        text = sups[sid].text
        
        # skip case 1: the overlap duration at the beginning is larger than the threshold
        if sid and sups[sid-1].end - sups[sid].start > args.edge_ovl_thres: # checking overlap duration at the beginning
            sid += 1
            continue
        # skip case 2: the first segment is too short, will also remove that later in _process function
        words = text.split()
        if len(words) <= args.leading_word_thres and sups[sid].speaker != sups[sid+1].speaker:
            sid += 1
            continue

        # skip case 3: the start time of the first segment is exactly the same as that of previous segment
        if sid and sups[sid-1].start == sups[sid].start:
            sid += 1
            continue
        
        starts, ends = [], []
        for eid in range(sid, len(sups)):
            if text: 
                starts.append(sups[eid].start)
                ends.append(sups[eid].end)
                cut_sups.append(sups[eid])
            else: # if words is empty, this means that there is a mismatch between rttm and ctm, and we will skip this segment
                break

            # large gap between two segments
            if eid < len(sups) - 1 and sups[eid+1].start - max(ends) > args.min_gap:                
                break
            
            if max(ends) - min(starts) > min_length: 

                # check if all segments are included, this is the rare case but will cause the disagreement between rttm and lhotse
                for eid_sub in range(eid+1, len(sups)):
                    if sups[eid_sub].end <= max(ends):
                        starts.append(sups[eid_sub].start)
                        ends.append(sups[eid_sub].end)
                        cut_sups.append(sups[eid_sub])
                        eid = eid_sub
                        use_seg_hop = False
                    elif sups[eid_sub].start >= max(ends):
                        break

                    if max(ends) - min(starts) > args.max_length:
                        break

                if not use_seg_hop:
                    sid = eid_sub
                break
                
        # update the start index of the next segment
        if use_seg_hop:
            if args.seg_hop == 0: 
                sid = eid + 1
            elif args.seg_hop > 0:
                sid = sid + args.seg_hop
            else:
                raise ValueError(f"args.seg_hop should be greater than or equal to 0 but got {args.seg_hop}")
            
        # skip case 4: the overlap duration at the end is larger than the threshold
        if eid < len(sups) - 1 and max(ends) - sups[eid+1].start > args.edge_ovl_thres: # checking overlap duration at the end
            continue

        # skip case 5: two segments start at the same time
        if len(starts) > 1 and starts[0] == starts[1]:
            continue

        # skip case 6: too short or too long segment
        start, end = min(starts), max(ends)
        dur=round(end - start, round_digits)
        if dur < args.min_length or dur > args.max_length:
            continue

        cut_sups = SupervisionSet.from_segments(cut_sups)
        cut = MonoCut(
            id=f'{recording_id}-{int(start*10**round_digits):09d}-{int(end*10**round_digits):09d}',
            start=start,
            duration=dur,
            channel=0,
            supervisions=cut_sups
        )
        cuts.append(cut)

    return CutSet.from_cuts(cuts)

def remove_leading_short_segment(args, cut: MonoCut, round_digits: int=3):
    """
    Remove the leading short segment
    Args:
        cut: MonoCut
    """
    if len(cut.supervisions[0].text.split()) > args.leading_word_thres:
        return cut

    words, spks = [], []
    change_idx = 0
    for i in range(len(cut.supervisions)):
        sup = cut.supervisions[i]
        seg_words = sup.text.split()
        if spks and sup.speaker != spks[-1]:
            change_idx = i
            break
        words.extend(seg_words)
        spks.append(sup.speaker)

    # change the start of the segment
    if change_idx and len(words) <= args.leading_word_thres:
        if cut.supervisions[change_idx].start == cut.supervisions[change_idx-1].start:
            return None # for rare cases
        new_start = cut.supervisions[change_idx].start
        new_duration = round(cut.duration - (new_start - cut.start), round_digits)
        cut.start = new_start
        cut.duration = new_duration
        cut.supervisions = SupervisionSet.from_segments(cut.supervisions[change_idx:])
    return cut

def write_rttm_file(rttm_filepath, rttm_lines):
    with open(rttm_filepath, 'w') as f:
        for rline in rttm_lines:
            f.write(rline + '\n')


def _process(args, uniq_id, seglst_lines):  
    """
    Process one audio file with the corresponding rttm and force alignment files
    
    Args:
        json_dict: {
            audio_filepath: path to the audio file
            rttm_filepath: path to the rttm file
            ctm_filepath: path to the ctm file
            seglst_filepath: path to the seglst file
        }
    """
    sups, revised_rttm_lines = make_sups_from_seglines(uniq_id, seglst_lines, round_digits=3)
    
    sups = sorted(sups, key=lambda x: (x.start, x.duration, x.text))
    cuts = group_segments(args, uniq_id, sups)

    # generate manifests
    return cuts

def get_audio(recording, sample_rate):
    return recording.resample(sample_rate).load_audio()

def get_step(dur, osample_rate):
    return int(dur*osample_rate)

def parse_args(): 
    parser = argparse.ArgumentParser()
    # arguments for generating cuts
    parser.add_argument(
        "--dataset_id", default='davidai', help="The name of dataset", type=str
    )
    parser.add_argument(
        "--length", default=[120, 180], type=float, nargs='+', help="the approximate length of the segment"
    )
    parser.add_argument(
        "--seg_hop", default=0, type=int, help="The number of single-speaker segments hopping between two consecutive grouped segments"
    ) 
    parser.add_argument(
        "--min_gap", default=100, type=int, help="The minimum gap between two segments"
    )
    parser.add_argument(
        "--max_speakers", default=2, type=int, help="The maximum number of speakers"
    )
    parser.add_argument(
        "--word_merge_thres", default=-1.0, type=float, help="The threshold to control the length of the word gaps for the same speaker"
    )
    parser.add_argument(
        "--edge_ovl_thres", default=0, type=float, help="The threshold to control the overlap duration at the beginning and the end"
    )
    parser.add_argument(
        "--leading_word_thres", default=3, type=int, help="The threshold to control the number of leading words"
    )
    # New arguments
    parser.add_argument(
        "--dirs_file", required=True, help="File containing list of directories to process"
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for shards"
    )
    parser.add_argument(
        "--num_shards", type=int, default=6, help="Number of shards to create"
    )
    parser.add_argument(
        "--user_sr", type=int, default=16000, help="Sample rate for user audio"
    )
    parser.add_argument(
        "--agent_sr", type=int, default=22050, help="Sample rate for agent audio"
    )
    
    args = parser.parse_args()
    args.min_length, args.max_length = args.length[0], args.length[1]
    return args

def create_shar_cuts(
    args,
    all_seglst_path,
    num_shard,
    out_dir=None
):
    """
    Args:
        manifest: List[Dict] [{"audio_filepath": ..., 'duration': ... }, ...]
        segments: List[List[[st, et, spk]]] [[[start_time0, end_time0, 'speaker_0'], [start_time1, end_time1, 'speaker_1'], ...]]
        indice: List[int] [1, 3, 4, 6, ...]
        shard_size: int
        out_dir: str
    """
    training_cuts = []
    for i in tqdm(range(len(all_seglst_path))):
        seglst_path = all_seglst_path[i]
        with open(seglst_path, 'r') as f:
            seglst_lines = json.load(f)['transcript']
        cuts = _process(args, seglst_path.split('/')[-2], seglst_lines)

        for i_cut, cut in enumerate(cuts):
            
            cut_id = cut.id + f"-{i_cut}"
            # Step 1: supervision
            supervisions = []

            audio_filepath1 = seglst_path.replace('transcription.json', 'speaker1.wav')
            audio_filepath2 = seglst_path.replace('transcription.json', 'speaker2.wav')

            sample_rate = sf.info(audio_filepath1).samplerate
            num_samples1 = sf.info(audio_filepath1).frames
            num_samples2 = sf.info(audio_filepath2).frames
            duration1 = sf.info(audio_filepath1).duration
            duration2 = sf.info(audio_filepath2).duration
            
            # Step 2: recording
            recording1 = Recording(
                id=cut_id,
                sources=[
                    AudioSource(
                        type='file',
                        channels=[0],
                        source=audio_filepath1
                    )
                ],
                sampling_rate=sample_rate,
                num_samples=num_samples1,
                duration=duration1,
                channel_ids=[0]
            )
            recording2 = Recording(
                id=cut_id,
                sources=[
                    AudioSource(
                        type='file',
                        channels=[0],
                        source=audio_filepath2
                    )
                ],
                sampling_rate=sample_rate,
                num_samples=num_samples2,
                duration=duration2,
                channel_ids=[0]
            )
            user_sr, agent_sr = args.user_sr, args.agent_sr
            original_audio1=get_audio(recording1, sample_rate=user_sr)
            original_audio2=get_audio(recording2, sample_rate=agent_sr)
            
            user_start, user_end = cut.start, min(cut.start + cut.duration, duration1)
            user_start_idx, user_end_idx = get_step(user_start, osample_rate=user_sr), get_step(user_end, osample_rate=user_sr)
            user_audio=copy.deepcopy(original_audio1[0, user_start_idx:user_end_idx])

            agent_start, agent_end = cut.start, min(cut.start + cut.duration, duration2)
            agent_start_idx, agent_end_idx = get_step(agent_start, osample_rate=agent_sr), get_step(agent_end, osample_rate=agent_sr)
            agent_audio=copy.deepcopy(original_audio2[0, agent_start_idx:agent_end_idx])

            user_stream = BytesIO()
            agent_stream = BytesIO()
            save_audio(dest=user_stream, src=user_audio, sampling_rate=user_sr, format="wav")
            save_audio(dest=agent_stream, src=agent_audio, sampling_rate=agent_sr, format="wav")
            user_stream.seek(0)
            agent_stream.seek(0)
            recording = Recording.from_bytes(user_stream.getvalue(), f"{cut_id}_user")
            target_audio = Recording.from_bytes(agent_stream.getvalue(), f"{cut_id}_agent")

            # Step 3: custom
            custom = {
                "user_segments": [],
                "agent_segments": []
            }
            for sup in cut.supervisions:
                if sup.speaker == 1:
                    custom["user_segments"].append({
                        "speaker": sup.speaker,
                        "start": sup.start - cut.start,
                        "end": min(sup.start + sup.duration, duration1) - cut.start,
                        "text": sup.text
                    })     
                else:
                    custom["agent_segments"].append({
                        "speaker": sup.speaker,
                        "start": sup.start - cut.start,
                        "end": min(sup.start + sup.duration, duration2) - cut.start,
                        "text": sup.text
                    })
            if len(custom["user_segments"]) == 0 or len(custom["agent_segments"]) == 0:
                continue

            # Done
            training_cut_duration = cut.duration
            if training_cut_duration + cut.start > min(duration1, duration2):
                training_cut_duration = min(duration1, duration2) - cut.start
            training_cut = MonoCut(
                id=cut_id,
                start=0,
                duration=training_cut_duration,
                channel=0,
                supervisions=supervisions,
                recording=recording,
                custom=custom
            )
            training_cut.target_audio=target_audio
            training_cuts.append(training_cut)
        
    shard_size = max(int(len(training_cuts) / num_shard), 1)
    if len(training_cuts) % shard_size != 0:
        shard_size += 1
    print(f"shard_size {shard_size} num_shards {num_shard}")

    os.makedirs(out_dir, exist_ok=True)
    training_cuts = CutSet(cuts=training_cuts)

    exported = training_cuts.to_shar(
        out_dir, fields={"recording": "wav", "target_audio": "wav"}, num_jobs=1, shard_size=shard_size
    )
    # exported = training_cuts.to_shar(
    #     out_dir, fields={}, num_jobs=1, shard_size=shard_size
    # )

def main(args):
    # Read directories from the provided file
    with open(args.dirs_file, 'r') as f:
        dirs = [line.strip() for line in f.readlines()]

    # Get all transcription.json files from the provided directories
    all_seglst_path = []
    for dir_path in dirs:
        trans_path = os.path.join(dir_path, 'transcription.json')
        if os.path.exists(trans_path):
            all_seglst_path.append(trans_path)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Process the data
    create_shar_cuts(
        args=args,
        all_seglst_path=all_seglst_path,
        num_shard=args.num_shards,
        out_dir=args.out_dir
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
