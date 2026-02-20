#!/usr/bin/env python3
"""
Transcribe Fisher lhotse shar dataset using SenseVoice.
Outputs a new lhotse shar with SenseVoice transcripts and emotion/event tags
added to each segment. Original audio is symlinked (not copied).

Usage:
    python transcribe_fisher_sensevoice.py \
        --input_dir /path/to/fisher/fisher_v2 \
        --output_dir /path/to/fisher/fisher_v2_sensevoice \
        --device cuda:0

    # Process only first 2 shards for testing:
    python transcribe_fisher_sensevoice.py \
        --input_dir /path/to/fisher/fisher_v2 \
        --output_dir /path/to/fisher/fisher_v2_sensevoice \
        --num_shards 2
"""

import argparse
import gzip
import io
import json
import logging
import os
import re
import tarfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EMOTION_TAGS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED", "EMO_UNKNOWN"}
EVENT_TAGS = {"BGM", "Speech", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough"}
MIN_SEGMENT_DURATION = 0.1  # skip segments shorter than this (seconds)
SENSEVOICE_SR = 16000


def parse_sensevoice_output(raw_text: str) -> dict:
    """
    Parse SenseVoice raw output to extract transcript, emotion, and event.

    Raw format example: '<|en|><|Speech|><|HAPPY|><|woitn|>Hello how are you'
    Returns: {"text": "Hello how are you", "emotion": "HAPPY", "event": "Speech"}
    """
    tags = re.findall(r"<\|(\w+)\|>", raw_text)
    clean_text = re.sub(r"<\|[^|]+\|>", "", raw_text).strip()

    emotion = "NEUTRAL"
    event = "Speech"
    for tag in tags:
        if tag in EMOTION_TAGS:
            emotion = tag
        elif tag in EVENT_TAGS:
            event = tag

    return {"text": clean_text, "emotion": emotion, "event": event}


def load_all_audio_from_tar(tar_path: str) -> dict:
    """Load all wav files from a tar archive. Returns {cut_id: (audio_np, sr)}."""
    audio_dict = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".wav"):
                cut_id = member.name[:-4]
                f = tar.extractfile(member)
                audio, sr = sf.read(io.BytesIO(f.read()))
                audio_dict[cut_id] = (audio, sr)
    return audio_dict


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    audio_t = torch.from_numpy(audio).float().unsqueeze(0)
    resampled = torchaudio.functional.resample(audio_t, orig_sr, target_sr)
    return resampled.squeeze(0).numpy()


def extract_result_text(res) -> str:
    """Robustly extract text from a SenseVoice result (handles different return formats)."""
    if isinstance(res, dict):
        return res.get("text", "")
    if isinstance(res, list) and len(res) > 0:
        return extract_result_text(res[0])
    return ""


def process_shard(input_dir, output_dir, shard_idx, model, batch_size_s=300, language="en"):
    """Process a single shard: transcribe segments, write new manifest, symlink audio."""
    manifest_file = f"cuts.{shard_idx}.jsonl.gz"
    out_manifest = os.path.join(output_dir, manifest_file)

    if os.path.exists(out_manifest):
        log.info(f"Shard {shard_idx}: already done, skipping")
        return

    manifest_path = os.path.join(input_dir, manifest_file)
    rec_tar = os.path.join(input_dir, f"recording.{shard_idx}.tar")
    tgt_tar = os.path.join(input_dir, f"target_audio.{shard_idx}.tar")

    # Read manifest
    cuts = []
    with gzip.open(manifest_path, "rt") as f:
        for line in f:
            if line.strip():
                cuts.append(json.loads(line))
    log.info(f"Shard {shard_idx}: {len(cuts)} cuts")

    # Load all audio from tars
    log.info(f"Shard {shard_idx}: loading audio...")
    rec_audio = load_all_audio_from_tar(rec_tar)
    tgt_audio = load_all_audio_from_tar(tgt_tar)

    # Collect all segments across all cuts for batch transcription
    # Each entry: (cut_index, segment_type, segment_index, audio_16k_numpy)
    segments = []

    for ci, cut in enumerate(cuts):
        cid = cut["id"]
        custom = cut.get("custom", {})

        for seg_type, audio_src in [("user_segments", rec_audio), ("agent_segments", tgt_audio)]:
            if seg_type not in custom:
                continue
            if cid not in audio_src:
                log.warning(f"Shard {shard_idx}: cut '{cid}' missing from {seg_type} tar")
                continue

            full_audio, sr = audio_src[cid]
            for si, seg in enumerate(custom[seg_type]):
                dur = seg["end"] - seg["start"]
                if dur < MIN_SEGMENT_DURATION:
                    continue
                s = int(seg["start"] * sr)
                e = int(seg["end"] * sr)
                seg_audio = full_audio[s:e]
                if len(seg_audio) == 0:
                    continue
                seg_audio_16k = resample(seg_audio, sr, SENSEVOICE_SR)
                segments.append((ci, seg_type, si, seg_audio_16k))

    log.info(f"Shard {shard_idx}: transcribing {len(segments)} segments...")

    if segments:
        audio_list = [s[3] for s in segments]

        # Try batch transcription; fall back to per-segment on failure
        try:
            results = model.generate(
                input=audio_list,
                language=language,
                use_itn=False,
                batch_size_s=batch_size_s,
            )
        except Exception as e:
            log.warning(f"Shard {shard_idx}: batch generate failed ({e}), falling back to per-segment")
            results = []
            for audio_arr in audio_list:
                try:
                    res = model.generate(input=audio_arr, language=language, use_itn=False)
                    results.append(res[0] if isinstance(res, list) else res)
                except Exception:
                    results.append({"text": ""})

        # Write results back into the cut manifests
        for (ci, seg_type, si, _), res in zip(segments, results):
            raw_text = extract_result_text(res)
            parsed = parse_sensevoice_output(raw_text)
            seg = cuts[ci]["custom"][seg_type][si]
            seg["sensevoice_text"] = parsed["text"]
            seg["emotion"] = parsed["emotion"]
            seg["event"] = parsed["event"]

    # Write updated manifest
    with gzip.open(out_manifest, "wt") as f:
        for cut in cuts:
            f.write(json.dumps(cut, ensure_ascii=False) + "\n")

    # Symlink audio tars (audio unchanged, no need to copy)
    for tar_name in [f"recording.{shard_idx}.tar", f"target_audio.{shard_idx}.tar"]:
        src = os.path.abspath(os.path.join(input_dir, tar_name))
        dst = os.path.join(output_dir, tar_name)
        if not os.path.exists(dst):
            os.symlink(src, dst)

    log.info(f"Shard {shard_idx}: done")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Fisher lhotse shar with SenseVoice (transcripts + emotion + event tags)"
    )
    parser.add_argument("--input_dir", required=True, help="Input lhotse shar directory")
    parser.add_argument("--output_dir", required=True, help="Output lhotse shar directory")
    parser.add_argument("--device", default="cuda:0", help="Device for SenseVoice model")
    parser.add_argument("--batch_size_s", type=int, default=300, help="Dynamic batch size in seconds")
    parser.add_argument("--language", default="en", help="Language hint (en, zh, auto, ...)")
    parser.add_argument("--model_name", default="iic/SenseVoiceSmall", help="SenseVoice model ID")
    parser.add_argument("--start_shard", type=int, default=0, help="Start from this shard index")
    parser.add_argument("--num_shards", type=int, default=None, help="Process only N shards (for testing)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover shards
    shard_indices = sorted(
        set(f.split(".")[1] for f in os.listdir(args.input_dir) if f.startswith("cuts.") and f.endswith(".jsonl.gz"))
    )
    shard_indices = [s for s in shard_indices if int(s) >= args.start_shard]
    if args.num_shards is not None:
        shard_indices = shard_indices[: args.num_shards]

    log.info(f"Processing {len(shard_indices)} shards from {args.input_dir} -> {args.output_dir}")

    # Load SenseVoice
    from funasr import AutoModel

    log.info(f"Loading SenseVoice model: {args.model_name}")
    model = AutoModel(model=args.model_name, device=args.device)

    for shard_idx in shard_indices:
        try:
            process_shard(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                shard_idx=shard_idx,
                model=model,
                batch_size_s=args.batch_size_s,
                language=args.language,
            )
        except Exception as e:
            log.error(f"Shard {shard_idx} failed: {e}", exc_info=True)

    log.info("All shards processed.")


if __name__ == "__main__":
    main()
