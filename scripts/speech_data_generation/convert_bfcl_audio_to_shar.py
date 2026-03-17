"""
Convert BFCL_v3_audio HuggingFace dataset to Lhotse shar format.

Creates shars compatible with the Glaive FC format used by duplex STT:
  - recording (user speech audio)
  - target_audio (zeros — no agent speech for FC-only evaluation)
  - supervisions: system (tools) → User (speech) → Assistant (TOOLCALL)

Usage:
    # Convert BFCL_v3_simple
    python scripts/speech_data_generation/convert_bfcl_audio_to_shar.py \
        --subset BFCL_v3_simple \
        --out_dir /path/to/output

    # Convert all subsets
    python scripts/speech_data_generation/convert_bfcl_audio_to_shar.py \
        --subset all \
        --out_dir /path/to/output
"""

import argparse
import copy
import json
import os
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment
from lhotse.audio import save_audio
from lhotse.shar.readers.lazy import LazySharIterator
from tqdm import tqdm

DATASET_REPO = "ServiceNow-AI/BFCL_v3_audio"

ALL_SUBSETS = [
    "BFCL_v3_simple",
    "BFCL_v3_multiple",
    "BFCL_v3_parallel",
    "BFCL_v3_parallel_multiple",
    "BFCL_v3_irrelevance",
]


def load_bfcl_audio_dataset(subset: str, cache_dir: str | None = None):
    """Load a BFCL_v3_audio subset from HuggingFace.

    Uses fastparquet to work around pyarrow 19 incompatibility.
    Returns a pandas DataFrame with columns: id, question, reference, tools, audio.bytes
    """
    import pandas as pd

    parquet_file = f"{subset}/test-00000-of-00001.parquet"
    local_path = hf_hub_download(
        DATASET_REPO, parquet_file, repo_type="dataset", cache_dir=cache_dir
    )
    df = pd.read_parquet(local_path, engine="fastparquet")
    print(f"Loaded {len(df)} examples from {subset}")
    return df


def decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes (FLAC/WAV) to numpy array + sampling rate."""
    buf = BytesIO(audio_bytes)
    data, sr = sf.read(buf, dtype="float32")
    if data.ndim == 1:
        data = data[np.newaxis, :]  # (1, samples) for mono
    elif data.ndim == 2:
        data = data.T  # (channels, samples)
    return data, sr


def bfcl_reference_to_toolcall_tag(reference: list[dict]) -> str:
    """Convert BFCL reference format to <TOOLCALL> tag format.

    BFCL:   [{"func": {"p": [v1, v2]}}]
    Glaive: <TOOLCALL>[{"name": "func", "arguments": {"p": v1}}]</TOOLCALL>
    """
    calls = []
    for item in reference:
        for func_name, params in item.items():
            args = {}
            for k, v_list in params.items():
                # Pick the first non-empty acceptable value
                for v in v_list:
                    if v != "":
                        args[k] = v
                        break
            calls.append({"name": func_name, "arguments": args})
    return f"<TOOLCALL>{json.dumps(calls)}</TOOLCALL>"


def tools_to_system_prompt(tools: list[dict]) -> str:
    """Wrap tool definitions in <AVAILABLE_TOOLS> tags like the Glaive format."""
    return f"<AVAILABLE_TOOLS>{json.dumps(tools)}</AVAILABLE_TOOLS>"


def convert_subset(
    df,
    subset_name: str,
    out_dir: Path,
    target_sr: int = 16000,
    turn_silence_sec: float = 0.8,
    fc_start_delay: float = 0.32,
    shard_size: int = 1024,
):
    """Convert a BFCL_v3_audio DataFrame to Lhotse shar format."""
    subset_out = out_dir / subset_name
    subset_out.mkdir(parents=True, exist_ok=True)

    temp_dir = subset_out / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    new_cuts = []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {subset_name}"):
        cut_id = str(row["id"]).replace(" ", "_")

        # Parse JSON fields
        try:
            question_list = json.loads(row["question"]) if isinstance(row["question"], str) else row["question"]
            reference = json.loads(row["reference"]) if isinstance(row["reference"], str) else row["reference"]
            tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  Skipping {cut_id}: JSON parse error: {e}")
            skipped += 1
            continue

        # Get user text from question
        if isinstance(question_list, list) and len(question_list) > 0:
            user_text = question_list[0] if isinstance(question_list[0], str) else str(question_list[0])
        else:
            user_text = str(question_list)

        # Decode audio
        audio_col = "audio.bytes" if "audio.bytes" in df.columns else "audio"
        audio_bytes = row[audio_col]
        if audio_bytes is None:
            print(f"  Skipping {cut_id}: no audio")
            skipped += 1
            continue

        # Handle nested dict from HF audio column
        if isinstance(audio_bytes, dict):
            audio_bytes = audio_bytes.get("bytes", audio_bytes.get("array", None))
            if audio_bytes is None:
                print(f"  Skipping {cut_id}: cannot extract audio bytes")
                skipped += 1
                continue

        try:
            user_audio, orig_sr = decode_audio_bytes(audio_bytes)
        except Exception as e:
            print(f"  Skipping {cut_id}: audio decode error: {e}")
            skipped += 1
            continue

        # Resample if needed
        if orig_sr != target_sr:
            import librosa
            user_audio_1d = user_audio[0] if user_audio.ndim == 2 else user_audio
            user_audio_1d = librosa.resample(user_audio_1d, orig_sr=orig_sr, target_sr=target_sr)
            user_audio = user_audio_1d[np.newaxis, :]

        # Add silence padding
        num_silence_samples = int(turn_silence_sec * target_sr)
        silence_padding = np.zeros((1, num_silence_samples), dtype=user_audio.dtype)
        user_audio_padded = np.concatenate([user_audio, silence_padding], axis=1)

        # Create agent audio (zeros, same length)
        agent_audio = np.zeros_like(user_audio_padded)

        total_duration = user_audio_padded.shape[1] / target_sr
        user_speech_duration = user_audio.shape[1] / target_sr

        # Build supervisions
        # 1. System prompt (tools)
        system_sup = SupervisionSegment(
            id=f"{cut_id}_system",
            recording_id=f"{cut_id}_user",
            start=0,
            duration=0,
            text=tools_to_system_prompt(tools),
            speaker="system",
            language="EN",
        )

        # 2. User speech
        user_sup = SupervisionSegment(
            id=f"{cut_id}_user",
            recording_id=f"{cut_id}_user",
            start=0,
            duration=user_speech_duration,
            text=user_text,
            speaker="User",
            language="EN",
            custom={"function": ""},
        )

        # 3. Assistant TOOLCALL (if reference is non-empty)
        # Place within the silence padding (not at the very end, which would be
        # out of bounds for the token channel).  Match Glaive convention: small
        # gap after user speech.
        fc_start = user_speech_duration + fc_start_delay
        toolcall_tag = bfcl_reference_to_toolcall_tag(reference) if reference else ""
        assistant_sup = SupervisionSegment(
            id=f"{cut_id}_assistant",
            recording_id=f"{cut_id}_user",
            start=fc_start,
            duration=0,
            text="",
            speaker="Assistant",
            language="EN",
            custom={"function": toolcall_tag},
        )

        supervisions = [system_sup, user_sup, assistant_sup]

        # Create recordings from bytes
        user_stream = BytesIO()
        agent_stream = BytesIO()
        save_audio(dest=user_stream, src=user_audio_padded, sampling_rate=target_sr, format="wav")
        save_audio(dest=agent_stream, src=agent_audio, sampling_rate=target_sr, format="wav")
        user_stream.seek(0)
        agent_stream.seek(0)

        user_recording = Recording.from_bytes(user_stream.getvalue(), f"{cut_id}_user")
        agent_recording = Recording.from_bytes(agent_stream.getvalue(), f"{cut_id}_agent")

        # Build cut from recording
        cut = CutSet.from_manifests(
            recordings=RecordingSet.from_recordings([user_recording])
        )[0]
        cut = copy.deepcopy(cut)
        cut.id = cut_id
        cut.start = 0.0
        cut.duration = total_duration
        cut.supervisions = supervisions
        cut.target_audio = agent_recording

        new_cuts.append(cut)

    print(f"Created {len(new_cuts)} cuts ({skipped} skipped)")

    if not new_cuts:
        print("No cuts to export!")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # Export to shar
    print("Exporting to shar format...")
    cuts = CutSet(cuts=new_cuts)
    exported = cuts.to_shar(
        str(subset_out),
        fields={"recording": "wav", "target_audio": "wav"},
        num_jobs=1,
        shard_size=shard_size,
    )
    print(f"Exported: {exported}")

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Sanity check
    print("Running sanity check on first few cuts...")
    import glob as globmod
    cut_files = sorted(globmod.glob(f"{subset_out}/cuts.*.jsonl.gz"))
    if cut_files:
        cuts_iter = LazySharIterator(
            {
                "cuts": cut_files,
                "recording": [f.replace("cuts", "recording").replace("jsonl.gz", "tar") for f in cut_files],
                "target_audio": [f.replace("cuts", "target_audio").replace("jsonl.gz", "tar") for f in cut_files],
            }
        )
        for j, cut in enumerate(cuts_iter):
            if j >= 3:
                break
            print(f"  Cut {j}: id={cut.id}, dur={cut.duration:.2f}s, "
                  f"sups={len(cut.supervisions)}, "
                  f"user_text={cut.supervisions[1].text[:60]}...")
            cut.recording.load_audio()
            cut.target_audio.load_audio()
            print(f"    Audio loaded OK")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert BFCL_v3_audio to Lhotse shar format")
    parser.add_argument(
        "--subset",
        default="BFCL_v3_simple",
        help="BFCL subset name, or 'all' for all subsets. "
             f"Choices: {', '.join(ALL_SUBSETS)}, all",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for shar files",
    )
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sampling rate")
    parser.add_argument("--turn_silence_sec", type=float, default=0.8, help="Silence padding after user speech")
    parser.add_argument("--fc_start_delay", type=float, default=0.32, help="Delay after user speech end before TOOLCALL supervision starts (must be < turn_silence_sec)")
    parser.add_argument("--shard_size", type=int, default=1024, help="Max cuts per shard")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit number of examples to convert (for debugging)")
    parser.add_argument("--cache_dir", default=None, help="HuggingFace cache directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets = ALL_SUBSETS if args.subset == "all" else [args.subset]

    for subset in subsets:
        print(f"\n{'='*60}")
        print(f"Processing: {subset}")
        print(f"{'='*60}")
        df = load_bfcl_audio_dataset(subset, cache_dir=args.cache_dir)
        if args.max_examples:
            df = df.head(args.max_examples)
            print(f"Limiting to first {args.max_examples} examples")
        convert_subset(
            df,
            subset_name=subset,
            out_dir=out_dir,
            target_sr=args.target_sr,
            turn_silence_sec=args.turn_silence_sec,
            fc_start_delay=args.fc_start_delay,
            shard_size=args.shard_size,
        )


if __name__ == "__main__":
    main()
