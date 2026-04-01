from __future__ import annotations
"""Convert VoiceAgentBench (VAB) HuggingFace dataset to Lhotse shar format.

Creates shars compatible with the duplex STT FC format:
  - recording (user speech audio)
  - target_audio (zeros — no agent speech for FC-only evaluation)
  - supervisions: system (tools) → User (speech) → Assistant (TOOLCALL)

For multi_turn: multiple User/Assistant turns with stitched user audio,
  final Assistant turn contains the TOOLCALL.

For safety: User query only, no TOOLCALL (model should refuse).

Usage:
    # Convert single_tool English
    python scripts/speech_data_generation/convert_vab_audio_to_shar.py \
        --subset single_tool \
        --out_dir /path/to/output

    # Convert all subsets
    python scripts/speech_data_generation/convert_vab_audio_to_shar.py \
        --subset all \
        --out_dir /path/to/output

    # Debug: convert 3 examples and dump supervision details
    python scripts/speech_data_generation/convert_vab_audio_to_shar.py \
        --subset single_tool \
        --out_dir /path/to/output \
        --max_examples 3 \
        --debug
"""

import argparse
import copy
import json
import os
import re
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

DATASET_REPO = "krutrim-ai-labs/VoiceAgentBench"

ALL_SUBSETS = [
    "single_tool",
    "single_tool_retrieval",
    "parallel_tool",
    "seqdep_tool",
    "multi_turn",
    "safety",
]

# JSON file paths relative to repo root, per subset
SUBSET_JSON_PATHS = {
    "single_tool": "data/single_tool_data/english/single_tool_english.json",
    "single_tool_retrieval": "data/single_tool_retrieval_data/english/single_tool_retrieval_english.json",
    "parallel_tool": "data/parallel_tool_data/english/parallel_tool_english.json",
    "seqdep_tool": "data/seqdep_tool_data/english/seqdep_tool_english.json",
    "multi_turn": "data/multi_turn_data/english/multi_turn_english.json",
    "safety": "data/safety_data/english/safety_english.json",
}


def load_vab_json(subset: str, cache_dir: str | None = None) -> list[dict]:
    """Download and load a VAB subset JSON from HuggingFace."""
    json_filename = SUBSET_JSON_PATHS[subset]
    local_path = hf_hub_download(
        DATASET_REPO, json_filename, repo_type="dataset", cache_dir=cache_dir
    )
    with open(local_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {subset}")
    return data


def download_audio(audio_path: str, cache_dir: str | None = None) -> str:
    """Download a single audio file from HuggingFace and return local path."""
    return hf_hub_download(
        DATASET_REPO, audio_path, repo_type="dataset", cache_dir=cache_dir
    )


def load_audio_file(local_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample if needed. Returns (1, samples) array."""
    data, sr = sf.read(local_path, dtype="float32")
    if data.ndim == 2:
        data = data[:, 0]  # take first channel
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data[np.newaxis, :]  # (1, samples)


def vab_expected_to_toolcall_tag(expected_tool_call: list[dict]) -> str:
    """Convert VAB expected_tool_call format to <TOOLCALL> tag format.

    VAB:    [{"func_name": {"param": ["value1", "value2"]}}]
    Glaive: <TOOLCALL>[{"name": "func_name", "arguments": {"param": "value1"}}]</TOOLCALL>
    """
    calls = []
    for item in expected_tool_call:
        for func_name, params in item.items():
            args = {}
            for k, v_list in params.items():
                if isinstance(v_list, list):
                    # Pick the first non-empty acceptable value
                    for v in v_list:
                        if v != "":
                            args[k] = v
                            break
                else:
                    args[k] = v_list
            calls.append({"name": func_name, "arguments": args})
    return f"<TOOLCALL>{json.dumps(calls)}</TOOLCALL>"


def tools_to_system_prompt(functions: list[dict]) -> str:
    """Wrap tool definitions in <AVAILABLE_TOOLS> tags."""
    return f"<AVAILABLE_TOOLS>{json.dumps(functions)}</AVAILABLE_TOOLS>"


def parse_api_request(content: str) -> tuple[str, str, str] | None:
    """Parse an inline API-Request from a VAB chat message.

    Format: ... API-Request: [FuncName(key='val', ...)]->response_text

    Returns (text_before, toolcall_tag, tool_response) or None if no API-Request found.
    """
    match = re.search(r'API-Request:\s*\[(\w+)\((.*?)\)\]\s*->\s*(.*)', content, re.DOTALL)
    if not match:
        return None

    text_before = content[:match.start()].strip()
    func_name = match.group(1)
    args_str = match.group(2)
    response_str = match.group(3).strip()

    # Parse keyword arguments: key='value' or key="value" or key=value
    arguments = {}
    # Match key=value pairs — values can be quoted strings or unquoted
    for arg_match in re.finditer(r"(\w+)\s*=\s*('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|[^,)]+)", args_str):
        key = arg_match.group(1)
        val = arg_match.group(2).strip()
        # Strip surrounding quotes
        if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
            val = val[1:-1]
        arguments[key] = val

    toolcall_tag = f"<TOOLCALL>{json.dumps([{'name': func_name, 'arguments': arguments}])}</TOOLCALL>"
    tool_response = f"<TOOL_RESPONSE>{response_str}</TOOL_RESPONSE>"
    return text_before, toolcall_tag, tool_response


def safety_tools_to_system_prompt(tool_names: list[str]) -> str:
    """For safety subset: tool names are just strings, wrap them."""
    return f"<AVAILABLE_TOOLS>{json.dumps(tool_names)}</AVAILABLE_TOOLS>"


def convert_single_turn(
    entry: dict,
    subset_name: str,
    cache_dir: str | None,
    target_sr: int,
    speech_tail_sec: float,
    fc_start_delay: float,
    debug: bool = False,
) -> "Cut | None":
    """Convert a single-turn VAB entry to a Lhotse cut."""
    cut_id = str(entry["id"]).replace(" ", "_")

    # Get user text
    user_text = entry.get("query") or entry.get("user_request") or ""

    # Get audio
    audio_path = entry.get("path")
    if not audio_path:
        print(f"  Skipping {cut_id}: no audio path")
        return None

    try:
        local_audio = download_audio(audio_path, cache_dir)
        user_audio = load_audio_file(local_audio, target_sr)
    except Exception as e:
        print(f"  Skipping {cut_id}: audio error: {e}")
        return None

    # Build system prompt
    functions = entry.get("functions", [])
    is_safety = subset_name == "safety"
    if is_safety and functions and isinstance(functions[0], str):
        system_text = safety_tools_to_system_prompt(functions)
    else:
        system_text = tools_to_system_prompt(functions)

    # Build TOOLCALL tag (not for safety — model should refuse)
    expected = entry.get("expected_tool_call")
    toolcall_tag = ""
    if expected and not is_safety:
        toolcall_tag = vab_expected_to_toolcall_tag(expected)

    # Add silence padding after user speech for ASR to finish decoding
    num_silence_samples = int(speech_tail_sec * target_sr)
    silence = np.zeros((1, num_silence_samples), dtype=user_audio.dtype)
    user_audio_padded = np.concatenate([user_audio, silence], axis=1)
    agent_audio = np.zeros_like(user_audio_padded)

    total_duration = user_audio_padded.shape[1] / target_sr
    user_speech_duration = user_audio.shape[1] / target_sr

    # Build supervisions
    system_sup = SupervisionSegment(
        id=f"{cut_id}_system",
        recording_id=f"{cut_id}_user",
        start=0, duration=0,
        text=system_text,
        speaker="system",
        language="EN",
    )

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

    supervisions = [system_sup, user_sup]

    if toolcall_tag:
        fc_start = user_speech_duration + fc_start_delay
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
        supervisions.append(assistant_sup)

    if debug:
        print(f"  [{cut_id}] dur={total_duration:.2f}s, user_text='{user_text[:80]}...'")
        print(f"    system_prompt={system_text[:100]}...")
        if toolcall_tag:
            print(f"    toolcall={toolcall_tag[:120]}...")
        print(f"    supervisions: {len(supervisions)} turns")

    # Create recordings
    user_stream = BytesIO()
    agent_stream = BytesIO()
    save_audio(dest=user_stream, src=user_audio_padded, sampling_rate=target_sr, format="wav")
    save_audio(dest=agent_stream, src=agent_audio, sampling_rate=target_sr, format="wav")
    user_stream.seek(0)
    agent_stream.seek(0)

    user_recording = Recording.from_bytes(user_stream.getvalue(), f"{cut_id}_user")
    agent_recording = Recording.from_bytes(agent_stream.getvalue(), f"{cut_id}_agent")

    cut = CutSet.from_manifests(
        recordings=RecordingSet.from_recordings([user_recording])
    )[0]
    cut = copy.deepcopy(cut)
    cut.id = cut_id
    cut.start = 0.0
    cut.duration = total_duration
    cut.supervisions = supervisions
    cut.target_audio = agent_recording

    return cut


def convert_multi_turn(
    entry: dict,
    cache_dir: str | None,
    target_sr: int,
    speech_tail_sec: float,
    fc_start_delay: float,
    debug: bool = False,
) -> "Cut | None":
    """Convert a multi-turn VAB entry to a Lhotse cut.

    Stitches all user audio clips together with silence gaps.
    Creates supervisions for each user/assistant turn.
    Final assistant turn gets the TOOLCALL.

    Silence strategy:
      - Between user speech and next agent/toolcall turn: fc_start_delay (0.32s)
      - After the last user speech: speech_tail_sec (1.2s) for ASR decoding
    """
    cut_id = str(entry["id"]).replace(" ", "_")
    chat_history = entry.get("chat_history", [])
    if not chat_history:
        print(f"  Skipping {cut_id}: empty chat_history")
        return None

    functions = entry.get("functions", [])
    system_text = tools_to_system_prompt(functions)
    expected = entry.get("expected_tool_call")
    toolcall_tag = vab_expected_to_toolcall_tag(expected) if expected else ""

    # Stitch user audio and build supervisions
    audio_segments = []
    supervisions = []
    current_time = 0.0

    # System prompt
    system_sup = SupervisionSegment(
        id=f"{cut_id}_system",
        recording_id=f"{cut_id}_user",
        start=0, duration=0,
        text=system_text,
        speaker="system",
        language="EN",
    )
    supervisions.append(system_sup)

    # Find the index of the last user turn (needs speech_tail_sec padding)
    last_user_idx = max(
        (i for i, m in enumerate(chat_history) if m["role"] == "user"),
        default=-1,
    )

    turn_idx = 0
    sub_idx = 0  # sub-index for splitting messages with inline API calls

    for msg_idx, msg in enumerate(chat_history):
        role = msg["role"]
        content = msg.get("content", "")

        # Check for inline API-Request in this message
        parsed = parse_api_request(content) if "API-Request:" in content else None

        if role == "user":
            audio_path = msg.get("path")
            if not audio_path:
                print(f"  Skipping {cut_id}: user turn missing audio path")
                return None
            try:
                local_audio = download_audio(audio_path, cache_dir)
                user_audio = load_audio_file(local_audio, target_sr)
            except Exception as e:
                print(f"  Skipping {cut_id}: audio error on turn {turn_idx}: {e}")
                return None

            user_dur = user_audio.shape[1] / target_sr
            audio_segments.append(user_audio)

            # If user message has inline API-Request, use only the text before it
            user_text = parsed[0] if parsed else content
            user_sup = SupervisionSegment(
                id=f"{cut_id}_user_{turn_idx}",
                recording_id=f"{cut_id}_user",
                start=current_time,
                duration=user_dur,
                text=user_text,
                speaker="User",
                language="EN",
                custom={"function": ""},
            )
            supervisions.append(user_sup)
            current_time += user_dur

            # Last user turn gets full speech_tail_sec padding for ASR;
            # intermediate user turns get fc_start_delay gap only.
            if msg_idx == last_user_idx:
                gap_sec = speech_tail_sec
            else:
                gap_sec = fc_start_delay
            silence_samples = int(gap_sec * target_sr)
            audio_segments.append(np.zeros((1, silence_samples), dtype=np.float32))
            current_time += gap_sec

            # If user msg had an inline API call, emit TOOLCALL + TOOL_RESPONSE
            if parsed:
                _, toolcall_str, tool_resp_str = parsed
                tc_sup = SupervisionSegment(
                    id=f"{cut_id}_assistant_{turn_idx}_tc{sub_idx}",
                    recording_id=f"{cut_id}_user",
                    start=current_time,
                    duration=0,
                    text="",
                    speaker="Assistant",
                    language="EN",
                    custom={"function": toolcall_str},
                )
                supervisions.append(tc_sup)
                tr_sup = SupervisionSegment(
                    id=f"{cut_id}_toolresp_{turn_idx}_tc{sub_idx}",
                    recording_id=f"{cut_id}_user",
                    start=current_time,
                    duration=0,
                    text=tool_resp_str,
                    speaker="User",
                    language="EN",
                    custom={"function": tool_resp_str},
                )
                supervisions.append(tr_sup)
                sub_idx += 1

        elif role == "assistant":
            if parsed:
                text_before, toolcall_str, tool_resp_str = parsed
                # Emit text before the API call as an Assistant text supervision
                if text_before:
                    text_sup = SupervisionSegment(
                        id=f"{cut_id}_assistant_{turn_idx}",
                        recording_id=f"{cut_id}_user",
                        start=current_time,
                        duration=0,
                        text=text_before,
                        speaker="Assistant",
                        language="EN",
                        custom={"function": ""},
                    )
                    supervisions.append(text_sup)
                # Emit TOOLCALL supervision
                tc_sup = SupervisionSegment(
                    id=f"{cut_id}_assistant_{turn_idx}_tc{sub_idx}",
                    recording_id=f"{cut_id}_user",
                    start=current_time,
                    duration=0,
                    text="",
                    speaker="Assistant",
                    language="EN",
                    custom={"function": toolcall_str},
                )
                supervisions.append(tc_sup)
                # Emit TOOL_RESPONSE supervision
                tr_sup = SupervisionSegment(
                    id=f"{cut_id}_toolresp_{turn_idx}_tc{sub_idx}",
                    recording_id=f"{cut_id}_user",
                    start=current_time,
                    duration=0,
                    text=tool_resp_str,
                    speaker="User",
                    language="EN",
                    custom={"function": tool_resp_str},
                )
                supervisions.append(tr_sup)
                sub_idx += 1
            else:
                # Plain assistant text turn (no API call)
                assistant_sup = SupervisionSegment(
                    id=f"{cut_id}_assistant_{turn_idx}",
                    recording_id=f"{cut_id}_user",
                    start=current_time,
                    duration=0,
                    text=content,
                    speaker="Assistant",
                    language="EN",
                    custom={"function": ""},
                )
                supervisions.append(assistant_sup)

        turn_idx += 1

    # Final TOOLCALL assistant turn
    # The last user turn already has speech_tail_sec of silence appended.
    # Place TOOLCALL at fc_start_delay after that last user speech ended
    # (i.e. within the already-appended silence).
    if toolcall_tag:
        # current_time = last_user_speech_end + speech_tail_sec
        # We want fc_start = last_user_speech_end + fc_start_delay
        # which is current_time - speech_tail_sec + fc_start_delay
        fc_start = current_time - speech_tail_sec + fc_start_delay

        toolcall_sup = SupervisionSegment(
            id=f"{cut_id}_toolcall",
            recording_id=f"{cut_id}_user",
            start=fc_start,
            duration=0,
            text="",
            speaker="Assistant",
            language="EN",
            custom={"function": toolcall_tag},
        )
        supervisions.append(toolcall_sup)

    # Concatenate all audio
    user_audio_full = np.concatenate(audio_segments, axis=1)
    agent_audio = np.zeros_like(user_audio_full)
    total_duration = user_audio_full.shape[1] / target_sr

    if debug:
        print(f"  [{cut_id}] dur={total_duration:.2f}s, {len(chat_history)} chat turns, "
              f"{len(supervisions)} supervisions")
        for sup in supervisions:
            tag = f"  {sup.speaker:10s} t={sup.start:.2f}s"
            text_preview = (sup.text or "")[:60]
            func = (sup.custom or {}).get("function", "")[:60]
            print(f"    {tag} text='{text_preview}' func='{func}'")

    # Create recordings
    user_stream = BytesIO()
    agent_stream = BytesIO()
    save_audio(dest=user_stream, src=user_audio_full, sampling_rate=target_sr, format="wav")
    save_audio(dest=agent_stream, src=agent_audio, sampling_rate=target_sr, format="wav")
    user_stream.seek(0)
    agent_stream.seek(0)

    user_recording = Recording.from_bytes(user_stream.getvalue(), f"{cut_id}_user")
    agent_recording = Recording.from_bytes(agent_stream.getvalue(), f"{cut_id}_agent")

    cut = CutSet.from_manifests(
        recordings=RecordingSet.from_recordings([user_recording])
    )[0]
    cut = copy.deepcopy(cut)
    cut.id = cut_id
    cut.start = 0.0
    cut.duration = total_duration
    cut.supervisions = supervisions
    cut.target_audio = agent_recording

    return cut


def convert_subset(
    data: list[dict],
    subset_name: str,
    out_dir: Path,
    cache_dir: str | None = None,
    target_sr: int = 16000,
    speech_tail_sec: float = 1.2,
    fc_start_delay: float = 0.32,
    shard_size: int = 1024,
    debug: bool = False,
):
    """Convert a VAB subset to Lhotse shar format."""
    subset_out = out_dir / subset_name
    subset_out.mkdir(parents=True, exist_ok=True)

    new_cuts = []
    skipped = 0
    is_multi_turn = subset_name == "multi_turn"

    for entry in tqdm(data, desc=f"Converting {subset_name}"):
        if is_multi_turn:
            cut = convert_multi_turn(
                entry, cache_dir, target_sr, speech_tail_sec, fc_start_delay, debug
            )
        else:
            cut = convert_single_turn(
                entry, subset_name, cache_dir, target_sr,
                speech_tail_sec, fc_start_delay, debug
            )

        if cut is None:
            skipped += 1
        else:
            new_cuts.append(cut)

    print(f"Created {len(new_cuts)} cuts ({skipped} skipped)")

    if not new_cuts:
        print("No cuts to export!")
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
                  f"sups={len(cut.supervisions)}")
            for sup in cut.supervisions:
                text = sup.text[:60] if sup.text else ""
                func = ""
                if sup.custom and sup.custom.get("function"):
                    func = sup.custom["function"][:60]
                print(f"    {sup.speaker:10s} t={sup.start:.2f}s text='{text}' func='{func}'")
            cut.recording.load_audio()
            cut.target_audio.load_audio()
            print(f"    Audio loaded OK")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VoiceAgentBench dataset to Lhotse shar format"
    )
    parser.add_argument(
        "--subset",
        default="single_tool",
        help=f"VAB subset name, or 'all' for all subsets. Choices: {', '.join(ALL_SUBSETS)}, all",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for shar files")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sampling rate")
    parser.add_argument("--speech_tail_sec", type=float, default=1.2,
                        help="Silence appended after user speech for ASR decoding headroom")
    parser.add_argument("--fc_start_delay", type=float, default=0.32,
                        help="Gap between user speech end and TOOLCALL/agent supervision")
    parser.add_argument("--shard_size", type=int, default=1024, help="Max cuts per shard")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of examples (for debugging)")
    parser.add_argument("--cache_dir", default=None, help="HuggingFace cache directory")
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed supervision info per cut")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets = ALL_SUBSETS if args.subset == "all" else [args.subset]

    for subset in subsets:
        print(f"\n{'='*60}")
        print(f"Processing: {subset}")
        print(f"{'='*60}")
        data = load_vab_json(subset, cache_dir=args.cache_dir)
        if args.max_examples:
            data = data[:args.max_examples]
            print(f"Limiting to first {args.max_examples} examples")
        convert_subset(
            data,
            subset_name=subset,
            out_dir=out_dir,
            cache_dir=args.cache_dir,
            target_sr=args.target_sr,
            speech_tail_sec=args.speech_tail_sec,
            fc_start_delay=args.fc_start_delay,
            shard_size=args.shard_size,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
