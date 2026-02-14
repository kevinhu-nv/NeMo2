#!/usr/bin/env python3
"""
Reorganize candor outputs to mirror original dataset layout.

Usage:
  python reorganize_candor_outputs.py [--output_path PATH] [--original_data_path PATH] [--revised_output_path PATH] [--dataset_name NAME] [--require_dataset_in_filename] [--strict_ids] [--clean_destination]

Example:
  python reorganize_candor_outputs.py --output_path a/b/c --revised_output_path output/dir

Example mapping:
  candor_candor_turn_taking_0001.wav  ->  v1.0/candor_turn_taking/1/output.wav

Notes:
  - The script copies (not moves) files from the output path into the revised output path.
  - It extracts the numeric id from filenames like '*_0001.wav' and uses it as the directory name.
  - It derives the 'vX.Y/<dataset_name>' subpath from the provided original data path.
  - For each numeric id, it also copies all contents from the corresponding original item directory into the destination (e.g., pause.json, transcription.json, input.wav). The script never overwrites the generated 'output.wav'.
  - If --require_dataset_in_filename is set, only WAVs whose filenames contain the dataset name are processed.
  - If --strict_ids is set, only process IDs that have a matching directory in the original dataset.
  - If --clean_destination is set, the destination dataset subpath is removed before copying to avoid stale IDs.
  - All arguments are optional and have default values.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reorganize candor outputs to mirror original dataset layout")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("/lustre/fsw/portfolios/llmservice/users/vtrinh/projects/s2s_ear_tts_oct_29/result/inferences/full_duplex_bench/validation_logs/agent"),
        help="Path to the current output directory (e.g., .../validation_logs/agent)",
    )
    parser.add_argument(
        "--original_data_path",
        type=Path,
        default=Path("/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/full-duplex/benchmarks/full_duplex_bench/original_data/v1.0"),
        help="Path to the original data directory (e.g., .../original_data/v1.0/candor_turn_taking)",
    )
    parser.add_argument(
        "--revised_output_path",
        type=Path,
        default=Path("/lustre/fsw/portfolios/llmservice/users/vtrinh/projects/s2s_ear_tts_oct_29/result/inferences/full_duplex_bench/metric"),
        help="Destination base directory where reorganized copies will be written",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="candor_turn_taking",
        help="Dataset name used to derive subpath (e.g., candor_turn_taking, candor_pause_handling)",
    )
    parser.add_argument(
        "--require_dataset_in_filename",
        action="store_true",
        help="If set, only process WAV files whose filenames contain the dataset name",
    )
    parser.add_argument(
        "--strict_ids",
        action="store_true",
        help="If set, only process IDs that exist as directories in the original dataset",
    )
    parser.add_argument(
        "--clean_destination",
        action="store_true",
        help="If set, remove the destination dataset subpath before copying to avoid stale IDs",
    )
    return parser.parse_args()


def derive_version_and_dataset_subpath(original_data_path: Path, dataset_name: str) -> Path:
    """Return a Path like 'v1.0/candor_turn_taking' derived from the original data path.

    The function attempts to find a version-like directory (e.g., 'v1.0') followed by the dataset name.
    If not found, it falls back to the last two path components or just the dataset name if needed.
    """
    version_regex = re.compile(r"^v\d+\.\d+$")
    parts = list(original_data_path.parts)

    # Try to find version + dataset pattern
    for i, part in enumerate(parts[:-1]):
        if version_regex.match(part):
            next_part = parts[i + 1] if i + 1 < len(parts) else None
            if next_part == dataset_name:
                return Path(part) / next_part

    # Try to find an 'original_data' anchor and use the remainder
    if "original_data" in parts:
        anchor_index = parts.index("original_data")
        remainder_parts = parts[anchor_index + 1 :]
        if remainder_parts:
            # Ensure dataset name is present at the end
            if remainder_parts[-1] != dataset_name:
                remainder_parts.append(dataset_name)
            return Path(*remainder_parts)

    # Fallback: last two components, ensuring dataset name is present
    if len(parts) >= 2:
        last_two = parts[-2:]
        if last_two[-1] != dataset_name:
            last_two[-1] = dataset_name
        return Path(*last_two)

    # Ultimate fallback
    return Path(dataset_name)


def find_wav_files(root: Path) -> Iterable[Path]:
    return (p for p in root.rglob("*.wav") if p.is_file())


def extract_numeric_id_from_filename(filename: str) -> Optional[int]:
    """Extract integer id from a filename like 'candor_candor_turn_taking_0001.wav' or 'candor_turn_taking_0119_audio.wav'.

    Returns None if no trailing numeric pattern is found.
    """
    # Match digits preceded by start/underscore/hyphen and followed by optional suffix before .wav
    match = re.search(r"(?:^|[_-])(\d+)(?:_[\w-]+)?\.[Ww][Aa][Vv]$", filename)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def extract_numeric_id_with_dataset(
    filename: str,
    dataset_name: Optional[str],
    require_dataset_in_filename: bool,
) -> Optional[int]:
    """Extract numeric id, optionally requiring that the id follows the dataset token.

    When require_dataset_in_filename is True, this enforces that the dataset token
    appears in the filename and the extracted id comes after that token, separated by
    an underscore or hyphen. Matching is case-insensitive.
    """
    if require_dataset_in_filename and dataset_name:
        pattern = re.compile(
            rf"{re.escape(dataset_name)}.*?[_-](\d+)(?:_[\w-]+)?\.[Ww][Aa][Vv]$",
            re.IGNORECASE,
        )
        match = pattern.search(filename)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None
    # Fallback to general extraction
    return extract_numeric_id_from_filename(filename)


def compute_destination_path(
    revised_output_base: Path,
    subpath_version_and_dataset: Path,
    numeric_id: int,
) -> Path:
    return revised_output_base / subpath_version_and_dataset / str(numeric_id) / "output.wav"


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    ensure_parent_directory(dst)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_arguments()

    output_path: Path = args.output_path.expanduser().resolve()
    original_data_path: Path = args.original_data_path.expanduser().resolve()
    revised_output_path: Path = args.revised_output_path.expanduser().resolve()

    if not output_path.exists() or not output_path.is_dir():
        print(f"ERROR: output_path does not exist or is not a directory: {output_path}", file=sys.stderr)
        sys.exit(1)

    if not original_data_path.exists() or not original_data_path.is_dir():
        # We can continue even if this path does not exist, but warn since we rely on it for subpath derivation
        print(f"WARNING: original_data_path does not exist as a directory: {original_data_path}", file=sys.stderr)

    dataset_name: str = args.dataset_name
    dataset_name_lc = dataset_name.lower()

    subpath_version_and_dataset = derive_version_and_dataset_subpath(original_data_path, dataset_name)
    # Optionally clean destination subpath (e.g., v1.0/<dataset>) before copying
    if args.clean_destination:
        dest_dataset_root = revised_output_path / subpath_version_and_dataset
        if dest_dataset_root.exists():
            try:
                shutil.rmtree(dest_dataset_root)
            except Exception as exc:
                print(f"WARNING: failed to clean destination '{dest_dataset_root}': {exc}", file=sys.stderr)

    # Counters
    copied = 0  # output WAVs copied (from output_path)
    skipped = 0
    errors = 0
    unique_ids_seen = set()
    unique_ids_with_original = set()
    dataset_ids_seen = set()
    dataset_ids_with_original = set()
    dataset_output_wavs_copied = 0
    # Original files copied (from original_data_path/<id>) breakdown
    original_files_copied_total = 0
    original_files_copied_wav = 0
    original_files_copied_json = 0
    original_files_copied_other = 0
    # Track which IDs have had their original files copied to avoid duplication per repeated output WAVs
    original_ids_copied = set()

    # Pre-compute allowed IDs if strict mode
    allowed_ids = None
    if args.strict_ids:
        try:
            allowed_ids = {int(p.name) for p in original_data_path.iterdir() if p.is_dir() and p.name.isdigit()}
        except Exception:
            allowed_ids = set()

    # First pass: select one output WAV per ID, preferring filenames without "_dup"
    selected_wav_for_id = {}
    dup_pattern = re.compile(r"_dup\d*", re.IGNORECASE)

    for wav_path in find_wav_files(output_path):
        numeric_id = extract_numeric_id_with_dataset(wav_path.name, dataset_name, True)
        if numeric_id is None:
            skipped += 1
            continue
        if allowed_ids is not None and numeric_id not in allowed_ids:
            skipped += 1
            continue
        current_is_dup = bool(dup_pattern.search(wav_path.name))
        prev = selected_wav_for_id.get(numeric_id)
        if prev is None:
            selected_wav_for_id[numeric_id] = wav_path
        else:
            prev_is_dup = bool(dup_pattern.search(prev.name))
            # Prefer non-dup over dup. If both same status, keep existing selection.
            if prev_is_dup and not current_is_dup:
                selected_wav_for_id[numeric_id] = wav_path

    # Second pass: copy only the selected WAV for each ID (sorted by ID for determinism)
    for numeric_id in sorted(selected_wav_for_id.keys()):
        wav_path = selected_wav_for_id[numeric_id]
        unique_ids_seen.add(numeric_id)
        dataset_ids_seen.add(numeric_id)

        dest_path = compute_destination_path(revised_output_path, subpath_version_and_dataset, numeric_id)
        try:
            copy_file(wav_path, dest_path)
            copied += 1
            dataset_output_wavs_copied += 1
        except Exception as exc:
            errors += 1
            print(f"ERROR copying '{wav_path}' -> '{dest_path}': {exc}", file=sys.stderr)

        # Also copy all contents from the corresponding original dataset directory, if present
        original_item_dir = original_data_path / str(numeric_id)
        if original_item_dir.exists() and original_item_dir.is_dir() and numeric_id not in original_ids_copied:
            unique_ids_with_original.add(numeric_id)
            dataset_ids_with_original.add(numeric_id)
            for item in original_item_dir.iterdir():
                try:
                    dest_item = dest_path.parent / item.name
                    if dest_item.name == "output.wav":
                        continue
                    if item.is_file():
                        copy_file(item, dest_item)
                        original_files_copied_total += 1
                        suffix = item.suffix.lower()
                        if suffix == ".wav":
                            original_files_copied_wav += 1
                        elif suffix == ".json":
                            original_files_copied_json += 1
                        else:
                            original_files_copied_other += 1
                    elif item.is_dir():
                        for sub in item.rglob("*"):
                            if sub.is_file():
                                rel = sub.relative_to(original_item_dir)
                                copy_file(sub, dest_path.parent / rel)
                                original_files_copied_total += 1
                                suffix = sub.suffix.lower()
                                if suffix == ".wav":
                                    original_files_copied_wav += 1
                                elif suffix == ".json":
                                    original_files_copied_json += 1
                                else:
                                    original_files_copied_other += 1
                except Exception as exc:
                    errors += 1
                    print(f"ERROR copying '{item}' -> '{dest_path.parent}': {exc}", file=sys.stderr)
            original_ids_copied.add(numeric_id)
        else:
            skipped += 1

    print(
        f"Done. DatasetOutputWAVsCopied={dataset_output_wavs_copied}, OriginalFilesCopied={original_files_copied_total} "
        f"(WAV={original_files_copied_wav}, JSON={original_files_copied_json}, Other={original_files_copied_other}), "
        f"Errors={errors}, "
        f"DatasetUniqueIDs={len(dataset_ids_seen)}, DatasetUniqueIDsWithOriginal={len(dataset_ids_with_original)}. "
        f"Base subpath='{subpath_version_and_dataset}'"
    )



if __name__ == "__main__":
    main()

