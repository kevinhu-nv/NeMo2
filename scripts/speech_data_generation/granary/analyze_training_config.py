#!/usr/bin/env python3
"""
Analyze training data configuration to compute training hours, weights, and duration statistics.

This script reads a YAML configuration file containing training data groups and computes:
1. Total training hours for each group and dataset
2. Suggested weights based on training hours
3. Duration statistics for each dataset (mean, median, std, min, max)
4. Overall summary statistics

Usage:
    python analyze_training_config.py --config_path /path/to/config.yaml
    python /lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo/scripts/speech_data_generation/granary/analyze_training_config.py --config_path /lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/configs/data/asr_data_sil2_gr_fa2.yaml --output_path training_analysis_full.json
"""

import json
import yaml
import argparse
import os
import glob
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class DatasetStats:
    """Statistics for a single dataset."""
    name: str
    manifest_path: str
    total_hours: float
    total_samples: int
    mean_duration: float
    median_duration: float
    std_duration: float
    min_duration: float
    max_duration: float
    percentile_25: float
    percentile_75: float


@dataclass
class GroupStats:
    """Statistics for a group of datasets."""
    group_id: int
    current_weight: float
    total_hours: float
    total_samples: int
    datasets: List[DatasetStats]
    suggested_weight: float = 0.0


def expand_manifest_path(manifest_path: str) -> List[str]:
    """
    Expand manifest path patterns like manifest__OP_0..63_CL_.json to actual file paths.
    
    Args:
        manifest_path: Path with potential range patterns
        
    Returns:
        List of actual manifest file paths
    """
    if "__OP_" in manifest_path and "_CL_" in manifest_path:
        # Extract the pattern
        base_dir = os.path.dirname(manifest_path)
        filename = os.path.basename(manifest_path)
        
        # Parse the range pattern
        if ".." in filename:
            # Pattern like manifest__OP_0..63_CL_.json
            start_marker = "__OP_"
            end_marker = "_CL_"
            
            start_idx = filename.find(start_marker) + len(start_marker)
            end_idx = filename.find(end_marker)
            range_part = filename[start_idx:end_idx]
            
            if ".." in range_part:
                start_num, end_num = range_part.split("..")
                start_num, end_num = int(start_num), int(end_num)
                
                # Generate all file paths in the range
                file_paths = []
                for i in range(start_num, end_num + 1):
                    expanded_filename = filename.replace(f"__OP_{range_part}_CL_", f"_{i}")
                    expanded_path = os.path.join(base_dir, expanded_filename)
                    if os.path.exists(expanded_path):
                        file_paths.append(expanded_path)
                
                return file_paths
    
    # If no pattern or single file, return as is (if exists)
    if os.path.exists(manifest_path):
        return [manifest_path]
    else:
        # Try to find similar files using glob
        base_dir = os.path.dirname(manifest_path)
        filename = os.path.basename(manifest_path)
        
        # Try different patterns
        patterns = [
            os.path.join(base_dir, "manifest_*.json"),
            os.path.join(base_dir, "*manifest*.json"),
            manifest_path
        ]
        
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                return sorted(matches)
        
        print(f"Warning: No manifest files found for pattern: {manifest_path}")
        return []


def parse_manifest_file(manifest_path: str, show_progress: bool = True) -> Tuple[List[float], int]:
    """
    Parse a single manifest file and extract duration information.
    
    Args:
        manifest_path: Path to the manifest JSON file
        show_progress: Whether to show progress for large files
        
    Returns:
        Tuple of (list of durations, total samples)
    """
    durations = []
    total_samples = 0
    
    try:
        file_size = os.path.getsize(manifest_path)
        if show_progress:
            print(f"    Processing {os.path.basename(manifest_path)} ({file_size / (1024*1024):.1f} MB)...")
        
        with open(manifest_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Show progress for large files
                if show_progress and line_num % 10000 == 0:
                    print(f"      Processed {line_num:,} lines, found {len(durations):,} samples with duration...")
                
                try:
                    entry = json.loads(line)
                    if 'duration' in entry:
                        duration = float(entry['duration'])
                        durations.append(duration)
                    total_samples += 1
                except json.JSONDecodeError as e:
                    if line_num <= 10:  # Only show first few errors
                        print(f"Warning: Failed to parse line {line_num} in {manifest_path}: {e}")
                except (ValueError, TypeError) as e:
                    if line_num <= 10:  # Only show first few errors
                        print(f"Warning: Invalid duration in line {line_num} of {manifest_path}: {e}")
        
        if show_progress:
            total_duration_hours = sum(durations) / 3600.0
            print(f"    Completed: {total_samples:,} samples, {len(durations):,} with duration, {total_duration_hours:.2f} hours")
                    
    except FileNotFoundError:
        print(f"Warning: Manifest file not found: {manifest_path}")
    except Exception as e:
        print(f"Error reading manifest file {manifest_path}: {e}")
    
    return durations, total_samples


def compute_dataset_stats(manifest_path: str, dataset_name: str, max_files: Optional[int] = None) -> Optional[DatasetStats]:
    """
    Compute statistics for a single dataset.
    
    Args:
        manifest_path: Path to the manifest file(s)
        dataset_name: Name of the dataset
        
    Returns:
        DatasetStats object or None if no data found
    """
    print(f"\n  📂 Processing dataset: {dataset_name}")
    print(f"     Manifest pattern: {manifest_path}")
    
    # Expand manifest path to handle patterns
    manifest_files = expand_manifest_path(manifest_path)
    
    if not manifest_files:
        print(f"  ❌ No manifest files found for {dataset_name}: {manifest_path}")
        return None
    
    # Limit files if specified
    if max_files and len(manifest_files) > max_files:
        print(f"  🔄 Limiting to first {max_files} files (out of {len(manifest_files)} total)")
        manifest_files = manifest_files[:max_files]
    
    print(f"  📄 Processing {len(manifest_files)} manifest files")
    
    all_durations = []
    total_samples = 0
    
    for i, manifest_file in enumerate(manifest_files, 1):
        print(f"  📊 Processing file {i}/{len(manifest_files)}: {os.path.basename(manifest_file)}")
        durations, samples = parse_manifest_file(manifest_file)
        all_durations.extend(durations)
        total_samples += samples
    
    if not all_durations:
        print(f"  ⚠️  Warning: No duration data found for {dataset_name}")
        return None
    
    print(f"  🧮 Computing statistics for {len(all_durations):,} samples...")
    
    # Convert to numpy array for statistics
    durations_array = np.array(all_durations)
    
    # Compute statistics
    total_hours = np.sum(durations_array) / 3600.0
    mean_duration = np.mean(durations_array)
    median_duration = np.median(durations_array)
    std_duration = np.std(durations_array)
    min_duration = np.min(durations_array)
    max_duration = np.max(durations_array)
    percentile_25 = np.percentile(durations_array, 25)
    percentile_75 = np.percentile(durations_array, 75)
    
    print(f"  ✅ Dataset complete: {total_hours:.2f} hours, {total_samples:,} samples")
    
    return DatasetStats(
        name=dataset_name,
        manifest_path=manifest_path,
        total_hours=total_hours,
        total_samples=total_samples,
        mean_duration=mean_duration,
        median_duration=median_duration,
        std_duration=std_duration,
        min_duration=min_duration,
        max_duration=max_duration,
        percentile_25=percentile_25,
        percentile_75=percentile_75
    )


def parse_config_file(config_path: str, max_files_per_dataset: Optional[int] = None, sample_only: bool = False) -> List[GroupStats]:
    """
    Parse the YAML configuration file and compute statistics for each group.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        List of GroupStats objects
    """
    print(f"📖 Loading configuration file: {config_path}")
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    print(f"🔍 Found {len(config_data)} groups in configuration")
    group_stats = []
    
    for group_id, group_config in enumerate(config_data):
        if 'input_cfg' not in group_config:
            print(f"⚠️  Skipping group {group_id}: no input_cfg found")
            continue
            
        current_weight = group_config.get('weight', 1.0)
        num_datasets = len(group_config['input_cfg'])
        
        print(f"\n🏷️  Processing Group {group_id} (weight: {current_weight}, {num_datasets} datasets)")
        datasets = []
        
        for dataset_id, dataset_config in enumerate(group_config['input_cfg']):
            if 'manifest_filepath' not in dataset_config:
                print(f"  ⚠️  Skipping dataset {dataset_id}: no manifest_filepath found")
                continue
                
            manifest_path = dataset_config['manifest_filepath']
            
            # Generate dataset name from manifest path
            dataset_name = f"Group{group_id}_Dataset{dataset_id}"
            if 'granary' in manifest_path:
                # Extract more meaningful name from path
                path_parts = manifest_path.split('/')
                for i, part in enumerate(path_parts):
                    if part == 'granary' and i + 1 < len(path_parts):
                        dataset_name = path_parts[i + 1]
                        break
            
            # Apply sampling limits if specified
            max_files = None
            if sample_only:
                max_files = 2  # Just process 2 files for quick testing
            elif max_files_per_dataset:
                max_files = max_files_per_dataset
                
            dataset_stats = compute_dataset_stats(manifest_path, dataset_name, max_files)
            
            if dataset_stats:
                datasets.append(dataset_stats)
        
        if datasets:
            total_hours = sum(ds.total_hours for ds in datasets)
            total_samples = sum(ds.total_samples for ds in datasets)
            
            print(f"  📊 Group {group_id} summary: {total_hours:.2f} hours, {total_samples:,} samples")
            
            group_stats.append(GroupStats(
                group_id=group_id,
                current_weight=current_weight,
                total_hours=total_hours,
                total_samples=total_samples,
                datasets=datasets
            ))
        else:
            print(f"  ❌ Group {group_id}: No valid datasets found")
    
    print(f"\n✅ Configuration parsing complete. Processed {len(group_stats)} valid groups.")
    return group_stats


def compute_suggested_weights(group_stats: List[GroupStats], method: str = "proportional") -> None:
    """
    Compute suggested weights based on training hours.
    
    Args:
        group_stats: List of GroupStats objects
        method: Method for computing weights ("proportional", "sqrt", "log")
    """
    total_hours = sum(gs.total_hours for gs in group_stats)
    
    if total_hours == 0:
        print("Warning: No training hours found, cannot compute weights")
        return
    
    for group_stat in group_stats:
        if method == "proportional":
            # Weight proportional to hours
            suggested_weight = (group_stat.total_hours / total_hours) * len(group_stats)
        elif method == "sqrt":
            # Square root scaling to reduce dominance of large datasets
            sqrt_hours = np.sqrt(group_stat.total_hours)
            total_sqrt_hours = sum(np.sqrt(gs.total_hours) for gs in group_stats)
            suggested_weight = (sqrt_hours / total_sqrt_hours) * len(group_stats)
        elif method == "log":
            # Log scaling for even more balanced weights
            log_hours = np.log1p(group_stat.total_hours)
            total_log_hours = sum(np.log1p(gs.total_hours) for gs in group_stats)
            suggested_weight = (log_hours / total_log_hours) * len(group_stats)
        else:
            suggested_weight = 1.0
            
        group_stat.suggested_weight = suggested_weight


def print_summary_report(group_stats: List[GroupStats]) -> None:
    """Print a comprehensive summary report."""
    
    print("\n" + "="*80)
    print("TRAINING DATA ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall statistics
    total_hours = sum(gs.total_hours for gs in group_stats)
    total_samples = sum(gs.total_samples for gs in group_stats)
    total_datasets = sum(len(gs.datasets) for gs in group_stats)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total Groups: {len(group_stats)}")
    print(f"  Total Datasets: {total_datasets}")
    print(f"  Total Training Hours: {total_hours:.2f} hours ({total_hours/24:.2f} days)")
    print(f"  Total Samples: {total_samples:,}")
    print(f"  Average Sample Duration: {(total_hours * 3600 / total_samples):.2f} seconds")
    
    # Group-by-group analysis
    print(f"\nGROUP ANALYSIS:")
    print(f"{'Group':<8} {'Current':<8} {'Suggested':<10} {'Hours':<10} {'Samples':<12} {'Datasets':<10} {'% of Total':<12}")
    print(f"{'ID':<8} {'Weight':<8} {'Weight':<10} {'(h)':<10} {'Count':<12} {'Count':<10} {'Hours':<12}")
    print("-" * 80)
    
    for gs in group_stats:
        pct_hours = (gs.total_hours / total_hours * 100) if total_hours > 0 else 0
        print(f"{gs.group_id:<8} {gs.current_weight:<8.1f} {gs.suggested_weight:<10.2f} "
              f"{gs.total_hours:<10.1f} {gs.total_samples:<12,} {len(gs.datasets):<10} {pct_hours:<12.1f}%")
    
    # Detailed dataset statistics
    print(f"\nDETAILED DATASET STATISTICS:")
    for gs in group_stats:
        print(f"\n--- Group {gs.group_id} (Weight: {gs.current_weight} → {gs.suggested_weight:.2f}) ---")
        
        for ds in gs.datasets:
            print(f"\nDataset: {ds.name}")
            print(f"  Manifest: {ds.manifest_path}")
            print(f"  Total Hours: {ds.total_hours:.2f} h")
            print(f"  Total Samples: {ds.total_samples:,}")
            print(f"  Duration Stats (seconds):")
            print(f"    Mean: {ds.mean_duration:.2f}")
            print(f"    Median: {ds.median_duration:.2f}")
            print(f"    Std Dev: {ds.std_duration:.2f}")
            print(f"    Min: {ds.min_duration:.2f}")
            print(f"    Max: {ds.max_duration:.2f}")
            print(f"    25th Percentile: {ds.percentile_25:.2f}")
            print(f"    75th Percentile: {ds.percentile_75:.2f}")
    
    # Weight comparison
    print(f"\nWEIGHT COMPARISON:")
    print(f"{'Group ID':<10} {'Current Weight':<15} {'Suggested Weight':<18} {'Ratio':<10} {'Change':<10}")
    print("-" * 70)
    
    for gs in group_stats:
        ratio = gs.suggested_weight / gs.current_weight if gs.current_weight > 0 else float('inf')
        change = ((gs.suggested_weight - gs.current_weight) / gs.current_weight * 100) if gs.current_weight > 0 else 0
        print(f"{gs.group_id:<10} {gs.current_weight:<15.1f} {gs.suggested_weight:<18.2f} "
              f"{ratio:<10.2f} {change:+.1f}%")


def save_results_to_file(group_stats: List[GroupStats], output_path: str) -> None:
    """Save analysis results to a JSON file."""
    
    results = {
        "summary": {
            "total_groups": len(group_stats),
            "total_datasets": sum(len(gs.datasets) for gs in group_stats),
            "total_hours": sum(gs.total_hours for gs in group_stats),
            "total_samples": sum(gs.total_samples for gs in group_stats)
        },
        "groups": []
    }
    
    for gs in group_stats:
        group_data = {
            "group_id": gs.group_id,
            "current_weight": gs.current_weight,
            "suggested_weight": gs.suggested_weight,
            "total_hours": gs.total_hours,
            "total_samples": gs.total_samples,
            "datasets": []
        }
        
        for ds in gs.datasets:
            dataset_data = {
                "name": ds.name,
                "manifest_path": ds.manifest_path,
                "total_hours": ds.total_hours,
                "total_samples": ds.total_samples,
                "duration_stats": {
                    "mean": ds.mean_duration,
                    "median": ds.median_duration,
                    "std": ds.std_duration,
                    "min": ds.min_duration,
                    "max": ds.max_duration,
                    "percentile_25": ds.percentile_25,
                    "percentile_75": ds.percentile_75
                }
            }
            group_data["datasets"].append(dataset_data)
        
        results["groups"].append(group_data)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze training data configuration")
    parser.add_argument("--config_path", required=True, 
                       help="Path to the YAML configuration file")
    parser.add_argument("--output_path", 
                       help="Path to save JSON results (optional)")
    parser.add_argument("--weight_method", choices=["proportional", "sqrt", "log"],
                       default="proportional",
                       help="Method for computing suggested weights")
    parser.add_argument("--max_files_per_dataset", type=int, default=None,
                       help="Limit number of manifest files per dataset (for testing)")
    parser.add_argument("--sample_only", action="store_true",
                       help="Process only first few files of each dataset for quick testing")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_path):
        print(f"❌ Error: Configuration file not found: {args.config_path}")
        return
    
    print(f"🚀 Starting training data analysis...")
    print(f"📁 Configuration file: {args.config_path}")
    print(f"⚖️  Weight computation method: {args.weight_method}")
    print("=" * 80)
    
    # Parse configuration and compute statistics
    group_stats = parse_config_file(args.config_path, args.max_files_per_dataset, args.sample_only)
    
    if not group_stats:
        print("❌ No valid groups found in configuration file")
        return
    
    print(f"\n🧮 Computing suggested weights using '{args.weight_method}' method...")
    # Compute suggested weights
    compute_suggested_weights(group_stats, method=args.weight_method)
    
    # Print summary report
    print_summary_report(group_stats)
    
    # Save results if output path specified
    if args.output_path:
        save_results_to_file(group_stats, args.output_path)
    else:
        # Default output path
        config_name = os.path.splitext(os.path.basename(args.config_path))[0]
        output_path = f"training_analysis_{config_name}.json"
        save_results_to_file(group_stats, output_path)
    
    print(f"\n🎉 Analysis complete!")


if __name__ == "__main__":
    main()
