#!/usr/bin/env python3
"""
Analyze audio statistics from Granary Lhotse manifests.

This script reads Lhotse manifest files and provides comprehensive statistics
including word counts, speech durations, speaker distributions, and more.
"""

import json
import glob
import os
import argparse
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import numpy as np
from pathlib import Path

def load_lhotse_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Load a Lhotse manifest file and return list of entries."""
    entries = []
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {manifest_path}: {e}")
    return entries

def strip_timestamps(text: str) -> str:
    """Remove timestamp tokens like <|x|> from text."""
    if not text:
        return text
    # Remove timestamp patterns like <|123|>
    timestamp_pattern = re.compile(r'<\|\d+\|>')
    text = timestamp_pattern.sub('', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_supervision_stats(entry: Dict[str, Any], entry_id: str = None) -> Dict[str, Any]:
    """Extract statistics from a single Lhotse entry."""
    stats = {
        'duration': 0.0,
        'num_supervisions': 0,
        'total_words': 0,
        'speakers': set(),
        'texts': [],
        'supervision_durations': [],
        'has_timestamps': False,
        'answer_text': '',
        'answer_words': 0,
        'has_empty_answer': False,
        'entry_id': entry_id,
        'empty_fields': []
    }
    
    # Get recording duration
    if 'duration' in entry:
        stats['duration'] = float(entry['duration'])
    
    # Process supervisions
    if 'supervisions' in entry:
        stats['num_supervisions'] = len(entry['supervisions'])
        
        for sup in entry['supervisions']:
            # Speaker info
            if 'speaker' in sup:
                stats['speakers'].add(sup['speaker'])
            
            # Text and word count
            if 'text' in sup and sup['text']:
                text = sup['text'].strip()
                stats['texts'].append(text)
                
                # Check for timestamps before cleaning
                if '<|' in text and '|>' in text:
                    stats['has_timestamps'] = True
                
                # Remove timestamps for word counting
                clean_text = strip_timestamps(text)
                word_count = len(clean_text.split()) if clean_text else 0
                stats['total_words'] += word_count
            
            # Supervision duration
            if 'duration' in sup:
                stats['supervision_durations'].append(float(sup['duration']))
    
    # Process answer field
    if 'answer' in entry and entry['answer']:
        answer_text = entry['answer'].strip()
        # Remove timestamps from answer text before counting words
        clean_answer_text = strip_timestamps(answer_text)
        stats['answer_text'] = clean_answer_text
        stats['answer_words'] = len(clean_answer_text.split()) if clean_answer_text else 0
        stats['has_empty_answer'] = (stats['answer_words'] == 0)
        if stats['has_empty_answer']:
            stats['empty_fields'].append('answer')
    else:
        stats['has_empty_answer'] = True
        stats['empty_fields'].append('answer')
    
    return stats

def analyze_manifest_file(manifest_path: str) -> Dict[str, Any]:
    """Analyze a single manifest file and return statistics."""
    print(f"Processing {os.path.basename(manifest_path)}...")
    
    entries = load_lhotse_manifest(manifest_path)
    file_stats = {
        'file_path': manifest_path,
        'num_entries': len(entries),
        'total_duration': 0.0,
        'total_words': 0,
        'total_supervisions': 0,
        'speakers': set(),
        'entries_with_timestamps': 0,
        'duration_per_entry': [],
        'words_per_entry': [],
        'supervisions_per_entry': [],
        'supervision_durations': [],
        'total_answer_words': 0,
        'entries_with_empty_answers': 0,
        'answer_words_per_entry': [],
        'empty_records': []
    }
    
    for i, entry in enumerate(entries):
        entry_id = entry.get('id', f'entry_{i}')
        entry_stats = extract_supervision_stats(entry, entry_id)
        
        file_stats['total_duration'] += entry_stats['duration']
        file_stats['total_words'] += entry_stats['total_words']
        file_stats['total_supervisions'] += entry_stats['num_supervisions']
        file_stats['speakers'].update(entry_stats['speakers'])
        file_stats['total_answer_words'] += entry_stats['answer_words']
        
        if entry_stats['has_timestamps']:
            file_stats['entries_with_timestamps'] += 1
        
        if entry_stats['has_empty_answer']:
            file_stats['entries_with_empty_answers'] += 1
            # Store empty record info
            file_stats['empty_records'].append({
                'entry_id': entry_stats['entry_id'],
                'empty_fields': entry_stats['empty_fields']
            })
        
        file_stats['duration_per_entry'].append(entry_stats['duration'])
        file_stats['words_per_entry'].append(entry_stats['total_words'])
        file_stats['supervisions_per_entry'].append(entry_stats['num_supervisions'])
        file_stats['supervision_durations'].extend(entry_stats['supervision_durations'])
        file_stats['answer_words_per_entry'].append(entry_stats['answer_words'])
    
    return file_stats

def print_file_stats(file_stats: Dict[str, Any]):
    """Print statistics for a single file."""
    print(f"  Entries: {file_stats['num_entries']}")
    print(f"  Duration: {file_stats['total_duration']:.2f}s ({file_stats['total_duration']/3600:.2f}h)")
    print(f"  Words: {file_stats['total_words']:,}")
    print(f"  Answer words: {file_stats['total_answer_words']:,}")
    print(f"  Supervisions: {file_stats['total_supervisions']}")
    print(f"  Unique speakers: {len(file_stats['speakers'])}")
    print(f"  Entries with timestamps: {file_stats['entries_with_timestamps']} ({file_stats['entries_with_timestamps']/file_stats['num_entries']*100:.2f}%)")
    print(f"  Entries with empty answers: {file_stats['entries_with_empty_answers']} ({file_stats['entries_with_empty_answers']/file_stats['num_entries']*100:.2f}%)")
    
    # Print empty records details
    if file_stats['empty_records']:
        print(f"  Empty records details:")
        for record in file_stats['empty_records'][:10]:  # Show first 10
            print(f"    ID: {record['entry_id']}, Empty fields: {', '.join(record['empty_fields'])}")
        if len(file_stats['empty_records']) > 10:
            print(f"    ... and {len(file_stats['empty_records']) - 10} more empty records")
    
    if file_stats['duration_per_entry']:
        durations = np.array(file_stats['duration_per_entry'])
        print(f"  Duration stats - Mean: {durations.mean():.2f}s, Median: {np.median(durations):.2f}s, Max: {durations.max():.2f}s")
    
    if file_stats['words_per_entry']:
        words = np.array(file_stats['words_per_entry'])
        print(f"  Word stats - Mean: {words.mean():.1f}, Median: {np.median(words):.1f}, Max: {words.max()}")
    
    if file_stats['supervisions_per_entry']:
        supervisions = np.array(file_stats['supervisions_per_entry'])
        print(f"  Supervision stats - Mean: {supervisions.mean():.1f}, Median: {np.median(supervisions):.1f}, Max: {supervisions.max()}")
    
    if file_stats['answer_words_per_entry']:
        answer_words = np.array(file_stats['answer_words_per_entry'])
        print(f"  Answer word stats - Mean: {answer_words.mean():.1f}, Median: {np.median(answer_words):.1f}, Max: {answer_words.max()}")

def print_empty_records_summary(all_stats: List[Dict[str, Any]], lang_code: str):
    """Print summary of empty records across all files for a language."""
    all_empty_records = []
    for file_stats in all_stats:
        for record in file_stats['empty_records']:
            all_empty_records.append({
                'file': os.path.basename(file_stats['file_path']),
                'entry_id': record['entry_id'],
                'empty_fields': record['empty_fields']
            })
    
    if all_empty_records:
        print(f"\n=== {lang_code} EMPTY RECORDS SUMMARY ===")
        print(f"Total empty records: {len(all_empty_records)}")
        
        # Group by file
        file_groups = {}
        for record in all_empty_records:
            file_name = record['file']
            if file_name not in file_groups:
                file_groups[file_name] = []
            file_groups[file_name].append(record)
        
        for file_name, records in file_groups.items():
            print(f"\nFile: {file_name} ({len(records)} empty records)")
            for record in records[:5]:  # Show first 5 per file
                print(f"  ID: {record['entry_id']}, Empty fields: {', '.join(record['empty_fields'])}")
            if len(records) > 5:
                print(f"  ... and {len(records) - 5} more empty records in this file")

def print_summary_stats(all_stats: List[Dict[str, Any]], lang_code: str):
    """Print summary statistics for a language code."""
    total_entries = sum(s['num_entries'] for s in all_stats)
    total_duration = sum(s['total_duration'] for s in all_stats)
    total_words = sum(s['total_words'] for s in all_stats)
    total_answer_words = sum(s['total_answer_words'] for s in all_stats)
    total_supervisions = sum(s['total_supervisions'] for s in all_stats)
    
    all_speakers = set()
    for s in all_stats:
        all_speakers.update(s['speakers'])
    
    entries_with_timestamps = sum(s['entries_with_timestamps'] for s in all_stats)
    entries_with_empty_answers = sum(s['entries_with_empty_answers'] for s in all_stats)
    
    print(f"\n=== {lang_code} SUMMARY ===")
    print(f"Total entries: {total_entries:,}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration/3600:.2f}h)")
    print(f"Total words: {total_words:,}")
    print(f"Total answer words: {total_answer_words:,}")
    print(f"Total supervisions: {total_supervisions:,}")
    print(f"Unique speakers: {len(all_speakers)}")
    print(f"Entries with timestamps: {entries_with_timestamps:,} ({entries_with_timestamps/total_entries*100:.2f}%)")
    print(f"Entries with empty answers: {entries_with_empty_answers:,} ({entries_with_empty_answers/total_entries*100:.2f}%)")
    print(f"Average words per hour: {total_words/(total_duration/3600):.0f}")
    print(f"Average answer words per hour: {total_answer_words/(total_duration/3600):.0f}")
    print(f"Average duration per entry: {total_duration/total_entries:.2f}s")
    print(f"Average words per entry: {total_words/total_entries:.1f}")
    print(f"Average answer words per entry: {total_answer_words/total_entries:.1f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Granary Lhotse manifest statistics")
    parser.add_argument("--base_dir", default="/lustre/fsw/portfolios/llmservice/users/kevinhu/data/granary", help="Base directory containing Granary data")
    parser.add_argument("--lang_codes", nargs="+", 
                       default=[f"YTC_en{i}" for i in range(1, 4)],
                       help="Language codes to analyze")
    parser.add_argument("--output_file", help="Output file to save detailed statistics")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    lang_codes = args.lang_codes
    
    overall_stats = {
        'total_entries': 0,
        'total_duration': 0.0,
        'total_words': 0,
        'total_answer_words': 0,
        'total_supervisions': 0,
        'all_speakers': set(),
        'entries_with_timestamps': 0,
        'entries_with_empty_answers': 0,
        'all_durations': [],
        'all_words': [],
        'all_answer_words': [],
        'all_supervisions': []
    }
    
    detailed_stats = []
    
    for lang_code in lang_codes:
        manifest_dir = base_dir / lang_code / "manifests"
        
        if not manifest_dir.exists():
            print(f"Warning: Directory {manifest_dir} does not exist, skipping {lang_code}")
            continue
            
        print(f"\n=== Processing {lang_code} ===")
        
        # Find all manifest files
        manifest_files = list(manifest_dir.glob("manifest_*.json"))
        print(f"Found {len(manifest_files)} manifest files")
        
        lang_stats = []
        
        for manifest_file in sorted(manifest_files):
            file_stats = analyze_manifest_file(str(manifest_file))
            print_file_stats(file_stats)
            lang_stats.append(file_stats)
            detailed_stats.append(file_stats)
            
            # Update overall stats
            overall_stats['total_entries'] += file_stats['num_entries']
            overall_stats['total_duration'] += file_stats['total_duration']
            overall_stats['total_words'] += file_stats['total_words']
            overall_stats['total_answer_words'] += file_stats['total_answer_words']
            overall_stats['total_supervisions'] += file_stats['total_supervisions']
            overall_stats['all_speakers'].update(file_stats['speakers'])
            overall_stats['entries_with_timestamps'] += file_stats['entries_with_timestamps']
            overall_stats['entries_with_empty_answers'] += file_stats['entries_with_empty_answers']
            overall_stats['all_durations'].extend(file_stats['duration_per_entry'])
            overall_stats['all_words'].extend(file_stats['words_per_entry'])
            overall_stats['all_answer_words'].extend(file_stats['answer_words_per_entry'])
            overall_stats['all_supervisions'].extend(file_stats['supervisions_per_entry'])
        
        print_summary_stats(lang_stats, lang_code)
        print_empty_records_summary(lang_stats, lang_code)
    
    # Print overall summary
    print(f"\n{'='*50}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*50}")
    print(f"Total entries: {overall_stats['total_entries']:,}")
    print(f"Total duration: {overall_stats['total_duration']:.2f}s ({overall_stats['total_duration']/3600:.2f}h)")
    print(f"Total duration: {overall_stats['total_duration']/3600/24:.2f} days")
    print(f"Total words: {overall_stats['total_words']:,}")
    print(f"Total answer words: {overall_stats['total_answer_words']:,}")
    print(f"Total supervisions: {overall_stats['total_supervisions']:,}")
    print(f"Unique speakers: {len(overall_stats['all_speakers'])}")
    print(f"Entries with timestamps: {overall_stats['entries_with_timestamps']:,} ({overall_stats['entries_with_timestamps']/overall_stats['total_entries']*100:.2f}%)")
    print(f"Entries with empty answers: {overall_stats['entries_with_empty_answers']:,} ({overall_stats['entries_with_empty_answers']/overall_stats['total_entries']*100:.2f}%)")
    print(f"Average words per hour: {overall_stats['total_words']/(overall_stats['total_duration']/3600):.0f}")
    print(f"Average answer words per hour: {overall_stats['total_answer_words']/(overall_stats['total_duration']/3600):.0f}")
    print(f"Average duration per entry: {overall_stats['total_duration']/overall_stats['total_entries']:.2f}s")
    print(f"Average words per entry: {overall_stats['total_words']/overall_stats['total_entries']:.1f}")
    print(f"Average answer words per entry: {overall_stats['total_answer_words']/overall_stats['total_entries']:.1f}")
    
    if overall_stats['all_durations']:
        durations = np.array(overall_stats['all_durations'])
        print(f"\nDuration distribution:")
        print(f"  Mean: {durations.mean():.2f}s")
        print(f"  Median: {np.median(durations):.2f}s")
        print(f"  Std: {durations.std():.2f}s")
        print(f"  Min: {durations.min():.2f}s")
        print(f"  Max: {durations.max():.2f}s")
        print(f"  95th percentile: {np.percentile(durations, 95):.2f}s")
    
    if overall_stats['all_words']:
        words = np.array(overall_stats['all_words'])
        print(f"\nWord count distribution:")
        print(f"  Mean: {words.mean():.1f}")
        print(f"  Median: {np.median(words):.1f}")
        print(f"  Std: {words.std():.1f}")
        print(f"  Min: {words.min()}")
        print(f"  Max: {words.max()}")
        print(f"  95th percentile: {np.percentile(words, 95):.1f}")
    
    if overall_stats['all_answer_words']:
        answer_words = np.array(overall_stats['all_answer_words'])
        print(f"\nAnswer word count distribution:")
        print(f"  Mean: {answer_words.mean():.1f}")
        print(f"  Median: {np.median(answer_words):.1f}")
        print(f"  Std: {answer_words.std():.1f}")
        print(f"  Min: {answer_words.min()}")
        print(f"  Max: {answer_words.max()}")
        print(f"  95th percentile: {np.percentile(answer_words, 95):.1f}")
    
    # Save detailed statistics if requested
    if args.output_file:
        output_data = {
            'overall_stats': {
                'total_entries': overall_stats['total_entries'],
                'total_duration': overall_stats['total_duration'],
                'total_words': overall_stats['total_words'],
                'total_answer_words': overall_stats['total_answer_words'],
                'total_supervisions': overall_stats['total_supervisions'],
                'unique_speakers': len(overall_stats['all_speakers']),
                'entries_with_timestamps': overall_stats['entries_with_timestamps'],
                'entries_with_empty_answers': overall_stats['entries_with_empty_answers'],
                'speakers': list(overall_stats['all_speakers'])
            },
            'file_stats': detailed_stats
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nDetailed statistics saved to {args.output_file}")

if __name__ == "__main__":
    main()
