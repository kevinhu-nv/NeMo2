#!/usr/bin/env python3
"""
Script to analyze the pred_text field in JSON files and count the percentage of empty records.
Supports both JSONL format and concatenated JSON objects format.
"""

import json
import argparse
import sys

def parse_json_file(file_path):
    """
    Parses a JSON file that can be either:
    1. JSONL format (each line is a JSON object) - new format
    2. Concatenated JSON objects separated by '}{' - old format
    
    Returns a list of dicts.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check if it's JSONL format (each line is a valid JSON object)
            lines = content.strip().split('\n')
            if len(lines) > 1:
                # Try to parse as JSONL first
                try:
                    jsonl_data = []
                    for line in lines:
                        if line.strip():  # Skip empty lines
                            jsonl_data.append(json.loads(line.strip()))
                    return jsonl_data
                except json.JSONDecodeError:
                    # Not JSONL, fall back to old format parsing
                    pass
            
            # Fall back to old format parsing (concatenated JSON objects)
            # The file is not a valid JSON array, but a concatenation of JSON objects
            # separated by '}{', so we split and fix it
            objects = content.strip().split('}{')
            for i, obj in enumerate(objects):
                if not obj.strip():
                    continue
                if not obj.startswith('{'):
                    obj = '{' + obj
                if not obj.endswith('}'):
                    obj = obj + '}'
                data.append(json.loads(obj))
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file '{file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        sys.exit(1)
    
    return data

def is_empty_text(text):
    """
    Check if text is considered empty.
    Returns True if text is None, empty string, or contains only whitespace.
    """
    if text is None:
        return True
    if isinstance(text, str):
        return text.strip() == ""
    return False

def analyze_pred_text(data, verbose=False):
    """
    Analyze the pred_text field in the data.
    Returns statistics about empty vs non-empty records.
    """
    total_records = len(data)
    empty_records = 0
    non_empty_records = 0
    missing_field_records = 0
    
    empty_indices = []
    non_empty_indices = []
    missing_field_indices = []
    
    for idx, entry in enumerate(data):
        if "pred_text" not in entry:
            missing_field_records += 1
            missing_field_indices.append(idx)
            if verbose:
                print(f"Record {idx+1}: Missing 'pred_text' field")
        else:
            pred_text = entry["pred_text"]
            if is_empty_text(pred_text):
                empty_records += 1
                empty_indices.append(idx)
                if verbose:
                    print(f"Record {idx+1}: Empty pred_text")
            else:
                non_empty_records += 1
                non_empty_indices.append(idx)
                if verbose:
                    print(f"Record {idx+1}: Non-empty pred_text: '{pred_text[:50]}{'...' if len(pred_text) > 50 else ''}'")
    
    return {
        'total_records': total_records,
        'empty_records': empty_records,
        'non_empty_records': non_empty_records,
        'missing_field_records': missing_field_records,
        'empty_indices': empty_indices,
        'non_empty_indices': non_empty_indices,
        'missing_field_indices': missing_field_indices
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze pred_text field in JSON files and count empty records.")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the JSON file.")
    parser.add_argument("--verbose", action="store_true", help="Print individual record details")
    parser.add_argument("--show_indices", action="store_true", help="Show indices of empty/non-empty records")
    args = parser.parse_args()

    print(f"Analyzing file: {args.json_file}")
    print("=" * 60)
    
    # Parse the JSON file
    data = parse_json_file(args.json_file)
    
    if not data:
        print("No data found in the file.")
        return
    
    # Analyze pred_text field
    stats = analyze_pred_text(data, verbose=args.verbose)
    
    # Calculate percentages
    total = stats['total_records']
    empty_pct = (stats['empty_records'] / total * 100) if total > 0 else 0
    non_empty_pct = (stats['non_empty_records'] / total * 100) if total > 0 else 0
    missing_pct = (stats['missing_field_records'] / total * 100) if total > 0 else 0
    
    # Print summary
    print(f"\n=== PRED_TEXT ANALYSIS SUMMARY ===")
    print(f"Total records: {total}")
    print(f"Empty pred_text: {stats['empty_records']} ({empty_pct:.2f}%)")
    print(f"Non-empty pred_text: {stats['non_empty_records']} ({non_empty_pct:.2f}%)")
    print(f"Missing pred_text field: {stats['missing_field_records']} ({missing_pct:.2f}%)")
    
    if args.show_indices:
        print(f"\n=== DETAILED INDICES ===")
        if stats['empty_indices']:
            print(f"Empty pred_text indices: {stats['empty_indices'][:20]}{'...' if len(stats['empty_indices']) > 20 else ''}")
        if stats['non_empty_indices']:
            print(f"Non-empty pred_text indices: {stats['non_empty_indices'][:20]}{'...' if len(stats['non_empty_indices']) > 20 else ''}")
        if stats['missing_field_indices']:
            print(f"Missing field indices: {stats['missing_field_indices'][:20]}{'...' if len(stats['missing_field_indices']) > 20 else ''}")
    
    # Additional analysis
    if stats['non_empty_records'] > 0:
        print(f"\n=== ADDITIONAL STATISTICS ===")
        print(f"Percentage of records with empty pred_text: {empty_pct:.2f}%")
        print(f"Percentage of records with non-empty pred_text: {non_empty_pct:.2f}%")
        
        if stats['missing_field_records'] > 0:
            print(f"Percentage of records missing pred_text field: {missing_pct:.2f}%")

if __name__ == "__main__":
    main()
