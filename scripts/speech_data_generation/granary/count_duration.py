#!/usr/bin/env python3
import json
import glob
import os

def main():
    base_dir = "/lustre/fsw/portfolios/llmservice/users/kevinhu/data/granary"
    lang_codes = [f"YTC_en{i}" for i in range(1, 4)]
    
    total_duration = 0.0
    total_entries = 0
    
    for lang_code in lang_codes:
        manifest_dir = os.path.join(base_dir, lang_code, "manifests")
        print(f"\n=== Processing {lang_code} ===")
        
        # Find all manifest files
        manifest_files = glob.glob(os.path.join(manifest_dir, "manifest_*.json"))
        print(f"Found {len(manifest_files)} manifest files")
        
        lang_duration = 0.0
        lang_entries = 0
        
        for manifest_file in sorted(manifest_files):
            print(f"Processing {os.path.basename(manifest_file)}...")
            
            file_duration = 0.0
            file_entries = 0
            
            with open(manifest_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        if 'duration' in entry:
                            duration = float(entry['duration'])
                            file_duration += duration
                            lang_duration += duration
                            total_duration += duration
                        file_entries += 1
                        lang_entries += 1
                        total_entries += 1
            
            print(f"  Entries: {file_entries}, Duration: {file_duration:.2f}s ({file_duration/3600:.2f}h)")
        
        print(f"{lang_code} total: {lang_entries} entries, {lang_duration:.2f}s ({lang_duration/3600:.2f}h)")
    
    print(f"\n=== OVERALL TOTALS ===")
    print(f"Total entries: {total_entries}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Total duration: {total_duration/3600/24:.2f} days")

if __name__ == "__main__":
    main()
