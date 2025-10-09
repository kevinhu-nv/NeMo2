import json
import os
import argparse

# Example usage:
# python clean_up_manifest.py --manifest /path/to/manifest.json --tar /path/to/audio_0.tar --output /path/to/filtered_manifest.json --tar-output /path/to/filtered_audio_0.tar

def filter_manifest(manifest_path, tar_path, output_path, tar_output_path=None):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Create tar output directory if tar_output_path is provided
    if tar_output_path:
        tar_output_dir = os.path.dirname(tar_output_path)
        if tar_output_dir and not os.path.exists(tar_output_dir):
            os.makedirs(tar_output_dir, exist_ok=True)
            print(f"Created tar output directory: {tar_output_dir}")

    # Get all file names in the tar file (without extracting)
    import tarfile
    with tarfile.open(tar_path, 'r') as tar:
        tar_filenames = set(os.path.basename(m.name) for m in tar.getmembers() if m.isfile())

    # Read manifest entries
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    filtered_lines = []
    removed = 0
    manifest_filenames = set()
    total_lines = len(lines)
    
    print(f"Processing {total_lines} manifest entries...")
    
    for i, line in enumerate(lines, 1):
        # Print progress every 1000 entries or at the end
        if i % 1000 == 0 or i == total_lines:
            print(f"Processed {i}/{total_lines} entries ({i/total_lines*100:.1f}%)")
            
        try:
            entry = json.loads(line)
            audio_fp = entry.get('audio_filepath', '')
            audio_base = os.path.basename(audio_fp)
            manifest_filenames.add(audio_base)
            
            if audio_base in tar_filenames:
                filtered_lines.append(line)
            else:
                removed += 1
        except Exception as e:
            # Skip malformed lines
            removed += 1

    # Check for files in tar that don't have manifest entries
    orphaned_tar_files = tar_filenames - manifest_filenames
    orphaned_manifest_files = manifest_filenames - tar_filenames

    print(f"\nProcessing complete!")
    print(f"Original manifest entries: {len(lines)}")
    print(f"Entries kept: {len(filtered_lines)}")   
    print(f"Entries removed: {removed}")
    print(f"\nBidirectional validation:")
    print(f"Files in tar but not in manifest: {len(orphaned_tar_files)}")
    print(f"Files in manifest but not in tar: {len(orphaned_manifest_files)}")
    
    if orphaned_tar_files:
        print(f"\nOrphaned tar files (first 10):")
        for i, filename in enumerate(sorted(orphaned_tar_files)[:10]):
            print(f"  {filename}")
        if len(orphaned_tar_files) > 10:
            print(f"  ... and {len(orphaned_tar_files) - 10} more")
    
    if orphaned_manifest_files:
        print(f"\nOrphaned manifest files (first 10):")
        for i, filename in enumerate(sorted(orphaned_manifest_files)[:10]):
            print(f"  {filename}")
        if len(orphaned_manifest_files) > 10:
            print(f"  ... and {len(orphaned_manifest_files) - 10} more")

    # Write filtered manifest
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in filtered_lines:
            f.write(line)
    
    # Create filtered tar file if tar_output_path is provided
    if tar_output_path:
        print(f"\nCreating filtered tar file: {tar_output_path}")
        with tarfile.open(tar_path, 'r') as source_tar:
            with tarfile.open(tar_output_path, 'w') as target_tar:
                files_copied = 0
                total_files = len([m for m in source_tar.getmembers() if m.isfile()])
                
                for member in source_tar.getmembers():
                    if member.isfile():
                        files_copied += 1
                        if files_copied % 1000 == 0 or files_copied == total_files:
                            print(f"Copied {files_copied}/{total_files} files ({files_copied/total_files*100:.1f}%)")
                        
                        # Only copy files that exist in the filtered manifest
                        if os.path.basename(member.name) in manifest_filenames:
                            # Extract file data and add to new tar
                            file_data = source_tar.extractfile(member)
                            if file_data:
                                tarinfo = tarfile.TarInfo(name=member.name)
                                tarinfo.size = member.size
                                tarinfo.mtime = member.mtime
                                tarinfo.mode = member.mode
                                tarinfo.type = member.type
                                tarinfo.uid = member.uid
                                tarinfo.gid = member.gid
                                tarinfo.uname = member.uname
                                tarinfo.gname = member.gname
                                
                                target_tar.addfile(tarinfo, file_data)
                                file_data.close()
        
        print(f"Filtered tar file created: {tar_output_path}")
        print(f"Files in original tar: {len(tar_filenames)}")
        print(f"Files in filtered tar: {len(manifest_filenames & tar_filenames)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter JSON manifest to match tar file contents and optionally create a filtered tar file.")
    parser.add_argument("--manifest", required=True, help="Path to the JSON manifest file")
    parser.add_argument("--tar", required=True, help="Path to the tar file")
    parser.add_argument("--output", required=True, help="Path to write the filtered manifest")
    parser.add_argument("--tar-output", help="Path to write the filtered tar file (optional)")
    args = parser.parse_args()

    filter_manifest(args.manifest, args.tar, args.output, args.tar_output)

