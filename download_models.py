#!/usr/bin/env python3
"""
Script to download pretrained models separately to avoid disk space issues.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model(model_id, local_dir):
    """Download a model from Hugging Face Hub"""
    print(f"Downloading {model_id} to {local_dir}...")
    
    # Create directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Use hard copies to avoid symlink issues
        )
        print(f"✅ Successfully downloaded {model_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_id}: {e}")
        return False

def main():
    # Base directory for models
    base_dir = "/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo/pretrained_models"
    
    # Models to download
    models = {
        "TinyLlama/TinyLlama_v1.1": f"{base_dir}/TinyLlama--TinyLlama_v1.1",
        "nvidia/low-frame-rate-speech-codec-22khz": f"{base_dir}/low-frame-rate-speech-codec-22khz.nemo",
        "stt_en_fastconformer_hybrid_large_streaming_80ms": f"{base_dir}/stt_en_fastconformer_hybrid_large_streaming_80ms.nemo",
        "stt_en_fastconformer_transducer_large": f"{base_dir}/stt_en_fastconformer_transducer_large.nemo",
    }
    
    # Check which models already exist
    existing_models = []
    missing_models = []
    
    for model_id, local_path in models.items():
        if os.path.exists(local_path):
            print(f"✅ {model_id} already exists at {local_path}")
            existing_models.append(model_id)
        else:
            print(f"❌ {model_id} missing, will download to {local_path}")
            missing_models.append((model_id, local_path))
    
    if not missing_models:
        print("All models are already downloaded!")
        return
    
    # Download missing models
    print(f"\nDownloading {len(missing_models)} missing models...")
    
    for model_id, local_path in missing_models:
        success = download_model(model_id, local_path)
        if not success:
            print(f"Failed to download {model_id}. Skipping...")
            continue
    
    print("\nDownload complete!")

if __name__ == "__main__":
    main() 