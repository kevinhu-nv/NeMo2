#!/usr/bin/env bash
set -euo pipefail

export AIS_ENDPOINT=http://asr.iad.oci.aistore.nvidia.com:51080

# Configuration
MAX_LANG_NUM=9
BASE_OUT_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/data/granary"
MAX_MANIFEST_NUM=999
MAX_TAR_NUM=999

# for subdir_name in en/0_by_whisper en/0_from_captions en/1_by_whisper; do
# for subdir_name in en/0_from_captions en/1_by_whisper; do
for subdir_name in en/1_by_whisper/bucket1 en/1_by_whisper/bucket2 en/1_by_whisper/bucket3 en/1_by_whisper/bucket4; do
  OUT_DIR="$BASE_OUT_DIR/YODAS2_${subdir_name}"
  mkdir -p "$OUT_DIR/manifests" "$OUT_DIR/tars"
  
  echo "=== Processing $subdir_name ==="
  
  # 1. Download all manifests
  echo "Downloading manifests for $subdir_name..."
  for m in $(seq 0 $MAX_MANIFEST_NUM); do
    MANIFEST="s3://yodas2/${subdir_name}/sharded_manifests/manifest_${m}.json"
    MANIFEST_LOCAL="$OUT_DIR/manifests/manifest_${m}.json"
    if [ -f "$MANIFEST_LOCAL" ]; then
      echo "Skipping $MANIFEST (already exists)"
    else
      echo "Fetching $MANIFEST"
      ais object get "$MANIFEST" "$MANIFEST_LOCAL" || echo "Missing: $MANIFEST"
    fi
  done

  # 2. Download all tar shards
  echo "Downloading tar shards for $subdir_name..."
  for i in $(seq 0 $MAX_TAR_NUM); do
    TAR="s3://yodas2/${subdir_name}/audio_${i}.tar"
    TAR_LOCAL="$OUT_DIR/tars/audio_${i}.tar"
    if [ -f "$TAR_LOCAL" ]; then
      echo "Skipping $TAR (already exists)"
    else
      echo "Fetching $TAR"
      ais object get "$TAR" "$TAR_LOCAL" || echo "Missing: $TAR"
    fi
  done
  
  echo "Completed processing $subdir_name"
  echo ""
done