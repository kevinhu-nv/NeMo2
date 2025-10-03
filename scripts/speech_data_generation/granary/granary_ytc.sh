#!/usr/bin/env bash
set -euo pipefail

export AIS_ENDPOINT=http://asr.iad.oci.aistore.nvidia.com:51080

# Configuration
MAX_LANG_NUM=16
BASE_OUT_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/data/granary"
MAX_MANIFEST_NUM=63
MAX_TAR_NUM=63

for lang_num in $(seq 10 $MAX_LANG_NUM); do
  LANG_CODE="en${lang_num}"
  OUT_DIR="$BASE_OUT_DIR/YTC_${LANG_CODE}"
  mkdir -p "$OUT_DIR/manifests" "$OUT_DIR/tars"
  
  echo "=== Processing $LANG_CODE ==="
  
  # 1. Download all manifests
  echo "Downloading manifests for $LANG_CODE..."
  for m in $(seq 0 $MAX_MANIFEST_NUM); do
    MANIFEST="s3://YTC/${LANG_CODE}/sharded_manifests/manifest_${m}.json"
    MANIFEST_LOCAL="$OUT_DIR/manifests/manifest_${m}.json"
    if [ -f "$MANIFEST_LOCAL" ]; then
      echo "Skipping $MANIFEST (already exists)"
    else
      echo "Fetching $MANIFEST"
      ais object get "$MANIFEST" "$MANIFEST_LOCAL" || echo "Missing: $MANIFEST"
    fi
  done

  # 2. Download all tar shards
  echo "Downloading tar shards for $LANG_CODE..."
  for i in $(seq 0 $MAX_TAR_NUM); do
    TAR="s3://YTC/${LANG_CODE}/audio_${i}.tar"
    TAR_LOCAL="$OUT_DIR/tars/audio_${i}.tar"
    if [ -f "$TAR_LOCAL" ]; then
      echo "Skipping $TAR (already exists)"
    else
      echo "Fetching $TAR"
      ais object get "$TAR" "$TAR_LOCAL" || echo "Missing: $TAR"
    fi
  done
  
  echo "Completed processing $LANG_CODE"
  echo ""
done

echo "All language variants (en1-en${MAX_LANG_NUM}) have been processed!"
