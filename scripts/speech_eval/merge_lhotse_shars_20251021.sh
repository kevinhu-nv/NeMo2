#!/bin/bash

# Script to merge three shar directories into demo_20251021
# Date: October 21, 2025

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Input shar directories
SHAR_DIR_1="/lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/tmp/shar"
SHAR_DIR_2="/lustre/fsw/portfolios/llmservice/users/kevinhu/data/elena_16oct25_recordings/wav/shar_duplex_transcribed/"
SHAR_DIR_3="/lustre/fsw/portfolios/llmservice/users/kevinhu/data/kevin_17oct/shar_duplex_transcribed/"

# Output directory
OUTPUT_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/data/demo_20251021/"

echo "Merging shar directories..."
echo "Input directories:"
echo "  1. $SHAR_DIR_1"
echo "  2. $SHAR_DIR_2"
echo "  3. $SHAR_DIR_3"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the merge script
python3 "${SCRIPT_DIR}/merge_lhotse_shars.py" \
    --shar_dirs \
        "$SHAR_DIR_1" \
        "$SHAR_DIR_2" \
        "$SHAR_DIR_3" \
    --output_dir "$OUTPUT_DIR" \
    --recording_format flac \
    --target_audio_format flac

echo ""
echo "Merge complete!"

