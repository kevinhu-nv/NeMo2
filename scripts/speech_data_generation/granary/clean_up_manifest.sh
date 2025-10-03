#!/bin/bash

# Base paths
BASE_PATH="/lustre/fsw/portfolios/llmservice/users/kevinhu/data/granary"
SCRIPT_PATH="/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo/scripts/speech_data_generation/granary/clean_up_manifest.py"

# Function to process a dataset directory
process_dataset() {
    local DATASET_DIR="$1"
    local DATASET_NAME="$2"
    
    echo "Processing ${DATASET_NAME}..."
    
    # Check if dataset directory exists
    if [ ! -d "$DATASET_DIR" ]; then
        echo "Warning: Directory $DATASET_DIR does not exist, skipping..."
        return
    fi
    
    # Check if manifests directory exists
    MANIFESTS_DIR="${DATASET_DIR}/manifests"
    if [ ! -d "$MANIFESTS_DIR" ]; then
        echo "Warning: Manifests directory $MANIFESTS_DIR does not exist, skipping..."
        return
    fi
    
    # Check if tars directory exists
    TARS_DIR="${DATASET_DIR}/tars"
    if [ ! -d "$TARS_DIR" ]; then
        echo "Warning: Tars directory $TARS_DIR does not exist, skipping..."
        return
    fi
    
    # Create output directories if they don't exist
    OUTPUT_DIR="${DATASET_DIR}/manifests_cleaned"
    TAR_OUTPUT_DIR="${DATASET_DIR}/tars_cleaned"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$TAR_OUTPUT_DIR"
    
    # Process all manifest files
    for manifest_file in "$MANIFESTS_DIR"/manifest_*.json; do
        if [ -f "$manifest_file" ]; then
            # Extract manifest number from filename (e.g., manifest_0.json -> 0)
            manifest_name=$(basename "$manifest_file")
            manifest_num=$(echo "$manifest_name" | sed 's/manifest_\([0-9]*\)\.json/\1/')
            
            # Construct corresponding tar file path
            tar_file="${TARS_DIR}/audio_${manifest_num}.tar"
            
            # Construct output paths
            output_manifest="${OUTPUT_DIR}/${manifest_name}"
            output_tar="${TAR_OUTPUT_DIR}/audio_${manifest_num}.tar"
            
            echo "  Processing $manifest_name..."
            
            # Check if tar file exists
            if [ ! -f "$tar_file" ]; then
                echo "    Warning: Tar file $tar_file does not exist, skipping manifest $manifest_name"
                continue
            fi
            
            # Run the clean_up_manifest.py script
            python "$SCRIPT_PATH" --manifest "$manifest_file" --tar "$tar_file" --output "$output_manifest" --tar-output "$output_tar"
            
            if [ $? -eq 0 ]; then
                echo "    Successfully processed $manifest_name"
            else
                echo "    Error processing $manifest_name"
            fi
        fi
    done
    
    echo "Completed processing ${DATASET_NAME}"
    echo "----------------------------------------"
}

# Process YTC_en datasets (YTC_en1 to YTC_en4)
for i in {10..16}; do
    DATASET_DIR="${BASE_PATH}/YTC_en${i}"
    process_dataset "$DATASET_DIR" "YTC_en${i}"
done

# Process LibriLight_en dataset
# LIBRILIGHT_DIR="${BASE_PATH}/LibriLight_en/ll2/webds"
# process_dataset "$LIBRILIGHT_DIR" "LibriLight_en/ll2/webds"

echo "All datasets processed!"