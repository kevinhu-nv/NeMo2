#!/bin/bash
set -euo pipefail

DEVICE="${1:-cuda:0}"
NUM_SHARDS="${2:-}"

INPUT_DIR=/lustre/fsw/portfolios/llmservice/users/cchen1/code/data_generation/shar_generation/fisher/fisher_v2
OUTPUT_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/data_generation/shar_generation/fisher/fisher_v2_sensevoice
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cache dirs (avoid writing to home directory)
export HF_HOME="/lustre/fsw/portfolios/llmservice/users/kevinhu/hfcache"
export TORCH_HOME="/lustre/fsw/portfolios/llmservice/users/kevinhu/hfcache"
export MODELSCOPE_CACHE="/lustre/fsw/portfolios/llmservice/users/kevinhu/hfcache/modelscope"
export FUNASR_CACHE="/lustre/fsw/portfolios/llmservice/users/kevinhu/hfcache/funasr"

# Install dependencies if missing
pip install -q funasr modelscope soundfile torchaudio

CMD="python ${SCRIPT_DIR}/transcribe_fisher_sensevoice.py \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --device ${DEVICE}"

if [ -n "${NUM_SHARDS}" ]; then
    CMD="${CMD} --num_shards ${NUM_SHARDS}"
fi

echo "Running: ${CMD}"
eval ${CMD}
