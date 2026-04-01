#!/bin/bash

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/projects/nemo_s2s_merged_dec/NeMo_fc
SCRIPT=$CODE_DIR/scripts/speech_data_generation/convert_vab_audio_to_shar.py

export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HF_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HF_TOKEN=$(cat /lustre/fsw/portfolios/llmservice/users/kevinhu/tokens/huggingface_token)
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"

# Install dependencies if missing
pip install huggingface_hub soundfile librosa --quiet

OUT_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/vab_audio_shars

set -x
# Debug run: 3 examples from single_tool with detailed output
# python ${SCRIPT} \
#     --subset multi_turn \
#     --out_dir ${OUT_DIR} \
#     --cache_dir ${HF_HOME} \
#     --max_examples 3 \
#     --debug

# exit 0

# Full conversion of all subsets
python ${SCRIPT} \
    --subset all \
    --out_dir ${OUT_DIR} \
    --cache_dir ${HF_HOME} \
    --speech_tail_sec 1.2 \
    --fc_start_delay 0.32
set +x
