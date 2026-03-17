#!/bin/bash

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/projects/nemo_s2s_merged_dec/NeMo_KH_stt-inter
SCRIPT=$CODE_DIR/scripts/speech_data_generation/convert_bfcl_audio_to_shar.py

export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HF_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HF_TOKEN=$(cat /lustre/fsw/portfolios/llmservice/users/kevinhu/tokens/huggingface_token)
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"

# Install dependencies if missing
pip install fastparquet huggingface_hub soundfile --quiet

OUT_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/bfcl_v3_audio_shars

set -x
# Quick debug run (3 examples)
# python ${SCRIPT} \
#     --subset BFCL_v3_simple \
#     --out_dir ${OUT_DIR} \
#     --cache_dir ${HF_HOME} \
#     --max_examples 3

# Full conversion of all subsets
# turn_silence_sec=1.2: silence padding after user speech, giving room for
# the TOOLCALL supervision (placed at user_end + 0.32s) to fit within bounds.
python ${SCRIPT} \
    --subset all \
    --out_dir ${OUT_DIR} \
    --cache_dir ${HF_HOME} \
    --turn_silence_sec 1.2
set +x