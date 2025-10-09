#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HUGGINGFACE_HUB_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/torch"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/torch/nemo"
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/datasets"
export TRANSFORMERS_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/models"
export PIP_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/pip"
export XDG_CACHE_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"

MOONCAST_DIR=/workspace/MoonCast
export PYTHONPATH="$MOONCAST_DIR:${PYTHONPATH-}"

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo/scripts/speech_data_generation
BASE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_mooncast
WAV_DIR=$BASE_DIR

python -c "import whisper" 2>/dev/null || pip install openai-whisper

# Run without paraphrasing
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_mooncast
# python ${CODE_DIR}/mooncast_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR}

# Run with paraphrasing
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_paraphrase_mooncast
python ${CODE_DIR}/mooncast_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR} --paraphrase_user_text --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct