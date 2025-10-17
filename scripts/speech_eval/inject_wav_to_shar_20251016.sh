#!/bin/bash

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo

SCRIPT=$CODE_DIR/scripts/speech_eval/inject_wav_to_shar.py
in_dir=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/kevinhu/triviaqa_train/shar_duplex/manifest_000000

# BASE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/elena_16oct25_recordings/wav
BASE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/kevin_17oct

OUT_DIR=$BASE_DIR/shar_duplex_transcribed
WAV_DIR=$BASE_DIR

export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HF_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"

# python ${SCRIPT} --in_dir ${in_dir} --out_shar_dir ${OUT_DIR} --shar_index 0 --new_wav_dir ${WAV_DIR}

python ${SCRIPT} --in_dir ${in_dir} --out_shar_dir ${OUT_DIR} --shar_index 0 --new_wav_dir ${WAV_DIR} --transcribe --asr_model nvidia/parakeet-tdt-0.6b-v2

exit 0

# python ${SCRIPT} --in_dir ${in_dir} --out_shar_dir ${OUT_DIR} --shar_index 0 --new_wav_dir ${WAV_DIR} --transcribe --asr_model nvidia/parakeet-tdt-0.6b-v2 --use_llm --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct --use_end_timestamp