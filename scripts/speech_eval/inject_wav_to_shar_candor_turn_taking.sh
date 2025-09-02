#!/bin/bash

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo

SCRIPT=$CODE_DIR/scripts/speech_eval/inject_wav_to_shar_candor_turn_taking.py
BASE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking
# OUT_DIR=$BASE_DIR/shar_duplex
OUT_DIR=$BASE_DIR/shar_duplex_transcribed
WAV_DIR=$BASE_DIR
CONTAINER=/lustre/fsw/portfolios/convai/users/cchen1/containers/nemo_duplex_jun.sqsh

in_dir=/lustre/fsw/portfolios/convai/projects/convai_convaird_nemo-speech/data/duplex/kevinhu/triviaqa_train/shar_duplex/manifest_000000
# enroot start --mount /lustre:/lustre --mount /home:/home ${CONTAINER} python ${SCRIPT} --in_dir ${in_dir} --out_shar_dir ${OUT_DIR} --shar_index 0 --new_wav_dir ${WAV_DIR}

# python ${SCRIPT} --in_dir ${in_dir} --out_shar_dir ${OUT_DIR} --shar_index 0 --new_wav_dir ${WAV_DIR}

# python ${SCRIPT} --in_dir ${in_dir} --out_shar_dir ${OUT_DIR} --shar_index 0 --new_wav_dir ${WAV_DIR} --transcribe --asr_model nvidia/parakeet-tdt-0.6b-v2

export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HF_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
python ${SCRIPT} --in_dir ${in_dir} --out_shar_dir ${OUT_DIR} --shar_index 0 --new_wav_dir ${WAV_DIR} --transcribe --asr_model nvidia/parakeet-tdt-0.6b-v2 --use_llm --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct --use_end_timestamp