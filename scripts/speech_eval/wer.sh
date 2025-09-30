#!/bin/bash

################################################################################
# S2S models

# json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_asr_sil2_1nodes_reproduce-old_asr_local/validation_logs/metadatas/ls_test_other.json'
# json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_asr_sil_4nodes_reproduce-old_asr_ra_d6_noscale/validation_logs/metadatas/riva_asr_6p0.json'
# json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_asr_sil2_4nodes_reproduce-old_asr_la_d15/validation_logs/metadatas/ls_test_clean.json'
# json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_infer_asr_1nodes_reproduce-old_asr_tinyllama/validation_logs/metadatas/inf.json'
# json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_convasr_4nodes_reproduce-old_convasr_sa_la_d4_0828/validation_logs/metadatas/ultrachat.json'
json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_convasr_4nodes_reproduce-old_convasr_sa_la_d4_08281419/validation_logs/metadatas/ultrachat.json'

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo
# python ${CODE_DIR}/scripts/speech_eval/wer.py \
#   --json_file $json_output \
#   --verbose

# exit 0

################################################################################
# Streaming ASR models

CODE_DIR=/lustre/fsw/convai_convaird_nemo-speech/users/kevinhu/s2s/NeMo

# EXP_NAME=DFW_qwen_1b_asr_sil2_4nodes_reproduce-old_asr_la_d15_na2
# EXP_NAME=DFW_qwen_1b_convasr_4nodes_reproduce-old_convasr_sa_la_d4_08281419
# EXP_NAME=EOS_qwen_1b_asr_sil2_4nodes_reproduce-old_asr_la_d15_na2_sfix_sl128
# EXP_NAME=EOS_qwen_1b_asr_sil2_4nodes_reproduce-old_asr_la_d15_na2_sfix_sl1024
EXP_NAME=DFW_qwen_1b_asr_sil2_4nodes_asr_la_d15_na2_3B
# EXP_NAME=DFW_qwen_1b_asr_sil2_ma_4nodes_asr_la_d15_na2_3B_ma

LOG_DIR="/lustre/fsw/convai_convaird_nemo-speech/users/kevinhu/s2s/exp/${EXP_NAME}/inf"
BASE_DIR="${LOG_DIR}/validation_logs/metadatas"

TEST_SETS="ls_test_clean.json ls_test_other.json spgispeech.json gigaspeech.json earnings22.json riva_asr_6p0.json ami.json tedlium.json voxpopuli.json"
# TEST_SETS="ls_test_clean.json ls_test_other.json spgispeech.json"

TIMESTAMP=$(date +%Y%m%d)
for fname in $TEST_SETS; do
  log_dir="${LOG_DIR}/${TIMESTAMP}"
  mkdir -p "$log_dir"
  json_output="${BASE_DIR}/${fname}"
  log_output="${log_dir}/${fname}.log"
  echo "Processing $json_output"
  python ${CODE_DIR}/scripts/speech_eval/wer.py \
    --json_file "$json_output" \
    --verbose 2>&1 | tee "$log_output" 
done

# TIMESTAMP=20250910_1347
for fname in $TEST_SETS; do
  log_dir="${LOG_DIR}/${TIMESTAMP}"
  log_output="${log_dir}/${fname}.log"
  echo "================== $fname =================="
  awk '/SUMMARY/ {show=1; next} show' "$log_output"
done

exit 0

################################################################################
# Best streaming ASR models

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo
BASE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_asr_sil2_4nodes_reproduce-old_asr_la_d15_na2/inf_20250825/validation_logs/metadatas
for fname in ami.json earnings22.json gigaspeech.json ls_test_clean.json ls_test_other.json riva_asr_6p0.json spgispeech.json; do
  json_output="${BASE_DIR}/${fname}"
  log_output="${BASE_DIR}/${fname}.log"
  echo "Processing $json_output"
  python ${CODE_DIR}/scripts/speech_eval/wer.py \
    --json_file "$json_output" \
    --verbose 2>&1 | tee "$log_output" 
done

log_dir=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_asr_sil2_4nodes_reproduce-old_asr_la_d15_na2/inf_20250825/validation_logs/metadatas
for fname in ami.json earnings22.json gigaspeech.json ls_test_clean.json ls_test_other.json riva_asr_6p0.json spgispeech.json; do
  log_output="${log_dir}/${fname}.log"
  echo "================== $fname =================="
  awk '/SUMMARY/ {show=1; next} show' "$log_output"
done