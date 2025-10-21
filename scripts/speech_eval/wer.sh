#!/bin/bash

# json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_asr_sil2_1nodes_reproduce-old_asr_local/validation_logs/metadatas/ls_test_other.json'
# json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_asr_sil_4nodes_reproduce-old_asr_ra_d6_noscale/validation_logs/metadatas/riva_asr_6p0.json'
# json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_asr_sil2_4nodes_reproduce-old_asr_la_d15/validation_logs/metadatas/ls_test_clean.json'
# json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_infer_asr_1nodes_reproduce-old_asr_tinyllama/validation_logs/metadatas/inf.json'
# json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_convasr_4nodes_reproduce-old_convasr_sa_la_d4_0828/validation_logs/metadatas/ultrachat.json'
json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_convasr_4nodes_reproduce-old_convasr_sa_la_d4_08281419/validation_logs/metadatas/ultrachat.json'

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo
# python ${CODE_DIR}/scripts/speech_eval/wer.py \
#   --json_file $json_output \
#   --verbose

# exit 0

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo

# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa_15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-17507-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa_15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-22509-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-22508-last
EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa_15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-31513-last && INF_NAME="pad-1_bos0"
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_sil2_4nodes_sa_15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_allsftsil2 && CKPT_NAME=step-7573-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa6_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-6002-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_percep && CKPT_NAME=step-7678-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_asrf_4nodes_sa_15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_af-2 && CKPT_NAME=step-11495-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left4_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-10461-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_asrf_4nodes_sa_15_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_af-3 && CKPT_NAME=step-17005-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.0sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-15254-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.01sft0.04_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-10485-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_percep_nool && CKPT_NAME=step-7738-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.01sft0.04_fa_snr0.5-30-60_ta0_percep && CKPT_NAME=step-4501-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.01sft0.04_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-10485-last && INF_NAME=""
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left0_asrl3_txtl3_pt0.98asr0.005sft0.015_fa_snr0.5-30-60_ta0_is2s.v2 && CKPT_NAME=step-2636-last && INF_NAME=""
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.01sft0.04_fa_snr0.5-30-60_ta0_percep && CKPT_NAME=step-4501 && INF_NAME="pad-1_bos0"
EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.01sft0.04_fa_snr0.5-30-60_ta0_percep && CKPT_NAME=step-4501 && INF_NAME="pad-0.5_bos0"
EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.025sft0.025_fa_snr0.5-30-60_ta0_is2s && CKPT_NAME=step-15468-last && INF_NAME="pad-1_bos0"

LOG_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/exp/${EXP_NAME}/inf_all_boost/${CKPT_NAME}/${INF_NAME}"
BASE_DIR="${LOG_DIR}/validation_logs/metadatas"

TEST_SETS="ls_test_clean.json ls_test_other.json spgispeech.json gigaspeech.json earnings22.json ami.json tedlium.json voxpopuli.json"

TIMESTAMP=$(date +%Y%m%d_%H%M)
for fname in $TEST_SETS; do
  log_dir="${BASE_DIR}/${TIMESTAMP}"
  mkdir -p "$log_dir"
  json_output="${BASE_DIR}/${fname}"
  log_output="${log_dir}/${fname}.log"
  echo "Processing $json_output"
  python ${CODE_DIR}/scripts/speech_eval/wer.py \
    --json_file "$json_output" \
    --verbose 2>&1 | tee "$log_output" 
done

# TIMESTAMP=20251016_1701
for fname in $TEST_SETS; do
  log_dir="${BASE_DIR}/${TIMESTAMP}"
  log_output="${log_dir}/${fname}.log"
  echo "================== $fname =================="
  awk '/SUMMARY/ {show=1; next} show' "$log_output"
done

# Compute the average WER from all test set logs
total_wer=0
count=0
echo "================== Average WER over all sets =================="
for fname in $TEST_SETS; do
  log_dir="${BASE_DIR}/${TIMESTAMP}"
  log_output="${log_dir}/${fname}.log"
  # Extract WER: find SUMMARY line, then get 'WER' value from next non-empty line
  wer=$(awk '/Normalized WER \(weighted\):/ {for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+$/) {print $i; exit}}' "$log_output")
  if [[ -n "$wer" ]]; then
    total_wer=$(awk "BEGIN {print $total_wer + $wer}")
    count=$((count + 1))
  fi
done
if [[ $count -gt 0 ]]; then
  avg_wer=$(awk "BEGIN {printf \"%.4f\", $total_wer / $count}")
  echo "Average WER across $count sets: $avg_wer"
else
  echo "No WER results found to average."
fi

exit 0

################################################################################

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