#!/bin/bash

export TORCH_HOME="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/Full-Duplex-Bench-NV

EXP_NAME=$1
STEP_NAME=$2
BOOST_NAME=$3
INF_NAME=$4

# pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/results/inf_rebase/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/

# Merge rank files into candor.json
cat ${pred_text_dir}/pause_handeling_rank{0..31}.json > ${pred_text_dir}/pause_handeling.json

pred_text_file=${pred_text_dir}/pause_handeling.json

pause_handling_dataset_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/candor_pause_handling

#####################
# Evaluate user interruption
python $CODE_DIR/evaluation/eval_pause_handling_text.py \
  --data_dir ${pause_handling_dataset_dir} \
  --pred_text_file ${pred_text_file}