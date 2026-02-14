#!/bin/bash

export TORCH_HOME="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/Full-Duplex-Bench-NV

pred_text_dir=${1}
pred_text_file=${pred_text_dir}/candor_turn_taking.jsonl

candor_dataset_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/candor_turn_taking

#####################
# Evaluate turn taking
python $CODE_DIR/evaluation/eval_smooth_turn_taking_text.py \
  --data_dir ${candor_dataset_dir} \
  --pred_text_file ${pred_text_file}

echo "pred_text_file: ${pred_text_file}"