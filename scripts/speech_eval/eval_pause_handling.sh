#!/bin/bash

export TORCH_HOME="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/Full-Duplex-Bench-NV

pred_text_dir=${1}

pred_text_file=${pred_text_dir}/candor_pause_handling.jsonl
pause_handling_dataset_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/candor_pause_handling

#####################
# Evaluate user interruption
python $CODE_DIR/evaluation/eval_pause_handling_text.py \
  --data_dir ${pause_handling_dataset_dir} \
  --pred_text_file ${pred_text_file}