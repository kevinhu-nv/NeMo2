#!/bin/bash

export TORCH_HOME="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/Full-Duplex-Bench-NV

pred_text_dir=${1}

pred_text_file=${pred_text_dir}/synthetic_user_interruption.jsonl

user_interruption_dataset_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/synthetic_user_interruption

#####################
# Evaluate user interruption
python $CODE_DIR/evaluation/eval_user_interruption_text.py \
  --data_dir ${user_interruption_dataset_dir} \
  --pred_text_file ${pred_text_file} \
  --write_outputs \
  --llm_judge gpt-4-turbo \
  --openai_key_file /lustre/fsw/portfolios/llmservice/users/kevinhu/HFCACHE/zh_openai_key.txt \
  --verbose