#!/bin/bash

export TORCH_HOME="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/Full-Duplex-Bench-NV

EXP_NAME=$1
CKPT_NAME=$2

pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/exp/${EXP_NAME}/inf_all/${CKPT_NAME}/validation_logs/metadatas/

pred_text_file=${pred_text_dir}/candor.json



candor_dataset_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/candor_turn_taking

#####################
# Evaluate turn taking
python $CODE_DIR/evaluation/eval_smooth_turn_taking_text.py \
  --data_dir ${candor_dataset_dir} \
  --pred_text_file ${pred_text_file}