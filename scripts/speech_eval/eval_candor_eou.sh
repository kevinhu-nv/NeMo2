#!/bin/bash

export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"

# CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo
CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/Full-Duplex-Bench-NV

#####################
# Evaluate EOU

candor_dataset_dir="/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking"
jsonl_file=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_asr_sil2_4nodes_reproduce-old_asr_la_d6_na2_et12_eb2/inf/validation_logs/metadatas/candor.json
python $CODE_DIR/evaluation/eval_eou.py \
  --data_dir ${candor_dataset_dir} \
  --jsonl_file ${jsonl_file}