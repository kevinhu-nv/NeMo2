#!/bin/bash

json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_and_candor_train_4nodes_repro_recipe2_cfttrain/inf/validation_logs/metadatas/candor.json'
json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_and_candor_4nodes_repro_recipe2_cft_0902/inf/validation_logs/metadatas/candor.json'
json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_and_candor_train_4nodes_repro_recipe2_cfttrain/inf_train/validation_logs/metadatas/candor.json'
json_output='/lustre/fsw/portfolios/convai/users/kevinhu/s2s/exp/DFW_qwen_1b_and_candor_4nodes_repro_recipe2_cft_0829/inf/validation_logs/metadatas/candor.json'

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo
python ${CODE_DIR}/scripts/speech_eval/empty_text.py \
  --json_file $json_output \
  --verbose