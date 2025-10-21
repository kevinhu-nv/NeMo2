#!/bin/bash

export TORCH_HOME="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/Full-Duplex-Bench-NV

EXP_NAME=$1
CKPT_NAME=$2

pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/exp/${EXP_NAME}/inf_all/${CKPT_NAME}/validation_logs/metadatas/

pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_from_cc_20251003/exp_qwen7b/Qwen2.5-7B_SFT_adapter200m_from_step-27399-last_64gpu_5e-5_SFT0.1_TEXT0.1_QA0.02_t2t_loss0.0/results/inf/infer_qwen_7b_step_33407/validation_logs/metadatas

# Merge rank files into candor.json
cat ${pred_text_dir}/candor_rank0.json \
    ${pred_text_dir}/candor_rank1.json \
    ${pred_text_dir}/candor_rank2.json \
    ${pred_text_dir}/candor_rank3.json \
    ${pred_text_dir}/candor_rank4.json \
    ${pred_text_dir}/candor_rank5.json \
    ${pred_text_dir}/candor_rank6.json \
    ${pred_text_dir}/candor_rank7.json \
    > ${pred_text_dir}/candor.json

pred_text_file=${pred_text_dir}/candor.json



candor_dataset_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/candor_turn_taking

#####################
# Evaluate turn taking
python $CODE_DIR/evaluation/eval_smooth_turn_taking_text.py \
  --data_dir ${candor_dataset_dir} \
  --pred_text_file ${pred_text_file}