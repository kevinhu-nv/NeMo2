#!/bin/bash

export TORCH_HOME="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/Full-Duplex-Bench-NV

EXP_NAME=IAD_qwen_1b_and_pretrain_4nodes_repro_recipe2_preft1.0_sft0.0_noinit && CKPT_NAME=step-7239-last
# EXP_NAME="IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05"  # may have noise-augmentation
# EXP_NAME="IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1"
# EXP_NAME="IAD_qwen_1b_sft_pt_4nodes_pt0.9sft_0.05st0.05_na" && CKPT_NAME=step-6001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_nosa && CKPT_NAME=step-4001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft_0.05st0.05_na0.3_snr-30-6 && CKPT_NAME=step-6001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta2 && CKPT_NAME=step-14563-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60 && CKPT_NAME=step-5001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_snr0.5-50-80_ta2 && CKPT_NAME=step-7849-last
# EXP_NAME=IAD_qwen_1b_sft_pt_lr_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-24-15_lr5e-5 && CKPT_NAME=step-13726-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0 && CKPT_NAME=step-13003-last
# EXP_NAME=IAD_qwen_1b_sft_pt-bd200_4nodes_pt0.98sft0.01st0.01_na0.5_snr0.5-30-60_ptbd150 && CKPT_NAME=step-8169-last
# EXP_NAME=IAD_qwen_1b_sft_pt-bd200_4nodes_pt0.98sft0.01st0.01_na0.5_snr0.5-30-60_ptbd200 && CKPT_NAME=step-4001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-50-80 && CKPT_NAME=step-6553-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0 && CKPT_NAME=step-25339-last
EXP_NAME=$1
CKPT_NAME=$2

# pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/exp/${EXP_NAME}/inf/${CKPT_NAME}/validation_logs/metadatas/
pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/exp/${EXP_NAME}/inf_all/${CKPT_NAME}/validation_logs/metadatas/

pred_text_file=${pred_text_dir}/candor.json
candor_dataset_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/candor_turn_taking

#####################
# Evaluate turn taking
python $CODE_DIR/evaluation/eval_smooth_turn_taking_text.py \
  --data_dir ${candor_dataset_dir} \
  --pred_text_file ${pred_text_file}