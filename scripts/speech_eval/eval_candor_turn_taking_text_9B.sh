#!/bin/bash

export TORCH_HOME="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/Full-Duplex-Bench-NV

EXP_NAME=$1
STEP_NAME=$2
BOOST_NAME=$3
INF_NAME=$4
FTT=$5
PW=$6

# pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT-nano9b/${EXP_NAME}/results/inf/infer_nano_9b_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
# pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
# pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge_rebase2/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
# pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge_rebase/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
# pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/IAD_Nano-9B_SFT_Parakeet600m_asr_sp_from_checkpoints_hf_32002_64gpu_PT0.65_SFT0.05_QA0.02_TEXT0.1_MCQ0.03_ASR0.01_SP0.05-b200_LR5e-5_na0.5-30-60_ci_sa15_la2_tls10.0_facuda_val200.v2/results/inf/infer_nano_9b_step_6354/pad0_bos0_eos0/validation_logs/metadatas/
# pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/results/inf/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/ftt_${FTT}_pw${PW}_upad0_ubos0_ueos0/validation_logs/metadatas/
pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/results/inf_rebase/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
# pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_nov/exp_SFT-nano9b/${EXP_NAME}/results/inf/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
# pred_text_dir=/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/results/exp/nov25/exp_SFT_9b/${EXP_NAME}/results/inf_rebase/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
# pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_nov/exp_SFT-nano9b/${EXP_NAME}/results/inf_rebase_squash/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
# pred_text_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20260114/exp_SFT_9b/${EXP_NAME}/results/inf_rebase/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/

# Merge rank files into candor.json
cat ${pred_text_dir}/candor_rank{0..31}.json > ${pred_text_dir}/candor.json

pred_text_file=${pred_text_dir}/candor.json

candor_dataset_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/candor_turn_taking

#####################
# Evaluate turn taking
python $CODE_DIR/evaluation/eval_smooth_turn_taking_text.py \
  --data_dir ${candor_dataset_dir} \
  --pred_text_file ${pred_text_file}

echo "pred_text_file: ${pred_text_file}"