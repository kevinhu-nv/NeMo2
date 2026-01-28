#!/bin/bash

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo
GET_SCORES=$CODE_DIR/scripts/speech_eval/get_scores.sh

eval_single() {
    local eval_data=$1
    local file=$2

    if [ -f "$file" ]; then
        echo -e "\n Evaluating $eval_data ...\n"
        # Arguments:
        #   eval_data, inference_output_path, modality, id_key, pred_text_key, force_rewrite
        # bash $GET_SCORES $eval_data $file text audio_path pred_text false
        bash $GET_SCORES $eval_data $file text id pred_text false
    else
        echo "Skipping $eval_data: file $file not found"
    fi
}

eval_voicebench() {
    BASE_DIR=$ROOT_DIR/${EXP_NAME}/results/inf/infer_nano_9b_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas
    # BASE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
    BASE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
    # BASE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge_rebase2/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
    # BASE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/IAD_Nano-9B_SFT_Parakeet600m_asr_sp_from_checkpoints_hf_32002_64gpu_PT0.65_SFT0.05_QA0.02_TEXT0.1_MCQ0.03_ASR0.01_SP0.05-b200_LR5e-5_na0.5-30-60_ci_sa15_la2_tls10.0_facuda_val200.v2/results/inf/infer_nano_9b_step_6354/pad0_bos0_eos0/validation_logs/metadatas
    # BASE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/results/inf/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
    BASE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/results/inf_rebase/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
    # BASE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_nov/exp_SFT-nano9b/${EXP_NAME}/results/inf/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
    # BASE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/results/exp/nov25/exp_SFT_9b/${EXP_NAME}/results/inf_rebase/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
    # BASE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_nov/exp_SFT-nano9b/${EXP_NAME}/results/inf_rebase_squash/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/
    # BASE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20260114/exp_SFT_9b/${EXP_NAME}/results/inf_rebase/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/metadatas/

    # for d in alpacaeval commoneval openbookqa sdqa; do
    for d in alpacaeval commoneval openbookqa; do
    # for d in sdqa; do
    # for d in advbench; do
        # Merge rank files into a single json for each dataset
        if [ -f "$BASE_DIR/${d}_rank0.json" ]; then
            cat $BASE_DIR/${d}_rank{0..31}.json > $BASE_DIR/${d}.json
        fi
        eval_single $d $BASE_DIR/${d}.json
    done
}

# ROOT_DIR=/lustre/fsw/convai_convaird_nemo-speech/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT_9b
# ROOT_DIR=/lustre/fsw/llmservice_nemo_speechlm/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT_9b
ROOT_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT-nano9b

EXP_NAME=$1
STEP_NAME=$2
BOOST_NAME=$3
INF_NAME=$4

eval_voicebench
