#!/bin/bash
set -e

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo
GET_SCORES=$CODE_DIR/scripts/speech_eval/get_scores.sh

eval_single() {
    local eval_data=$1
    local file=$2

    if [ -f "$file" ]; then
        echo -e "\n Evaluating $eval_data ...\n"
        # Arguments:
        #   eval_data, inference_output_path, modality, id_key, pred_text_key, force_rewrite
        bash $GET_SCORES $eval_data $file text audio_path pred_text false
    else
        echo "Skipping $eval_data: file $file not found"
    fi
}

eval_voicebench() {
    BASE_DIR=$ROOT_DIR/${EXP_NAME}/inf_all/${CKPT_NAME}/validation_logs/metadatas

    # for d in alpacaeval commoneval openbookqa sdqa; do
    for d in alpacaeval commoneval openbookqa; do
    # for d in sdqa; do
        eval_single $d $BASE_DIR/${d}.json
    done
}

ROOT_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/exp

# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60 && CKPT_NAME=step-5001-last
# EXP_NAME="IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05" && CKPT_NAME=step-7084-last
# EXP_NAME="IAD_qwen_1b_sft_pt_4nodes_pt0.9sft_0.05st0.05_na" && CKPT_NAME=step-6001-last
# EXP_NAME="IAD_qwen_1b_and_pretrain_4nodes_repro_recipe2_preft1.0_sft0.0_noinit" && CKPT_NAME=step-7239-last
# EXP_NAME="IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1" && CKPT_NAME=step-6001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0 && CKPT_NAME=step-13003-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1_na0.5_snr0.5-6-60 && CKPT_NAME=step-6001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1_na0.5_snr0.5-30-60 && CKPT_NAME=step-6001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0 && CKPT_NAME=step-25339-last

EXP_NAME=$1
CKPT_NAME=$2
eval_voicebench
