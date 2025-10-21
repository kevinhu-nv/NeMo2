#!/bin/bash
set -e

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo
GET_SCORES=$CODE_DIR/scripts/speech_eval/get_scores.sh

eval_single() {
    local eval_data=$1
    local file=$2

    if [ -f "$file" ]; then
        echo -e "\n Evaluating $eval_data ...\n"
        bash $GET_SCORES $eval_data $file text audio_path pred_text false
    else
        echo "Skipping $eval_data: file $file not found"
    fi
}

eval_voicebench() {
    BASE_DIR=$ROOT_DIR/${EXP_NAME}/inf_all_boost/${CKPT_NAME}/${INF_NAME}/validation_logs/metadatas

    # for d in alpacaeval commoneval openbookqa sdqa; do
    for d in alpacaeval commoneval openbookqa; do
        eval_single $d $BASE_DIR/${d}.json
    done
}

ROOT_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/exp

EXP_NAME=$1
CKPT_NAME=$2
INF_NAME=$3
eval_voicebench
