#!/bin/bash

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo
GET_SCORES=$CODE_DIR/scripts/speech_eval/get_scores.sh

BASE_DIR=${1}

eval_single() {
    local eval_data=$1
    local file=$2

    if [ -f "$file" ]; then
        echo -e "\n Evaluating $eval_data ...\n"
        # Arguments:
        #   eval_data, inference_output_path, modality, id_key, pred_text_key, force_rewrite
        # bash $GET_SCORES $eval_data $file text audio_path pred_text false
        bash $GET_SCORES $eval_data $file text audio_path pred_text false
    else
        echo "Skipping $eval_data: file $file not found"
    fi
}

eval_voicebench() {
    for d in commoneval openbookqa; do
        eval_single $d $BASE_DIR/${d}.jsonl
    done
}

eval_voicebench
