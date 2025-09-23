#!/bin/bash

## Usage: bash score_voicebench.sh <eval_data> <jsonl_path> <force_rewrite> <llm_judge> <api_key> <api>
# eval_data: ifeval, bbh, advbench, mmsu, openbookqa, alpacaeval, alpacaeval_full, commoneval, wildvoice, sdqa
# jsonl_path: path to the jsonl file
# force_rewrite: true or false
# llm_judge: gpt-4.1-20250414 or gpt-4o-mini-20240718
# api_key: API key for authentication.
# api: nv_oneapi or openai

eval_data=$1
jsonl_path=$2
force_rewrite=$3
llm_judge=$4
api_key=$5
api=$6

set -e

# CODE_DIR=/lustre/fsw/portfolios/convai/users/apasad/nemo_duplex_s2s/code/VoiceBench
CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/eval-intelligence/packages/VoiceBench_a214ff2_072525

export PYTHONPATH=$CODE_DIR:$PYTHONPATH

###################################################
# STEP A: Evaluate with GPT for alpacaeval, alpacaeval_full, commoneval, wildvoice, and sdqa datasets
###################################################

cd $CODE_DIR

if [[ "$eval_data" == "alpacaeval" || "$eval_data" == "alpacaeval_full" || "$eval_data" == "commoneval" || "$eval_data" == "sdqa" || "$eval_data" == "wildvoice" ]]; then
    # Check if results already exist and force_rewrite is false
    eval_file="${jsonl_path%.jsonl}_${llm_judge}_scores.jsonl"
    if [[ "$force_rewrite" != "true" && -f "$eval_file" ]]; then
        eval_file_basename=$(basename "$eval_file")
        echo "Results already exist at $eval_file_basename. Skipping evaluation with $llm_judge. Set force_rewrite to true to overwrite."
        PRINT_STR="============== Final results using existing scores with $llm_judge for $eval_data =============="
    else
        echo "Evaluating with $llm_judge using $api API key"
        if [[ "$api" == "nv_oneapi" ]]; then
            python api_judge.py --src_file $jsonl_path --llm_judge $llm_judge --nv_oneapi_key $api_key
        else
            python api_judge.py --src_file $jsonl_path --llm_judge $llm_judge --openai_key $api_key
        fi
        PRINT_STR="============== Final results using $llm_judge for $eval_data =============="
    fi
else
    eval_file=${jsonl_path}
    PRINT_STR="============== Final results for $eval_data =============="
fi

###################################################
# STEP B: Score with official voicebench eval
###################################################
    
# Set evaluator
if [[ "$eval_data" == "alpacaeval" || "$eval_data" == "alpacaeval_full" || "$eval_data" == "commoneval" || "$eval_data" == "wildvoice" ]]; then
    evaluator=open
elif [[ "$eval_data" == "sdqa" ]]; then
    evaluator=qa
elif [[ "$eval_data" == "advbench" ]]; then
    evaluator=harm
elif [[ "$eval_data" == "openbookqa" || "$eval_data" == "mmsu" ]]; then
    evaluator=mcq
elif [[ "$eval_data" == "ifeval" ]]; then
    evaluator=ifeval
elif [[ "$eval_data" == "bbh" ]]; then
    evaluator=bbh
fi

echo -e "\nScoring $eval_data with $evaluator evaluator"

echo -e "\n\n$PRINT_STR"
python evaluate.py --src_file $eval_file --evaluator $evaluator --data_name $eval_data
