#!/bin/bash

## This script is used to get the scores for a given inference output jsonl.
## Currently supported eval_data:
# Voicebench: openbookqa, alpacaeval, commoneval, sdqa, mmsu
# URO-bench: gaokaoeval, summary
# llama_questions, triviaqa, web_questions

## Example usage:
# bash get_scores.sh \
#   alpacaeval \
#   /lustre/fsw/portfolios/convai/users/apasad/nemo_duplex_s2s_inf/results/dfw_t2t_all_data_lora_dfw_2node_16gpus_1e-4lr_1000warmup_64loradim/validation_outputs_step7501_bs32/val_alpacaeval_outputs.jsonl \
#   text \
#   data_id \
#   pred_agent

# set your openai or oneapi key here if eval_data is alpacaeval, alpacaeval_full, commoneval, wildvoice or sdqa
# API_KEY=$(cat /lustre/fsw/portfolios/convai/users/apasad/HFCACHE/yifan_oneapi_key.txt)
# API="nv_oneapi"
API_KEY=$(cat /lustre/fsw/portfolios/convai/users/kevinhu/HFCACHE/zh_openai_key.txt)
API="openai"

set -e

eval_data=$1
inference_output_path=$2
modality=$3 # defaults to text
id_key=$4 # defaults to data_id
pred_text_key=$5 # defaults to pred_agent
force_rewrite=$6 # defaults to false; avoids re-running GPT scoring if results already exist
ref_text_key=$7 # provide this if you don't want the script to read reference answer text from the manifest

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/eval-intelligence

echo
# if key mappings are not provided, use defaults
if [ -z "$pred_text_key" ]; then
    echo "Warning: pred_text_key mapping not provided, defaulting to pred_agent"
    pred_text_key="pred_agent"
fi

if [ -z "$id_key" ]; then
    echo "Warning: id_key mapping not provided, defaulting to data_id"
    id_key="data_id"
fi

# if modality not provided, default to text
if [ -z "$modality" ]; then
    echo "Warning: modality not provided, defaulting to text"
    modality="text"
fi

if [ -z "$force_rewrite" ]; then
    echo "Warning: force_rewrite not provided, defaulting to false"
    force_rewrite=false
fi

# check if eval_data is supported
case "$eval_data" in
    openbookqa|mmsu|alpacaeval|commoneval|sdqa|llama_questions|triviaqa|web_questions|gaokaoeval|summary)
        ;;
    *)
        echo "Error: Unsupported eval_data: $eval_data, please choose from openbookqa, alpacaeval, commoneval, sdqa, llama_questions, triviaqa, web_questions, mmsu, gaokaoeval, summary"
        exit 1
        ;;
esac

# check if json_path exists
if [ ! -f "$inference_output_path" ]; then
    echo "Error: inference_output_path $inference_output_path does not exist"
    exit 1
fi

# check if modality is supported
if [ "$modality" != "text" ] && [ "$modality" != "speech" ]; then
    echo "Error: modality $modality is not supported, please choose from text or speech"
    exit 1
fi

# check if force_rewrite is a valid boolean
if [ "$force_rewrite" != "true" ] && [ "$force_rewrite" != "false" ]; then
    echo "Error: force_rewrite $force_rewrite is not a valid boolean"
    exit 1
fi

# Activate conda environment
. "/lustre/fsw/portfolios/convai/users/yifanp/miniforge3/etc/profile.d/conda.sh"
conda activate voicebench

if [ -z "$score_dir" ]; then
    # Create directory in the same location as inference_output_path
    inference_dir=$(dirname "$inference_output_path")
    score_dir="${inference_dir}/score_qa_intelligence"
    echo
    echo "NOTE: Intermediate outputs and scores will be saved in $score_dir/"
    mkdir -p "$score_dir"
    cp $inference_output_path $score_dir/ # saving the original inference outputs for reference
    inference_output_path=$score_dir/$(basename "${inference_output_path}")
fi

###################################################
# STEP 1: Reformat inference outputs into a standard format for scoring
###################################################
echo -e "\n\n============== STEP 1: Reformatting inference outputs for scoring =============="
SCRIPT="${CODE_DIR}/tools/reformat_inference_out.py"
# Remove .json or .jsonl extension and add new suffix
base_filename=$(basename "${inference_output_path}")
base_filename=${base_filename%.json}
base_filename=${base_filename%.jsonl}
if [ "$eval_data" == "sdqa" ]; then
    echo "NOTE: Using only usa split samples for $eval_data; processing 553 instead of 6083 samples"
    jsonl_path="${score_dir}/${base_filename}_reformat_${modality}_usa_output.jsonl"
else
    jsonl_path="${score_dir}/${base_filename}_reformat_${modality}_output.jsonl"
fi

# NOTE: You can add --audio_path_key arguments if you want to include path to
#       generated audio files in the output

# NOTE: You can add --ref_text_key arguments if you want to read reference
#       from inference outputs

# Build the command with required arguments
cmd="python $SCRIPT \
    --input_jsonl $inference_output_path \
    --output_jsonl $jsonl_path \
    --data_name $eval_data \
    --modality $modality \
    --id_key $id_key \
    --pred_text_key $pred_text_key \
    --manifest_dir $CODE_DIR/manifest_files"

# Add optional ref_text_key if it's set
if [ -n "$ref_text_key" ]; then
    cmd="$cmd --ref_text_key $ref_text_key"
fi

# Execute the command
eval $cmd


###################################################
# STEP 2: Score
###################################################
echo -e "\n\n\033[0;32m============== STEP 2: Scoring $eval_data ==============\033[0m"

case "$eval_data" in
    mmsu|openbookqa|alpacaeval|commoneval|sdqa)
        # llm_judge is used for alpacaeval, alpacaeval_full, commoneval, wildvoice, sdqa
        if [ "$API" == "openai" ]; then
            llm_judge="gpt-4o-mini" # same as official voicebench eval
        else
            llm_judge="gpt-4o-mini-20240718" # "gpt-4.1-20250414"
        fi

        SCRIPT="$CODE_DIR/tools/score_voicebench.sh"
        bash $SCRIPT $eval_data $jsonl_path $force_rewrite $llm_judge $API_KEY $API
        ;;
esac

case "$eval_data" in
    gaokaoeval|summary|triviaqa|llama_questions|web_questions)
        SCRIPT="$CODE_DIR/tools/score_with_llm.py"
        cmd="python $SCRIPT --src_file $jsonl_path --dataset_name $eval_data"
        
        if [ "$API" == "openai" ]; then
            llm_judge="gpt-4o-mini" # same as official uro-bench eval
            cmd="$cmd --llm_judge $llm_judge"
            cmd="$cmd --openai_key $API_KEY"
        else
            llm_judge="gpt-4o-mini-20240718" # "gpt-4.1-20250414"
            cmd="$cmd --llm_judge $llm_judge"
            cmd="$cmd --nv_oneapi_key $API_KEY"
        fi
        if [ "$force_rewrite" == "true" ]; then
            cmd="$cmd --force_rewrite"
        fi
        eval $cmd
        ;;
esac

case "$eval_data" in
    triviaqa|llama_questions|web_questions)
        SCRIPT="$CODE_DIR/tools/eval_qa_acc.py"
        python $SCRIPT --src_file $jsonl_path --dataset_name $eval_data
        ;;
esac
