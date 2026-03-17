#!/bin/bash

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/projects/nemo_s2s_merged_dec/NeMo_fc
SCRIPT=$CODE_DIR/scripts/speech_eval/eval_fc_toolcall.py

export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HF_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HF_TOKEN=$(cat /lustre/fsw/portfolios/llmservice/users/kevinhu/tokens/huggingface_token)
export PYTHONPATH="${CODE_DIR}/scripts/speech_eval:${CODE_DIR}:${PYTHONPATH}"

MODEL="Qwen/Qwen2.5-7B-Instruct"
# MODEL="Qwen/Qwen2.5-72B-Instruct"
# MODEL="nvidia/Nemotron-Mini-4B-Instruct"
# MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# ---------------------------------------------------------------------------
# Duplex STT inference output (from model validation)
# ---------------------------------------------------------------------------
# EXP_NAME=IAD_nano9b_parakeet600m_from_SFT_feb20_32gpu_5e-5_PT0.5_SFT0.15_QA0.02_TEXT0.1_loss0.5_MCQ0.03_prompt2_ASR0.01_fillerlong_offset2_sysp0.05_NoiseDefault_asr_dtc2_dst15_lossDefault_ei0.033_ot8_all_data_v3.2_fc0.5_fcnv30.5
# STEP_NUM=2502
# EXP_NAME=IAD_nano9b_parakeet600m_from_SFT_feb20_32gpu_5e-5_PT0.5_SFT0.15_QA0.02_TEXT0.1_loss0.5_MCQ0.03_prompt2_ASR0.01_fillerlong_offset2_sysp0.05_NoiseDefault_asr_dtc2_dst15_lossDefault_ei0.033_ot8_all_data_v3.2_fc0.05_fcnv30.05
# STEP_NUM=4501
EXP_NAME=IAD_nano9b_parakeet600m_from_SFT_feb20_32gpu_5e-5_PT0.5_SFT0.15_QA0.02_TEXT0.1_loss0.5_MCQ0.03_prompt2_ASR0.01_fillerlong_offset2_sysp0.05_NoiseDefault_asr_dtc2_dst15_lossDefault_ei0.033_ot8_all_data_v3.2_fc0.01_fcnv30.01
STEP_NUM=9030

RESULTS_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/projects/nemo_s2s_merged_dec/exp_SFT_9b/${EXP_NAME}/results/inf_fc/infer_nano_9b_bfclv3_step_${STEP_NUM}/pad0_bos0_eos0
MODEL_NAME=$(echo $MODEL | tr '/' '-')
OUTPUT_DIR=${RESULTS_DIR}/fc_eval/${MODEL_NAME}

for SUBSET in bfcl_v3_simple bfcl_v3_multiple bfcl_v3_parallel bfcl_v3_parallel_multiple bfcl_v3_irrelevance; do
    echo "=== Evaluating $SUBSET ==="
    INFERENCE_JSON="${RESULTS_DIR}/validation_logs/metadatas/${SUBSET}_rank*.json"

    USE_GT_USER_TEXT=false
    APPLY_ITN=true
    NORMALIZE=true

    SUFFIX=""
    EXTRA_FLAGS=""
    if [ "$USE_GT_USER_TEXT" = true ]; then
        SUFFIX="${SUFFIX}_gt"
        EXTRA_FLAGS="${EXTRA_FLAGS} --use_gt_user_text"
    fi
    if [ "$APPLY_ITN" = true ]; then
        SUFFIX="${SUFFIX}_itn"
        EXTRA_FLAGS="${EXTRA_FLAGS} --apply_itn"
    fi
    if [ "$NORMALIZE" = true ]; then
        SUFFIX="${SUFFIX}_norm"
        EXTRA_FLAGS="${EXTRA_FLAGS} --normalize"
    fi
    if [ -z "$SUFFIX" ]; then
        SUFFIX="_raw"
    fi
    OUTPUT_JSONL=${OUTPUT_DIR}/${SUBSET}_inference_eval${SUFFIX}.jsonl

    set -x
    python3 ${SCRIPT} --inference_json "${INFERENCE_JSON}" --model ${MODEL} --output_jsonl ${OUTPUT_JSONL} ${EXTRA_FLAGS}
    set +x
done

# ---------------------------------------------------------------------------
# BFCL v3 audio — all subsets (Lhotse shar, ground-truth text eval)
# ---------------------------------------------------------------------------
# SHAR_BASE=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/bfcl_v3_audio_shars
# SHAR_OUTPUT_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/fc_eval

# for SUBSET in BFCL_v3_simple BFCL_v3_multiple BFCL_v3_parallel BFCL_v3_parallel_multiple BFCL_v3_irrelevance; do
#     echo "=== Evaluating $SUBSET ==="
#     SHAR_OUTPUT_JSONL=${SHAR_OUTPUT_DIR}/${SUBSET}_eval.jsonl
#     set -x
#     python3 ${SCRIPT} \
#         --shar_path ${SHAR_BASE}/${SUBSET} \
#         --model ${MODEL} \
#         --output_jsonl ${SHAR_OUTPUT_JSONL}
#     set +x
# done

# ---------------------------------------------------------------------------
# BFCL v4 simple (Python) benchmark — text only, commented out
# ---------------------------------------------------------------------------
# GORILLA_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/projects/gorilla
# BFCL_DATA=${GORILLA_DIR}/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json
# BFCL_ANSWER=${GORILLA_DIR}/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json
# OUTPUT_JSONL=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/fc_eval/bfcl_simple_python_eval.jsonl
# python3 ${SCRIPT} --bfcl_data ${BFCL_DATA} --bfcl_answer ${BFCL_ANSWER} --model ${MODEL} --output_jsonl ${OUTPUT_JSONL}

# ---------------------------------------------------------------------------
# Glaive FC test set (Lhotse shar) — commented out
# ---------------------------------------------------------------------------
# SHAR_PATH=/lustre/fsw/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/ALM/SFT/brainy_mantis/duplex_s2s_shars/overlap_0.00/even_toolcall_response/glaive-functioncalling-v2+toolcall+respond_test
# OUTPUT_JSONL=/lustre/fsw/portfolios/llmservice/users/kevinhu/data/fc_eval/glaive_test_fc_eval.jsonl
# python3 ${SCRIPT} --shar_path ${SHAR_PATH} --model ${MODEL} --output_jsonl ${OUTPUT_JSONL}
