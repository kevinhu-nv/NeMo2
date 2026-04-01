#!/bin/bash

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/projects/nemo_s2s_merged_dec/NeMo_fc
SCRIPT=$CODE_DIR/scripts/speech_eval/eval_vab.py
VAB_REPO=/lustre/fsw/portfolios/llmservice/users/kevinhu/projects/VoiceAgentBench

export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HF_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HF_TOKEN=$(cat /lustre/fsw/portfolios/llmservice/users/kevinhu/tokens/huggingface_token)
export PYTHONPATH="${CODE_DIR}/scripts/speech_eval:${CODE_DIR}:${VAB_REPO}:${VAB_REPO}/voice_agent_bench:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/triton_cache"

# Required for VAB evaluators (GPT-4o-mini judge)
export OPENAI_API_KEY=$(cat /lustre/fsw/portfolios/llmservice/users/kevinhu/HFCACHE/zh_openai_key.txt)

MODEL="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

# ---------------------------------------------------------------------------
# Duplex STT inference output
# ---------------------------------------------------------------------------
# Accept args: eval_vab.sh [EXP_NAME] [STEP_NUM] [INF_PREFIX]
EXP_NAME=${1:-IAD_nano9b_parakeet600m_from_PT_32k_32gpu_5e-5_PT0.5_SFT0.15_QA0.02_TEXT0.1_loss0.5_MCQ0.03_prompt2_ASR0.01_fillerlong_offset2_sysp0.05_NoiseDefault_asr_dtc2_dst15_lossDefault_ei0.033_ot8_all_data_v3.2_fc0.1_fcnv30.1_fcref0.02_pf_ah}
STEP_NUM=${2:-7502}
INF_PREFIX=${3:-infer_nano_9b_vab}
RESULTS_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/projects/nemo_s2s_merged_dec/exp_SFT_9b/${EXP_NAME}/results/inf_fc_pf/${INF_PREFIX}_step_${STEP_NUM}/pad0_bos0_eos0
INFERENCE_DIR=${RESULTS_DIR}/validation_logs/metadatas

MODEL_NAME=$(echo $MODEL | tr '/' '-')
OUTPUT_DIR=${RESULTS_DIR}/vab_eval/${MODEL_NAME}

USE_GT_USER_TEXT=false

EXTRA_FLAGS=""
if [ "$USE_GT_USER_TEXT" = true ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --use_gt_user_text"
fi

# ---------------------------------------------------------------------------
# Run evaluation on all VAB subsets that have inference data
# ---------------------------------------------------------------------------
# Note: This requires:
# 1. VAB shar data converted and inference run on all VAB subsets
# 2. OPENAI_API_KEY set for GPT-4o-mini judge (PF and safety metrics)
# 3. pip install llama-index-llms-openai pydantic loguru

pip install llama-index-llms-openai pydantic loguru

# Single subset example:
# python3 ${SCRIPT} \
#     --inference_json "${INFERENCE_DIR}/single_tool_rank*.json" \
#     --subset single_tool \
#     --model ${MODEL} \
#     --output_dir ${OUTPUT_DIR} \
#     --vab_repo_dir ${VAB_REPO} \
#     --cache_dir ${HF_HOME} \
#     ${EXTRA_FLAGS}

# Evaluate VAB subsets (evaluator type auto-detected from subset name):
for SUBSET in single_tool single_tool_retrieval; do
    echo "=== Evaluating ${SUBSET} ==="
    set -x
    python3 ${SCRIPT} \
        --inference_dir ${INFERENCE_DIR} \
        --subset ${SUBSET} \
        --model ${MODEL} \
        --output_dir ${OUTPUT_DIR} \
        --vab_repo_dir ${VAB_REPO} \
        --cache_dir ${HF_HOME} \
        ${EXTRA_FLAGS}
    set +x
done
