#!/bin/sh

EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0 && CKPT_NAME=step-25339-last
# TEST_NAME=elena_20251016
TEST_NAME=kevin_20251017

# INF_NAME="pad0_bos0_eos0"
# INF_NAME="pad-0.5_bos0_eos0"
INF_NAME="pad-0.5_bos0_eos0_hyptokens"
# INF_NAME="pad-1_bos0_eos0"
# INF_NAME="pad-2_bos1_eos1"
# INF_NAME="pad-3_bos2_eos2"
# INF_NAME="pad-4_bos3_eos3"
# INF_NAME="pad-5_bos4_eos4"
# INF_NAME="pad-6_bos4_eos4"

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo

set -x
bash ${CODE_DIR}/scripts/speech_eval/eval_conv.sh ${EXP_NAME} ${CKPT_NAME} ${INF_NAME} ${TEST_NAME} 1.5 1.5
set +x
