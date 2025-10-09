#!/bin/sh

EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0 && CKPT_NAME=step-25339-last


CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo_20250923

bash ${CODE_DIR}/scripts/speech_eval/eval_candor_turn_taking_text.sh ${EXP_NAME} ${CKPT_NAME}
sleep 5
bash ${CODE_DIR}/scripts/speech_eval/eval_conv.sh ${EXP_NAME} ${CKPT_NAME}
sleep 5
bash ${CODE_DIR}/scripts/speech_eval/eval_intel.sh ${EXP_NAME} ${CKPT_NAME}
