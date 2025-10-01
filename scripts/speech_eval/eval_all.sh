#!/bin/sh

# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0 && CKPT_NAME=step-25339-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1_na0.5_snr0.5-6-60 && CKPT_NAME=step-6001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1_na0.5_snr0.5-30-60 && CKPT_NAME=step-6001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1_na0.5_snr0.5-30-60_ta0 && CKPT_NAME=step-9598-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1_na0.3_snr0.3-6-30_us0.5-24-15 && CKPT_NAME=step-16659-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0_percep && CKPT_NAME=step-7001-last
# EXP_NAME=IAD_qwen_1b_sft_pt-bd200_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0_percep_hdd0_ptbd200 && CKPT_NAME=step-10932-last
# EXP_NAME=IAD_qwen_1b_sft_pt-bd200_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0_percep_hdd4_ptbd200 && CKPT_NAME=step-11358-last
EXP_NAME=IAD_qwen_1b_convasr_joint_4nodes_left4_asrl3_txtl3_initsa-hdd4s2sta0 && CKPT_NAME=step-10726-last


CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo

bash ${CODE_DIR}/scripts/speech_eval/eval_candor_turn_taking_text.sh ${EXP_NAME} ${CKPT_NAME}
sleep 5
bash ${CODE_DIR}/scripts/speech_eval/eval_conv.sh ${EXP_NAME} ${CKPT_NAME}
sleep 5
bash ${CODE_DIR}/scripts/speech_eval/eval_intel.sh ${EXP_NAME} ${CKPT_NAME}
