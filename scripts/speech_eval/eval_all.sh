#!/bin/sh

# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1_na0.5_snr0.5-6-60 && CKPT_NAME=step-6001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1_na0.5_snr0.5-30-60 && CKPT_NAME=step-6001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1_na0.5_snr0.5-30-60_ta0 && CKPT_NAME=step-9598-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1_na0.3_snr0.3-6-30_us0.5-24-15 && CKPT_NAME=step-16659-last
# EXP_NAME=IAD_qwen_1b_sft_pt-bd200_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0_percep_hdd0_ptbd200 && CKPT_NAME=step-10932-last
# EXP_NAME=IAD_qwen_1b_sft_pt-bd200_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0_percep_hdd4_ptbd200 && CKPT_NAME=step-11358-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_4nodes_left4_asrl3_txtl3_initsa-hdd4s2sta0 && CKPT_NAME=step-10726-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa_left4_asrl3_txtl3_pt0.45asr0.45sft0.05 && CKPT_NAME=step-21304-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa_left4_asrl3_txtl3_pt0.9asr0.05sft0.05_fa && CKPT_NAME=step-10543-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa_15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-17507-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa_15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-22509-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-22508-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_sil2_4nodes_sa_15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_allsftsil2 && CKPT_NAME=step-7573-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa6_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-6002-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_percep && CKPT_NAME=step-20304-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left4_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-10461-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.045sft0.005_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-7502-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa_15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-31513-last
EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa_15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-31513-last && INF_NAME="pad-1_bos0"
EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa_15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-31513-last && INF_NAME="pad-2_bos1"
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_asrf_4nodes_sa_15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_af-2 && CKPT_NAME=step-11495-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_asrf_texta_4nodes_sa_15_left8_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_af-3_ta && CKPT_NAME=step-11261-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_asrf_4nodes_sa_15_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_af-3 && CKPT_NAME=step-17005-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_percep && CKPT_NAME=step-10443-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa_15_left6_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-15123-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa6_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-10405-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_percep_nool && CKPT_NAME=step-2540-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_percep && CKPT_NAME=step-10443-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.45asr0.1sft0.45_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-12005-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.0sft0.05_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-15254-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_is2s && CKPT_NAME=step-2577-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_usft_4nodes_sa15_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_usft && CKPT_NAME=step-2635-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.01sft0.04_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-10485-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_percep_nool && CKPT_NAME=step-7738-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_asrfix && CKPT_NAME=step-5502-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.01sft0.04_fa_snr0.5-30-60_ta0_percep && CKPT_NAME=step-4501-last
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left0_asrl3_txtl3_pt0.98asr0.005sft0.015_fa_snr0.5-30-60_ta0_is2s.v2 && CKPT_NAME=step-2636-last && INF_NAME=pad-1_bos0
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.01sft0.04_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-10485-last && INF_NAME=""
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left0_asrl3_txtl3_pt0.98asr0.005sft0.015_fa_snr0.5-30-60_ta0_is2s.v2 && CKPT_NAME=step-2636-last && INF_NAME=""
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left0_asrl3_txtl3_pt0.95asr0.01sft0.04_fa_snr0.5-30-60_ta2_percep.v2 && CKPT_NAME=step-2576-last && INF_NAME=""
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.01sft0.04_fa_snr0.5-30-60_ta0_percep && CKPT_NAME=step-4501 && INF_NAME="pad-1_bos0"
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.01sft0.04_fa_snr0.5-30-60_ta0 && CKPT_NAME=step-10485-last && INF_NAME="pad-1_bos0"
EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left0_asrl3_txtl3_pt0.98asr0.005sft0.015_fa_snr0.5-30-60_ta0_is2s.v2 && CKPT_NAME=step-16773-last && INF_NAME=pad-0.5_bos0
EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.9asr0.05sft0.05_fa_snr0.5-30-60_ta0_is2s && CKPT_NAME=step-11692-last && INF_NAME="pad-0.5_bos0"
EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.025sft0.025_fa_snr0.5-30-60_ta0_is2s && CKPT_NAME=step-13505-last && INF_NAME="pad-0.5_bos0"
# EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.025sft0.025_fa_snr0.5-30-60_ta0_is2s && CKPT_NAME=step-15468-last && INF_NAME="pad-1_bos0"
EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.01sft0.04_fa_snr0.5-30-60_ta0_percep && CKPT_NAME=step-4501 && INF_NAME="pad-0.5_bos0"
EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.01sft0.04_fa_snr0.5-30-60_ta0_percep && CKPT_NAME=step-4501 && INF_NAME="pad-1_bos0"
EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0 && CKPT_NAME=step-25339-last && INF_NAME=""
EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0_percep && CKPT_NAME=step-7001-last && INF_NAME=""
EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.025sft0.025_fa_snr0.5-30-60_ta0_is2s && CKPT_NAME=step-20151-last && INF_NAME="pad-2_bos1"
EXP_NAME=IAD_qwen_1b_convasr_joint_pt-bd200_4nodes_sa15_left2_asrl3_txtl3_pt0.95asr0.025sft0.025_fa_snr0.5-30-60_ta0_is2s && CKPT_NAME=step-20151-last && INF_NAME="pad-2_bos1_eos1"

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo

bash ${CODE_DIR}/scripts/speech_eval/eval_candor_turn_taking_text.sh ${EXP_NAME} ${CKPT_NAME} ${INF_NAME}
sleep 2
bash ${CODE_DIR}/scripts/speech_eval/eval_conv.sh ${EXP_NAME} ${CKPT_NAME} ${INF_NAME}
sleep 2
bash ${CODE_DIR}/scripts/speech_eval/eval_intel.sh ${EXP_NAME} ${CKPT_NAME} ${INF_NAME}
