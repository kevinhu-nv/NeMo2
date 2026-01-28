#!/bin/sh

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo

# EXP_NAME=Qwen2.5-7B_SFT_adapter200m_from_step-27399-last_64gpu_5e-5_SFT0.1_TEXT0.1_QA0.02_t2t_loss0.0 && STEP_NAME=33407
# EXP_NAME=Qwen2.5-7B_SFT_ParakeetEnc600m_from_step-30002_64gpu_5e-5_SFT0.15_TEXT0.2_QA0.02_t2t_loss0.2_MCQ0.03 && STEP_NAME=48005
# EXP_NAME=Qwen2.5-7B_SFT_ParakeetEnc600m_noise_aug_from_step-24002_64gpu_5e-5_SFT0.30_TEXT0.2_QA0.02_t2t_loss0.2_MCQ0.03 && STEP_NAME=36004

EXP_NAME=${1}
STEP_NAME=${2}
BOOST_NAME=${3}
INF_NAME=${4:-inf}
DATASET_NAME=${5:-real_demo}
FTT=${6:-False}
PW=${7:-1}

set -x
# STT eval
bash ${CODE_DIR}/scripts/speech_eval/eval_candor_turn_taking_text_9B.sh ${EXP_NAME} ${STEP_NAME} ${BOOST_NAME} ${INF_NAME} ${FTT} ${PW}
# bash ${CODE_DIR}/scripts/speech_eval/eval_conv_9B.sh ${EXP_NAME} ${STEP_NAME} ${BOOST_NAME} ${INF_NAME} ${DATASET_NAME} ${FTT} ${PW}
# bash ${CODE_DIR}/scripts/speech_eval/eval_intel_text_9B.sh ${EXP_NAME} ${STEP_NAME} ${BOOST_NAME} ${INF_NAME}
# bash ${CODE_DIR}/scripts/speech_eval/eval_user_interruption_text_9B.sh ${EXP_NAME} ${STEP_NAME} ${BOOST_NAME} ${INF_NAME} ${FTT} ${PW}
# bash ${CODE_DIR}/scripts/speech_eval/eval_pause_handling_text_9B.sh ${EXP_NAME} ${STEP_NAME} ${BOOST_NAME} ${INF_NAME} ${FTT} ${PW}

# S2S eval
# bash ${CODE_DIR}/scripts/speech_eval/eval_candor_turn_taking_9B.sh ${EXP_NAME} ${STEP_NAME} ${BOOST_NAME} ${INF_NAME} ${FTT} ${PW}
# bash ${CODE_DIR}/scripts/speech_eval/eval_intel_9B.sh ${EXP_NAME} ${STEP_NAME} ${BOOST_NAME} ${INF_NAME}
# bash ${CODE_DIR}/scripts/speech_eval/eval_user_interruption_9B.sh ${EXP_NAME} ${STEP_NAME} ${BOOST_NAME} ${INF_NAME} ${FTT} ${PW}
# bash ${CODE_DIR}/scripts/speech_eval/eval_pause_handling_9B.sh ${EXP_NAME} ${STEP_NAME} ${BOOST_NAME} ${INF_NAME} ${FTT} ${PW}
set +x