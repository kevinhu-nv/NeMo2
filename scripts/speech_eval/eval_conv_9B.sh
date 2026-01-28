#!/bin/bash

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo

function eval_conv() {
extra_args="$1"
set -x
python $CODE_DIR/scripts/speech_eval/eval_conversation_behavior.py \
    --pred_audio_dir $pred_audio_dir \
    --manifest_dir $manifest_dir \
    --barge_in_threshold_sec $barge_in_threshold_sec \
    --end_time $end_time \
    --tt_latency_threshold_sec $tt_latency_threshold_sec \
    --tt_precision_buffer_sec $tt_precision_buffer_sec \
    --tt_recall_buffer_sec $tt_recall_buffer_sec \
    --barge_in_threshold_sec $barge_in_threshold_sec \
    --validation_set_name $validation_set_name $extra_args 2>&1 | tee $output_log
set +x
}

EXP_NAME=$1
STEP_NAME=$2
BOOST_NAME=$3
INF_NAME=$4
DATASET_NAME=${5:-real_demo}
FTT=${6:-False}
PW=${7:-1}

# log_dir=/lustre/fsw/llmservice_nemo_speechlm/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT_9b/${EXP_NAME}/results/inf/infer_nano_9b_step_${STEP_NAME}/${BOOST_NAME}/validation_logs
# log_dir=/lustre/fsw/convai_convaird_nemo-speech/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT_9b/${EXP_NAME}/results/inf/infer_nano_9b_step_${STEP_NAME}/${BOOST_NAME}/validation_logs
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT-nano9b/${EXP_NAME}/results/inf/infer_nano_9b_step_${STEP_NAME}/${BOOST_NAME}/validation_logs
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs
log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge_rebase2/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs
log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/results/inf_rebase/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_nov/exp_SFT-nano9b/${EXP_NAME}/results/inf/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/results/inf/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/ftt_${FTT}_pw${PW}_upad0_ubos0_ueos0/validation_logs/
# log_dir=/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/results/exp/nov25/exp_SFT_9b/${EXP_NAME}/results/inf_rebase/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_nov/exp_SFT-nano9b/${EXP_NAME}/results/inf_rebase_squash/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_nov/exp_SFT-nano9b/${EXP_NAME}/results/inf2/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/results/inf/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/results/inf/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/ftt_${FTT}_pw${PW}/validation_logs/
# log_dir=/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/results/exp/nov25/exp_SFT_9b/${EXP_NAME}/results/inf/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/ftt_${FTT}_pw${PW}_upad0_ubos0_ueos0/validation_logs/
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/results/inf/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/ftt_${FTT}_pw${PW}_upad0_ubos0_ueos0/validation_logs/
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20260114/exp_SFT_9b/${EXP_NAME}/results/inf_rebase/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/

pred_audio_dir=${log_dir}/pred_wavs/

jsonl_with_timestamp=${log_dir}/metadatas/${DATASET_NAME}.json
cat ${log_dir}/metadatas/${DATASET_NAME}_rank{0..31}.json > $jsonl_with_timestamp

output_log=${jsonl_with_timestamp}.log
barge_in_threshold_sec=1.5
end_time=None  # Note that this comes from predefined values when creating backchanneling data
tt_latency_threshold_sec=1.5
tt_recall_buffer_sec=20
tt_precision_buffer_sec=1
vad_min_silence_duration_ms=2000
validation_set_name="${DATASET_NAME}"
eval_conv "--verbose --jsonl_with_timestamp $jsonl_with_timestamp --vad_min_silence_duration_ms $vad_min_silence_duration_ms --enable_transcription --compute_user_eou"
echo "Output log: $output_log"