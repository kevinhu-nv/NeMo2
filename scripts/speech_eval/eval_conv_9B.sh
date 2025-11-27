#!/bin/bash

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo

function eval_conv() {
extra_args="$1"
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
}

pred_audio_dir="/lustre/fsw/portfolios/convai/users/kevinhu/results/s2s_rl/uc_samples"
manifest_dir="/lustre/fsw/portfolios/convai/users/kevinhu/data/ultrachat_200_0"
barge_in_threshold_sec=1.5
end_time=None  # Note that this comes from predefined values when creating backchanneling data
tt_accuracy_threshold_sec=0.64
verbose=True
# eval_conv

######################
# Eval HF baseline model
# /lustre/fsw/portfolios/convai/users/cchen1/duplex_s2s/exp/tiny-llama_4node_baseline/checkpoints/step=12001.ckpt
pred_audio_dir="/lustre/fsw/portfolios/convai/users/kevinhu/duplex_s2s/results/pred_audios/pred_wavs/"
manifest_dir="/lustre/fsw/portfolios/convai/projects/convai_convaird_nemo-speech/convai_convaird_nemo-speech/data/duplex/Mixtral8x22b_MMLPC_en/manifest_511"
barge_in_threshold_sec=1.5
end_time=None  # Note that this comes from predefined values when creating backchanneling data
tt_accuracy_threshold_sec=0.64
verbose=True
validation_set_name="mistral_511,voicebench_alpaca,voicebench_openbook,voicebench_commoneval"
# eval_conv
# validation_set_name="mistral_511"
# eval_conv "--verbose"

######################
# Eval demo audios
# copied from https://docs.google.com/document/d/1lLKhXoPiBbYBZefT5IJq9nko00ckpR42byoi_PL74SA/edit?tab=t.0#bookmark=id.de8rk7ofhy4b
pred_audio_dir="/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/results/inferences/1.78kbps/demo_model_no_aug_chen_chen_only_demo_1demo_model_no_aug_chen_chen_4nodes_nonsil10.0_zhehuai_01_jul_g_baseline_no_davidai_qwen_no_lat_with_bos_eos_dp_sd_with_bos_eos_dp_sd_sd_state/validation_logs/pred_wavs"
barge_in_threshold_sec=1.5
end_time=None  # Note that this comes from predefined values when creating backchanneling data
tt_accuracy_threshold_sec=0.64
validation_set_name="demo"
# eval_conv "--verbose"

EXP_NAME=$1
STEP_NAME=$2
BOOST_NAME=$3
INF_NAME=$4
DATASET_NAME=${5:-real_demo}

# EXP_NAME=EOS_Nano-9B_SFT_Parakeet600m_noise_aug_from_step-14003_64gpu_PT0.6_SFT0.15_QA0.02_TEXT0.2_MCQ0.03_LR5e-5_NoiseProb0.6
# STEP_NAME=30005
# BOOST_NAME=pad0_bos0_eos0
# BOOST_NAME=pad-1_bos0_eos0
# BOOST_NAME=pad-5_bos4_eos4

# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/exp/${EXP_NAME}/inf/${CKPT_NAME}/validation_logs
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/exp/${EXP_NAME}/inf_all/${CKPT_NAME}/validation_logs
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_from_cc_20251003/exp_SFT_qwen7b/${EXP_NAME}/results/inf/infer_qwen_7b_step_${STEP_NAME}/${BOOST_NAME}/validation_logs

# log_dir=/lustre/fsw/llmservice_nemo_speechlm/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT_9b/${EXP_NAME}/results/inf/infer_nano_9b_step_${STEP_NAME}/${BOOST_NAME}/validation_logs
# log_dir=/lustre/fsw/convai_convaird_nemo-speech/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT_9b/${EXP_NAME}/results/inf/infer_nano_9b_step_${STEP_NAME}/${BOOST_NAME}/validation_logs
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT-nano9b/${EXP_NAME}/results/inf/infer_nano_9b_step_${STEP_NAME}/${BOOST_NAME}/validation_logs
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs
log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge_rebase2/exp_SFT_9b/${EXP_NAME}/${INF_NAME}_step-${STEP_NAME}/${BOOST_NAME}/validation_logs
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/IAD_Nano-9B_SFT_Parakeet600m_asr_sp_from_checkpoints_hf_32002_64gpu_PT0.65_SFT0.05_QA0.02_TEXT0.1_MCQ0.03_ASR0.01_SP0.05-b200_LR5e-5_na0.5-30-60_ci_sa15_la2_tls10.0_facuda_val200.v2/results/inf/infer_nano_9b_step_6354/pad0_bos0_eos0/validation_logs
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/results/inf/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/
log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/${EXP_NAME}/results/inf_rebase/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_nov/exp_SFT-nano9b/${EXP_NAME}/results/inf/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/
# log_dir=/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/results/exp/nov25/exp_SFT_9b/${EXP_NAME}/results/inf_rebase/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_nov/exp_SFT-nano9b/${EXP_NAME}/results/inf_rebase_squash/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/
# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_nov/exp_SFT-nano9b/${EXP_NAME}/results/inf2/${INF_NAME}_step_${STEP_NAME}/${BOOST_NAME}/validation_logs/

pred_audio_dir=${log_dir}/pred_wavs/

# Merge demo_rank0.json through demo_rank7.json into demo.json
jsonl_with_timestamp=${log_dir}/metadatas/${DATASET_NAME}.json
# jsonl_with_timestamp=${log_dir}/metadatas/demo.json
# cat ${log_dir}/metadatas/real_demo_rank{0..7}.json > $jsonl_with_timestamp
cat ${log_dir}/metadatas/${DATASET_NAME}_rank{0..31}.json > $jsonl_with_timestamp
# cat ${log_dir}/metadatas/demo_rank{0..31}.json > $jsonl_with_timestamp

output_log=${jsonl_with_timestamp}.log
barge_in_threshold_sec=1.5
end_time=None  # Note that this comes from predefined values when creating backchanneling data
tt_latency_threshold_sec=1.5
tt_recall_buffer_sec=20
tt_precision_buffer_sec=1
vad_min_silence_duration_ms=2000
validation_set_name="${DATASET_NAME}"
eval_conv "--verbose --jsonl_with_timestamp $jsonl_with_timestamp --vad_min_silence_duration_ms $vad_min_silence_duration_ms --enable_transcription"
echo "Output log: $output_log"