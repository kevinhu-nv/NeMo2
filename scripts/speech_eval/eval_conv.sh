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

######################
# Eval demo text
# EXP_NAME="IAD_qwen_1b_and_pretrain_4nodes_repro_recipe2_preft1.0_sft0.0_noinit" && CKPT_NAME=step-7239-last
# EXP_NAME="IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05"  # may have noise-augmentation
# EXP_NAME="IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1"
# EXP_NAME="IAD_qwen_1b_sft_pt_4nodes_pt0.9sft_0.05st0.05_na" && CKPT_NAME=step-6001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_nosa && CKPT_NAME=step-4001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft_0.05st0.05_na0.3_snr-30-6 && CKPT_NAME=step-6001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta2 && CKPT_NAME=step-14563-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60 && CKPT_NAME=step-5001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_snr0.5-50-80_ta2 && CKPT_NAME=step-7849-last
# EXP_NAME=IAD_qwen_1b_sft_pt_lr_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-24-15_lr5e-5 && CKPT_NAME=step-13726-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0 && CKPT_NAME=step-13003-last
# EXP_NAME=IAD_qwen_1b_sft_pt-bd200_4nodes_pt0.98sft0.01st0.01_na0.5_snr0.5-30-60_ptbd150 && CKPT_NAME=step-8169-last
# EXP_NAME=IAD_qwen_1b_sft_pt-bd200_4nodes_pt0.98sft0.01st0.01_na0.5_snr0.5-30-60_ptbd200 && CKPT_NAME=step-4001-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-50-80 && CKPT_NAME=step-6553-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0 && CKPT_NAME=step-13003-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.9sft0.05st0.05_na0.5_snr0.5-30-60_ta0 && CKPT_NAME=step-25339-last
# EXP_NAME=IAD_qwen_1b_sft_pt_4nodes_pt0.8sft0.1st0.1_na0.5_snr0.5-6-60 && CKPT_NAME=step-6001-last
EXP_NAME=$1
CKPT_NAME=$2

# log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/exp/${EXP_NAME}/inf/${CKPT_NAME}/validation_logs
log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/exp/${EXP_NAME}/inf_all/${CKPT_NAME}/validation_logs
pred_audio_dir=${log_dir}/pred_wavs/

# Merge demo_rank0.json through demo_rank7.json into demo.json
jsonl_with_timestamp=${log_dir}/metadatas/demo.json

output_log=${jsonl_with_timestamp}.log
barge_in_threshold_sec=1.5
end_time=None  # Note that this comes from predefined values when creating backchanneling data
tt_latency_threshold_sec=1.5
tt_recall_buffer_sec=20
tt_precision_buffer_sec=1
vad_min_silence_duration_ms=2000
validation_set_name="demo"
eval_conv "--verbose --jsonl_with_timestamp $jsonl_with_timestamp --vad_min_silence_duration_ms $vad_min_silence_duration_ms"
echo "Output log: $output_log"

######################
# Eval repro baseline ckpt-recipe1 decoded demo audios
# ckpt-recipe1 from https://docs.google.com/document/d/17z9JeCK8Ht-ojLp1H8ynJqr8Pz7aGdohy0C0PYIvXvY/edit?pli=1&tab=t.0#bookmark=id.7im551u1k7e0
pred_audio_dir="/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/results/inferences/1.78kbps/demo_model_no_aug_chen_chen_only_demo_1demo_model_no_aug_chen_chen_4nodes_nonsil10.0_zhehuai_01_jul_g_baseline_no_davidai_qwen_no_lat_with_bos_eos_dp_sd_with_bos_eos_dp_sd_sd_state/validation_logs/pred_wavs"
barge_in_threshold_sec=1.5
end_time=None  # Note that this comes from predefined values when creating backchanneling data
tt_accuracy_threshold_sec=0.64
validation_set_name="demo"
# eval_conv "--verbose"

######################
# Transcribe using full-duplex-bench ASR
# pred_audio_dir="/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/results/inferences/1.78kbps/demo_model_no_aug_chen_chen_only_demo_1demo_model_no_aug_chen_chen_4nodes_nonsil10.0_zhehuai_01_jul_g_baseline_no_davidai_qwen_no_lat_with_bos_eos_dp_sd_with_bos_eos_dp_sd_sd_state/validation_logs/pred_wavs"

# python $CODE_DIR/scripts/speech_eval/Full-Duplex-Bench/get_transcript/asr.py --root_dir $pred_audio_dir --task full