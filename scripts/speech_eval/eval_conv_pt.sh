#!/bin/bash

CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo

function eval_conv() {
extra_args="$1"
python $CODE_DIR/scripts/speech_eval/eval_conversation_behavior.py \
    --pred_audio_dir $pred_audio_dir \
    ${manifest_dir:+--manifest_dir $manifest_dir} \
    --barge_in_threshold_sec $barge_in_threshold_sec \
    --end_time $end_time \
    --tt_latency_threshold_sec $tt_latency_threshold_sec \
    --tt_precision_buffer_sec $tt_precision_buffer_sec \
    --tt_recall_buffer_sec $tt_recall_buffer_sec \
    --barge_in_threshold_sec $barge_in_threshold_sec \
    --validation_set_name $validation_set_name $extra_args 2>&1 | tee $output_log
}

######################
# Eval demo text
EXP_NAME="IAD_qwen_1b_and_pretrain_4nodes_repro_recipe2_preft1.0_sft0.0_noinit" && CKPT_NAME=step-7239-last

log_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/exp/${EXP_NAME}/inf/${CKPT_NAME}/validation_logs

pred_audio_dir=${log_dir}/pred_wavs/
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