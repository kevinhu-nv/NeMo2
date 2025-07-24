#!/bin/bash

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/works/mod_speech_llm/code/NeMo_speech_decoder_v2_rebased_srctext_dbg

function eval_conv() {
PYTHONBREAKPOINT=0 python $CODE_DIR/scripts/speech_eval/eval_conversation_behavior.py \
    --pred_audio_path $pred_audio_path \
    --manifest_dir $manifest_dir \
    --barge_in_threshold_sec $barge_in_threshold_sec \
    --end_time $end_time \
    --tt_accuracy_threshold_sec 0.64 \
    --verbose
}

pred_audio_path="/lustre/fsw/portfolios/convai/users/kevinhu/results/s2s_rl/uc_samples"
manifest_dir="/lustre/fsw/portfolios/convai/users/kevinhu/data/ultrachat_200_0"
barge_in_threshold_sec=1.5
end_time=None  # Note that this comes from predefined values when creating backchanneling data
tt_accuracy_threshold_sec=0.64
verbose=True
eval_conv