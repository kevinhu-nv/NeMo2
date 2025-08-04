#!/bin/bash

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo

function eval_conv() {
extra_args="$1"
python $CODE_DIR/scripts/speech_eval/eval_conversation_behavior.py \
    --pred_audio_dir $pred_audio_dir \
    --manifest_dir $manifest_dir \
    --barge_in_threshold_sec $barge_in_threshold_sec \
    --end_time $end_time \
    --tt_accuracy_threshold_sec 1.5 \
    --barge_in_threshold_sec 1.5 \
    --validation_set_name $validation_set_name $extra_args
}

pred_audio_dir="/lustre/fsw/portfolios/convai/users/kevinhu/results/s2s_rl/uc_samples"
manifest_dir="/lustre/fsw/portfolios/convai/users/kevinhu/data/ultrachat_200_0"
barge_in_threshold_sec=1.5
end_time=None  # Note that this comes from predefined values when creating backchanneling data
tt_accuracy_threshold_sec=0.64
verbose=True
# eval_conv

# Eval HF baseline model
# /lustre/fsw/portfolios/convai/users/cchen1/duplex_s2s/exp/tiny-llama_4node_baseline/checkpoints/step=12001.ckpt
pred_audio_dir="/lustre/fsw/portfolios/convai/users/kevinhu/duplex_s2s/results/pred_audios/pred_wavs/"
manifest_dir="/lustre/fsw/portfolios/convai/projects/convai_convaird_nemo-speech/convai_convaird_nemo-speech/data/duplex/Mixtral8x22b_MMLPC_en/manifest_511"
barge_in_threshold_sec=1.5
end_time=None  # Note that this comes from predefined values when creating backchanneling data
tt_accuracy_threshold_sec=0.64
verbose=True
validation_set_name="mistral_511,voicebench_alpaca,voicebench_openbook,voicebench_commoneval"
eval_conv
# validation_set_name="mistral_511"
# eval_conv "--verbose"