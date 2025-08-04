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
# eval_conv
# validation_set_name="mistral_511"
# eval_conv "--verbose"

# Eval demo audios
# copied from https://docs.google.com/document/d/1lLKhXoPiBbYBZefT5IJq9nko00ckpR42byoi_PL74SA/edit?tab=t.0#bookmark=id.de8rk7ofhy4b
pred_audio_dir="/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/results/inferences/1.78kbps/demo_model_no_aug_chen_chen_only_demo_1demo_model_no_aug_chen_chen_4nodes_nonsil10.0_zhehuai_01_jul_g_baseline_no_davidai_qwen_no_lat_with_bos_eos_dp_sd_with_bos_eos_dp_sd_sd_state/validation_logs/pred_wavs"
barge_in_threshold_sec=1.5
end_time=None  # Note that this comes from predefined values when creating backchanneling data
tt_accuracy_threshold_sec=0.64
validation_set_name="demo"
eval_conv "--verbose"