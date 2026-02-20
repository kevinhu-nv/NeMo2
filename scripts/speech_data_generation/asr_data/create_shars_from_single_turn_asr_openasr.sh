#!/bin/bash

script=/lustre/fsw/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/asr_data/create_shars_from_single_turn_asr_openasr.py

root_dir=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/asr_evaluator/datasets/HF-audio
out_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/ASR

function create_shar() {
    if [ -n "$root_dir" ]; then
        python $script --root_dir $root_dir --input_manifest $input_manifest --out_dir $out_dir
    else
        python $script --input_manifest $input_manifest --out_dir $out_dir --input_manifest_path $input_manifest_path
    fi
}

# LS test-clean debug
# out_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/ASR/debug
# input_manifest=open-asr-leaderboarddatasets-test-only-librispeech-test.clean.json
# create_shar
# exit 0

#####################
# LS test-clean
input_manifest=open-asr-leaderboarddatasets-test-only-librispeech-test.clean.json
# create_shar

# LS test-other
input_manifest=open-asr-leaderboarddatasets-test-only-librispeech-test.other.json
# create_shar

# Gigaspeech
input_manifest=open-asr-leaderboarddatasets-test-only-gigaspeech-test.json
# create_shar

# AMI
input_manifest=open-asr-leaderboarddatasets-test-only-ami-test.json
# create_shar

# SPGISpeech
input_manifest=open-asr-leaderboarddatasets-test-only-spgispeech-test.json
# create_shar

# Earnings
input_manifest=open-asr-leaderboarddatasets-test-only-earnings22-test.json
# create_shar

input_manifest=open-asr-leaderboarddatasets-test-only-tedlium-test.json
# create_shar

input_manifest=open-asr-leaderboarddatasets-test-only-voxpopuli-test.json
create_shar

root_dir=
input_manifest=librivox-test-other_over20.json
input_manifest_path=/lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/tmp/librivox-test-other_over20.json
# create_shar