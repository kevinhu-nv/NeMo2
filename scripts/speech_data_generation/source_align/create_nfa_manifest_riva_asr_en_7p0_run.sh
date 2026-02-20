#!/bin/bash

script=/lustre/fsw/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/source_align/create_nfa_manifest_riva_asr_en_7p0.sh

root_dir=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/s2s_synthetic_data/Mixtral8x22b_riva_asr_en_set_7p0/lhotse_shars
# bucket_index 0..8
# shar_index 0..100
# for bucket_index in {1..8}; do
for bucket_index in {1..1}; do
bucket_dir=${root_dir}/bucket_${bucket_index}
num_manifest=$(ls ${bucket_dir}/manifest_* -d | wc -l)
for (( shar_index=1; shar_index<=num_manifest; shar_index++ )); do
    ./autorun.sh -n 1 ${script} $bucket_index $shar_index
done
done

exit 0

#####################
# remove wav files to free up space
root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/s2s_synthetic_data/Mixtral8x22b_riva_asr_en_set_7p0/source_align
for bucket_index in {1..8}; do
bucket_dir=${root_dir}/bucket_${bucket_index}
num_manifest=$(ls ${bucket_dir}/manifest_* -d | wc -l)
for (( shar_index=1; shar_index<=num_manifest; shar_index++ )); do
    rm -r ${root_dir}/bucket_${bucket_index}/manifest_${shar_index}/recording
done
done