#!/bin/bash

script=/lustre/fsw/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/source_align/add_timestamp_topic.sh
for i in {1..50}; do
manifest_index=$(printf "%06d" ${i})
./autorun.sh -n 1 ${script} $manifest_index
done