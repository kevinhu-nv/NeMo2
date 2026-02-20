#!/bin/bash

script=/lustre/fsw/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/source_align/create_nfa_manifest_ultrachat.sh
for i in {1..19}; do
manifest_index=$(printf "%06d" ${i})
./autorun.sh -n 1 ${script} $manifest_index
done