#!/bin/bash

script=/lustre/fsw/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/source_align/create_nfa_manifest.sh
for i in {511..511}; do
./autorun.sh -n 1 ${script} $i
done