#!/bin/bash

script=/lustre/fsw/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/source_align/add_timestamp.sh
for i in {379..450}; do
./autorun.sh -n 1 ${script} $i
done