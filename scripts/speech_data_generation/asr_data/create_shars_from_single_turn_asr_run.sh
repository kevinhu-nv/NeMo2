#!/bin/bash

script=/lustre/fsw/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/asr_data/create_shars_from_single_turn_asr.sh
for shard_number in {100..511}; do
./autorun.sh -n 1 ${script} ${shard_number}
done