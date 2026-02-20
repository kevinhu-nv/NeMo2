#!/bin/bash

for i in {1..50}; do
script=/lustre/fsw/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/source_align/run_nfa_topic.sh
./autorun.sh -n 1 ${script} ${i}
done

exit 0

#####################
# Sanity check aligned files
for i in {400..511}; do
echo "Checking manifest_${i}"   
count_words=$(ls /lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/Mixtral8x22b_MMLPC_en/manifest_${i}/nfa_manifest.jsonl_align/ctm/words/ | wc -l)
count_manifest=$(cat /lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/Mixtral8x22b_MMLPC_en/manifest_${i}/nfa_manifest.jsonl | wc -l)

if [[ "$count_words" -ne "$count_manifest" ]]; then
    echo "Inconsistent: /lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/Mixtral8x22b_MMLPC_en/manifest_${i}"
fi
done


#####################
# remove wav files to free up space
for i in {201..300}; do
rm -r /lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/Mixtral8x22b_MMLPC_en/manifest_${i}/recording
done

# Remove non-word level alignments
find /lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/topic_v2/Meta-Llama-3.1-70B-Instruct/shar_duplex/source_align -type d \( -name "tokens" -o -name "segments" \) -exec rm -rf {} +