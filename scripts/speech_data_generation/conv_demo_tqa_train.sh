# Create a stereo example based on the generated wav id

# model_name=s2s-otf-s2s_s2t-pt_salm_1abr_duplex4_data_scaletd1obrm_oci_S2S_llama_tiny__lr3e-4wd0_InverseSquareRootAnnealing_warmup2500_minlr1e-6_gbs512_mbs16_ep200_bd400_mistral
# model_name=s2s-otf-s2s_s2t-pt_salm_1abr_duplex4_data_scaletd1obrm_oci_S2S_llama_tiny__lr3e-4wd0_InverseSquareRootAnnealing_warmup2500_minlr1e-6_gbs512_mbs16_ep200_bd400_mistral_uc1top
# model_name=s2s-otf-s2s_s2t-pt_salm_1abr_duplex4_data_scaletd1obrm_oci_S2S_llama_tiny__lr3e-4wd0_InverseSquareRootAnnealing_warmup2500_minlr1e-6_gbs512_mbs16_ep200_bd400_mistral_5ktqa
# model_name=s2s-otf-s2s_s2t-pt_salm_1abr_duplex4_data_scalehe_oci_S2S_llama_tiny__lr3e-4wd0_InverseSquareRootAnnealing_warmup2500_minlr1e-6_gbs512_mbs16_ep200_bd1000_mistral_srctext_w10/
model_name=s2s-otf-s2s_s2t-pt_salm_1abr_duplex4_data_scalehe_oci_S2S_llama_tiny__lr3e-4wd0_InverseSquareRootAnnealing_warmup2500_minlr1e-6_gbs512_mbs16_ep200_bd1000_mistral_srctext_w10_srcalign_left

# val_data=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/triviaqa_train/Llama-2-13b-chat-hf/shar_duplex/manifest_000059
val_data=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/triviaqa_train/Llama-2-13b-chat-hf/s2s-otf-s2s_s2t-pt_salm_1abr_duplex4_data_scalehe_oci_S2S_llama_tiny__lr3e-4wd0_InverseSquareRootAnnealing_warmup2500_minlr1e-6_gbs512_mbs16_ep200_bd1000_mistral_srctext_w10_srcalign_left/shar_duplex/manifest_000059

# gen_wav_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/results/salm_s2s_inf/${model_name}/${model_name}/salm_duplex_infc_2lm/srctext_gt/wav/pred/
gen_wav_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/results/salm_s2s_inf/${model_name}/${model_name}/salm_duplex_infc_2lm_srctext/srctext_gt/wav/pred/
out_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/debug

# id=$(ls ${gen_wav_dir}/59_*.gen.wav | awk -F'/' '{print $NF}' | grep -oE '^59_[^.]+' | shuf -n 1)
id=59_60928_1


echo "Selected ID: $id"

for file in ${val_data}/cuts.*.jsonl.gz; do
    echo $file
    if zcat "$file" | grep -E "\"id\": \"${id}\""; then
        index=$(echo "$file" | grep -oE '[0-9]+' | tail -1)
        echo "Match found in: $file"

        zcat "$file" | grep "\"id\": \"${id}" | jq

        set -x
        tar -xvf "${val_data}/recording.${index}.tar" -C ${out_dir} "${id}.flac"
        set +x
    fi
done

# Create stereo wav
gen_wav=${gen_wav_dir}/${id}.gen.wav
set -x
python /home/kevinhu/s2s/duplex/create_stereo.py --file1 ${out_dir}/${id}.flac --file2 ${gen_wav} --output_file ${gen_wav}.stereo.wav
set +x