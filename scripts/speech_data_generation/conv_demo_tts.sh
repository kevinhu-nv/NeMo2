# Create a stereo example based on the generated wav id

val_data=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/demo/shar

# model_name=s2s-otf-s2s_s2t-pt_salm_1abr_duplex4_data_scaletd1obrm_oci_S2S_llama_tiny__lr3e-4wd0_InverseSquareRootAnnealing_warmup2500_minlr1e-6_gbs512_mbs16_ep200_bd400_mistral
# model_name=s2s-otf-s2s_s2t-pt_salm_1abr_duplex4_data_scaletd1obrm_oci_S2S_llama_tiny__lr3e-4wd0_InverseSquareRootAnnealing_warmup2500_minlr1e-6_gbs512_mbs16_ep200_bd400_mistral_uc1top
model_name=s2s-otf-s2s_s2t-pt_salm_1abr_duplex4_data_scaletd1obrm_oci_S2S_llama_tiny__lr3e-4wd0_InverseSquareRootAnnealing_warmup2500_minlr1e-6_gbs512_mbs16_ep200_bd400_mistral_5k.v2
# model_name=s2s-otf-s2s_s2t-pt_salm_1abr_duplex4_data_scaletd1obrm_oci_S2S_llama_tiny__lr3e-4wd0_InverseSquareRootAnnealing_warmup2500_minlr1e-6_gbs512_mbs16_ep200_bd400_mistral_5k
gen_wav_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/results/salm_s2s_huk/${model_name}/wav/pred/
out_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/debug

json_file=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/demo/shar/cuts.000000.jsonl.gz
zcat $json_file | jq -r '.id' | while read -r id; do
    echo "Processing ID: $id"
    tar -xvf "${val_data}/recording.000000.tar" -C ${out_dir} "${id}.flac"
    echo ${out_dir}/${id}.flac
    gen_wav=${gen_wav_dir}/${id}.gen.wav
    echo "$gen_wav"
    python /home/kevinhu/s2s/duplex/create_stereo.py --file1 ${out_dir}/${id}.flac --file2 ${gen_wav} --output_file ${gen_wav}.stereo.wav
done