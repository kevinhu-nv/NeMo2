#!/bin/bash
# #SBATCH -A convai_convaird_nemo-speech
# #SBATCH -J "convai_convaird_nemo-speech-speechllm:create_nfa_manifest"
#SBATCH -p cpu_short
#SBATCH -N 1  # number of nodes
#SBATCH -t 00:30:00
#SBATCH --time-min 00:30:00  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

#SBATCH -A llmservice_nemo_speechlm
#SBATCH -J "llmservice_nemo_speechlm:create_nfa_manifest"

CODE_DIR=/lustre/fs12/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/source_align

# llm_root_dir=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/Mixtral8x22b_MMLPC_en
# llm_root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/triviaqa_train/shar_duplex/
# llm_root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/ultrachat_v2/shar_duplex
# llm_root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/ultrachat/shar_duplex
# llm_root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/topic_v2/Meta-Llama-3.1-70B-Instruct/shar_duplex
llm_root_dir=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/s2s_synthetic_data/Mixtral8x22b_riva_asr_en_set_7p0/lhotse_shars
# user_root_dir=${llm_root_dir}/source_align
user_root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/s2s_synthetic_data/Mixtral8x22b_riva_asr_en_set_7p0/source_align
mkdir -p ${user_root_dir}
RESULTS_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/nfa_log

CONTAINER=/lustre/fsw/portfolios/llmservice/users/zhehuaic/containers/nemo_s2s_24.08zhc.sqsh
MOUNTS="--container-mounts=/lustre/fsw:/lustre/fsw,${RESULTS_DIR}:${RESULTS_DIR},${CODE_DIR}:${CODE_DIR},/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE/:/hfcache/,${llm_root_dir}:${llm_root_dir},${user_root_dir}:${user_root_dir}"

function create_nfa_manifest() {

in_shar_dir=${llm_root_dir}/bucket_${bucket_index}/manifest_${shar_index}
output_audio_dir=${user_root_dir}/bucket_${bucket_index}/manifest_${shar_index}/recording
output_manifest_dir=${user_root_dir}/bucket_${bucket_index}/manifest_${shar_index}/
mkdir -p ${output_audio_dir}
mkdir -p ${output_manifest_dir}

read -r -d '' cmd <<EOF
cd ${CODE_DIR} \
&& export HF_HOME="/hfcache/" \
&& export TORCH_HOME="/hfcache/torch" \
&& export NEMO_CACHE_DIR="/hfcache/torch/nemo" \
&& export HF_DATASETS_CACHE="/hfcache/datasets" \
&& export TRANSFORMERS_CACHE="/hfcache/models" \
&& export TOKENIZERS_PARALLELISM=false \
&& export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 \
&& HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 python -u -B ${CODE_DIR}/create_nfa_manifest.py \
--in_shar_dir=${in_shar_dir} \
--output_audio_dir=${output_audio_dir} \
--output_manifest_dir=${output_manifest_dir}
EOF

OUTFILE=${RESULTS_DIR}/slurm-%j-%n-nfa${shar_index}.out
ERRFILE=${RESULTS_DIR}/error-%j-%n-nfa${shar_index}.out

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
}

bucket_index="$1"
shar_index="$2"
create_nfa_manifest
