#!/bin/bash
# #SBATCH -A convai_convaird_nemo-speech
# #SBATCH -J "convai_convaird_nemo-speech-speechllm:data_duplex_single_turn"
#SBATCH -p cpu_short
#SBATCH -N 1  # number of nodes
#SBATCH -t 00:40:00
#SBATCH --time-min 00:40:00  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

#SBATCH -A llmservice_nemo_speechlm
#SBATCH -J "llmservice_nemo_speechlm:data_duplex_single_turn"

set -x

CODE_DIR=/lustre/fs12/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation

################## ultrachat
# dataset_name=ultrachat
# root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/${dataset_name}
################## topic_v2
# dataset_name=Meta-Llama-3.1-70B-Instruct
# root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/topic_v2/${dataset_name}
# target_tar_dir=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/s2s_synthetic_data/${dataset_name}/${dataset_name}
# source_tar_dir=${target_tar_dir}
################## ultrachat_v2
# dataset_name=ultrachat_v2
# dataset_name=triviaqa_train
# dataset_name=triviaqa_validation
# dataset_name=triviaqa_web_train
dataset_name=triviaqa_unfiltered_train
root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/${dataset_name}
target_tar_dir=${root_dir}/koel
source_tar_dir=${root_dir}/koel

RESULTS_DIR=${root_dir}/log
mkdir -p $RESULTS_DIR
input_manifest_dir=${root_dir}/conv
out_dir=${root_dir}/shar
mkdir -p ${out_dir}


CONTAINER=/lustre/fsw/portfolios/llmservice/users/zhehuaic/containers/nemo_s2s_24.08zhc.sqsh
MOUNTS="--container-mounts=/lustre/fsw:/lustre/fsw,${RESULTS_DIR}:${RESULTS_DIR},${CODE_DIR}:${CODE_DIR},${root_dir}:${root_dir},/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE/:/hfcache/,${input_manifest_dir}:${input_manifest_dir},${out_dir}:${out_dir},${source_tar_dir}:${source_tar_dir},${target_tar_dir}:${target_tar_dir}"

#####################
# topic

function create_single_turn() {
read -r -d '' cmd <<EOF
cd ${CODE_DIR} \
&& export HF_HOME="/hfcache/" \
&& export TORCH_HOME="/hfcache/torch" \
&& export NEMO_CACHE_DIR="/hfcache/torch/nemo" \
&& export HF_DATASETS_CACHE="/hfcache/datasets" \
&& export TRANSFORMERS_CACHE="/hfcache/models" \
&& export TOKENIZERS_PARALLELISM=false \
&& export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 \
&& HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 python -u -B create_shars_from_single_turn_triviaqa.py \
--input_manifest_dir=${input_manifest_dir} \
--target_tar_dir=${target_tar_dir} \
--source_tar_dir=${source_tar_dir} \
--out_dir=${out_dir} \
--shard_index=${shar_index}
EOF

OUTFILE=${RESULTS_DIR}/slurm-%j-%n-${dataset_name}_single_${shar_index}.out
ERRFILE=${RESULTS_DIR}/error-%j-%n-${dataset_name}_single_${shar_index}.out

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
# bash -c "$cmd"
}

shar_index="$1"
create_single_turn

set +x