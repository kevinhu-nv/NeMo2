#!/bin/bash
# #SBATCH -A convai_convaird_nemo-speech
# #SBATCH -J "convai_convaird_nemo-speech-speechllm:data_duplex_multiturn"
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
#SBATCH -J "llmservice_nemo_speechlm:data_duplex_multiturn"

CODE_DIR=/lustre/fs12/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation

# dataset_name=ultrachat
# root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/${dataset_name}
# dataset_name=Meta-Llama-3.1-70B-Instruct
# root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/topic_v2/${dataset_name}
# dataset_name=ultrachat_v2
# dataset_name=triviaqa_train
# dataset_name=triviaqa_validation
# dataset_name=triviaqa_web_train
dataset_name=triviaqa_unfiltered_train
root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/${dataset_name}

RESULTS_DIR=${root_dir}/log
mkdir -p $RESULTS_DIR
single_turn_shar_dir=${root_dir}/shar
in_dir=${single_turn_shar_dir}
out_shar_dir=${root_dir}/shar_duplex
num_turn=2
# out_shar_dir=${root_dir}/shar_duplex_1turn
# num_turn=1
mkdir -p ${out_shar_dir}


CONTAINER=/lustre/fsw/portfolios/llmservice/users/zhehuaic/containers/nemo_s2s_24.08zhc.sqsh
MOUNTS="--container-mounts=/lustre/fsw:/lustre/fsw,${RESULTS_DIR}:${RESULTS_DIR},${CODE_DIR}:${CODE_DIR},${root_dir}:${root_dir},/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE/:/hfcache/,${in_dir}:${in_dir},${out_shar_dir}:${out_shar_dir}"

#####################
# topic

function create_duplex() {

read -r -d '' cmd <<EOF
cd ${CODE_DIR} \
&& export HF_HOME="/hfcache/" \
&& export TORCH_HOME="/hfcache/torch" \
&& export NEMO_CACHE_DIR="/hfcache/torch/nemo" \
&& export HF_DATASETS_CACHE="/hfcache/datasets" \
&& export TOKENIZERS_PARALLELISM=false \
&& export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 \
&& HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 python -u -B create_shars_duplex_multi_from_shar.py \
--in_dir=${in_dir} \
--out_shar_dir=${out_shar_dir} \
--shar_index=${shar_index} \
--num_shard=10 \
--num_turn=${num_turn}
EOF

OUTFILE=${RESULTS_DIR}/slurm-%j-%n-${dataset_name}_${shar_index}.out
ERRFILE=${RESULTS_DIR}/error-%j-%n-${dataset_name}_${shar_index}.out

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
}

shar_index="$1"
create_duplex