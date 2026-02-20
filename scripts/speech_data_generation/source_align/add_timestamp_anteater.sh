#!/bin/bash
#SBATCH -A convai_convaird_nemo-speech
#SBATCH -J "convai_convaird_nemo-speech-speechllm:add_timestamp_tqa"
#SBATCH -p cpu_short
#SBATCH -N 1  # number of nodes
#SBATCH -t 04:00:00
#SBATCH --time-min 04:00:00  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# #SBATCH -A llmservice_nemo_speechlm
# #SBATCH -J "llmservice_nemo_speechlm:add_timestamp_tqa"

#!/bin/bash

CODE_DIR=/lustre/fs12/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/source_align

# llm_root_dir=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/Mixtral8x22b_MMLPC_en
llm_root_dir=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/340b_8.19_daring_anteater_lmsys_sft_8801rm3.7p
# user_root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/Mixtral8x22b_MMLPC_en
user_root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/340b_8.19_daring_anteater_lmsys_sft_8801rm3.7p/shar_duplex
RESULTS_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/data_duplex/nfa_log

CONTAINER=/lustre/fsw/portfolios/llmservice/users/zhehuaic/containers/nemo_s2s_24.08zhc.sqsh
MOUNTS="--container-mounts=/lustre/fsw:/lustre/fsw,${RESULTS_DIR}:${RESULTS_DIR},${CODE_DIR}:${CODE_DIR},/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE/:/hfcache/,${llm_root_dir}:${llm_root_dir},${user_root_dir}:${user_root_dir}"

function add_timestamp() {

in_shar_dir=${llm_root_dir}
ctm_dir=${user_root_dir}
num_shard=10

read -r -d '' cmd <<EOF
cd ${CODE_DIR} \
&& export HF_HOME="/hfcache/" \
&& export TORCH_HOME="/hfcache/torch" \
&& export NEMO_CACHE_DIR="/hfcache/torch/nemo" \
&& export HF_DATASETS_CACHE="/hfcache/datasets" \
&& export TRANSFORMERS_CACHE="/hfcache/models" \
&& export TOKENIZERS_PARALLELISM=false \
&& export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 \
&& HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 python -u -B add_timestamp.py \
--in_shar_dir=${in_shar_dir} \
--ctm_dir=${ctm_dir} \
--num_shard=${num_shard}
EOF

OUTFILE=${RESULTS_DIR}/slurm-%j-%n-addts${manifest_index}.out
ERRFILE=${RESULTS_DIR}/error-%j-%n-addts${manifest_index}.out

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
# set -x
# bash -c "${cmd}"
# set +x
}

add_timestamp

exit 0

llm_root_dir=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/340b_8.19_daring_anteater_synth_v2_prompts_8801rm3.7p
user_root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex/340b_8.19_daring_anteater_synth_v2_prompts_8801rm3.7p/shar_duplex
# add_timestamp

# ./autorun.sh -n 1 /lustre/fsw/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/source_align/add_timestamp_anteater.sh