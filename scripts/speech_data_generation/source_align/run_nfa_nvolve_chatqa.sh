#!/bin/bash
#SBATCH -A convai_convaird_nemo-speech
#SBATCH -J "convai_convaird_nemo-speech:tqa_nfa"
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1 # number of nodes
#SBATCH -t 00:40:00              # wall time
#SBATCH --time-min 00:40:00  
#SBATCH --ntasks-per-node=8    # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0

###########################
# use srun
SLURM_ACCOUNT=portfolios/llmservice
USERID=users/kevinhu
LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/${SLURM_ACCOUNT}
CODE_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/works/mod_speech_llm/code/NeMo_merge/

manifest_index="$1"
root_dir=/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex
manifest_dir=${root_dir}/nvolve_chatqa/shar_duplex
RESULTS_DIR=${root_dir}/nfa_log
mkdir -p ${RESULTS_DIR}

manifest=${manifest_dir}/nfa_manifest.jsonl
out_dir=${manifest}_align
mkdir -p ${out_dir}
lang=en


MOUNTS="--container-mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/llmservice/users/kevinhu/works/mod_speech_llm/code/NeMo_merge/:/code,/lustre/fsw/portfolios/llmservice/users/kevinhu/results/:/results,/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data:/data,/lustre/fsw:/lustre/fsw,/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE/:/hfcache/,/lustre/fsw/portfolios/llmservice/users/chchien/Data:/workspace/data,/lustre/fsw/portfolios/llmservice/users/zhehuaic/pretrained:/workspace/model,${root_dir}:${root_dir}"

CONTAINER="/lustre/fsw/portfolios/llmservice/users/zhehuaic/containers/nemo-steve-24.01.sqsh"

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& cd ${CODE_DIR} \
&& export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}" \
&& export HF_HOME="/hfcache/" \
&& export TORCH_HOME="/hfcache/torch" \
&& export NEMO_CACHE_DIR="/hfcache/torch/nemo" \
&& export HF_DATASETS_CACHE="/hfcache/datasets" \
&& export TRANSFORMERS_CACHE="/hfcache/models" \
&& export TOKENIZERS_PARALLELISM=false \
&& export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 \
&& HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 python -u -B /code/tools/nemo_forced_aligner/align.py \
batch_size=32 \
pretrained_name="stt_${lang}_fastconformer_hybrid_large_pc" \
manifest_filepath=${manifest} \
output_dir=${out_dir} \
save_output_file_formats=["ctm"]
EOF

OUTFILE=${RESULTS_DIR}/slurm-%j-align${manifest_index}-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-align${manifest_index}-%n.out
srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"