#!/bin/bash
#SBATCH -A convai_convaird_nemo-speech
#SBATCH -J "convai_convaird_nemo-speech:duplex_demo_zhehuai_without_david_ed_scale"
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 4  # number of nodes
#SBATCH -t 4:00:00              # wall time
#SBATCH --time-min 04:00:00  
#SBATCH --ntasks-per-node=8    # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0

set -x

# if [ -z "$1" ]; then
#     echo "First argument (random seed) is missing"
#     exit 1
# fi
# SEED="${1}"

GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
TOTAL_NUM_GPUS=`expr $GPUS_PER_NODE \* $SLURM_JOB_NUM_NODES`

WANDB="edfa402d01b985ab9ce1f53e7f63a1f7cbb1a3d5" # replace with your own WandB API key

CONTAINER=/lustre/fsw/portfolios/convai/users/ecasanova/docker_images/nemo_duplex_november_eartts.sqsh
CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/branches/NeMo-release_not_rebased

MOUNTS='--container-mounts=/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/:/workspace,/lustre:/lustre'

CONFIG_PATH=/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/scripts/configs/inference/
CONFIG_NAME="nanov2_demo_model_eartts_updated"


EXP_NAME="${CONFIG_NAME}_${SLURM_JOB_NUM_NODES}demo_qwen_8B_model_duplex_eartts_nanov2_eartts_new_special_tokens_no_context_test_eartts_updated"

RESULTS_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/results/inferences/1.78kbps/${EXP_NAME}"
mkdir -p ${RESULTS_DIR}

PROJECT_NAME="salm_s2s_speech_decoder_v2_new_codebase_chars_emb"

MODEL_PATH=/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Nemotron-VoiceChat-november/duplex-eartts-fp32-stt-3-december_stt_edresson_model_R_digits_norm_eip_0.1_EA_model_step_9005/

model_subdir=$(basename "${MODEL_PATH}")
RESULTS_DIR="${RESULTS_DIR}/${model_subdir}"
mkdir -p ${RESULTS_DIR}


json_input=/lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/tmp/librivox-test-other_over20.json

read -r -d '' cmd <<EOF
export WANDB_API_KEY="${WANDB}" \
&& export AIS_ENDPOINT="http://asr.iad.oci.aistore.nvidia.com:51080" \
&& export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}" \
&& export HF_HOME="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/hfcache" \
&& export TORCH_HOME="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/hfcache/torch" \
&& export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/hfcache/torch/nemo" \
&& export TRITON_CACHE_DIR="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/hfcache/torch/triton" \
&& export OMP_NUM_THREADS=1 \
&& export WANDB_MODE=offline \
&& umask 000 \
&& export TOKENIZERS_PARALLELISM=false \
&& export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 \
&& HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 \
python /lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/scripts/eartts/inference_streaming_realtime_niva_json_debug.py \
    --model_path="${MODEL_PATH}" \
    --llm_checkpoint_path="${MODEL_PATH}" \
    --input_json="${json_input}" \
    --output_dir="${RESULTS_DIR}" \
    --output_json="${RESULTS_DIR}/output.json"
EOF

    # --decode_audio \

OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

CNAME=staszek


bash -c "${cmd}"



# srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" --container-name=$CNAME $MOUNTS bash -c "${cmd}"



# To run it uses+
# bash /lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/scripts/eartts/run_streaming_real_time.sh 42