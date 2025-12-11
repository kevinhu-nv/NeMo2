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

##SBATCH -A llmservice_nemo_speechlm
##SBATCH -J "llmservice_nemo_speechlm:s2s_otf_duplex_4b_squadv2_baseline_14_feb_speech_decoder_v2_fix_kv_fix_f_s"

##SBATCH -A convai_convaird_nemo-speech
##SBATCH -J "convai_convaird_nemo-speech:speech_decoder_v2_1.78kbps_codec_spk_emb_codec_real_conv_data_35_perc_tts_25_perc"






# tts dp bos eos
# ++model.pretrained_tts_from_s2s="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/results/exp/1.78kbps/demo_model_no_aug_chen_chen_asr_llm_frozen_4nodes_nonsil10.0_zhehuai_01_jul_g_baseline_no_davidai_qwen_FT_speech_dec_only_chars_dp_eos_bos/checkpoints/bk/step\=52004-last.ckpt" \
# tts derop eos, bos, sil aug
# ++model.pretrained_tts_from_s2s="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/results/exp/1.78kbps/demo_model_no_aug_chen_chen_sil_asr_llm_frozen_4nodes_nonsil10.0_2nd_stage_qwen_sil_aug_cc_data_tts_data_fix_bos_eos_dp/checkpoints/bk/step\=9440-last.ckpt" \

# Models
# ++model.pretrained_s2s_model="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/results/exp/1.78kbps/demo_model_no_aug_chen_chen_4nodes_nonsil10.0_zhehuai_01_jul_g_baseline_no_davidai_qwen_no_lat_inter_data/checkpoints/step\=32003-last.ckpt" \


MOUNTS='--container-mounts=/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/:/workspace,/lustre:/lustre'

CONFIG_PATH=/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/scripts/configs/inference/
# CONFIG_NAME="demo_model_no_aug_user_silence"
CONFIG_NAME="nanov2_demo_model_eartts_updated"


EXP_NAME="${CONFIG_NAME}_${SLURM_JOB_NUM_NODES}demo_qwen_8B_model_duplex_eartts_nanov2_eartts_new_special_tokens_no_context_test_eartts_updated"

RESULTS_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/results/inferences/1.78kbps/${EXP_NAME}"
mkdir -p ${RESULTS_DIR}

PROJECT_NAME="salm_s2s_speech_decoder_v2_new_codebase_chars_emb"

# baseline model
#  ++model.pretrained_s2s_model="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/results/exp/1.78kbps/demo_model_no_aug_chen_chen_4nodes_nonsil10.0_zhehuai_01_jul_g_baseline_no_davidai_qwen_no_lat/checkpoints/step\=29003.ckpt" \

# ++model.speech_generation.model.pretrained_model="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/results/exp/eartts/eartts_rvq_cont_task_qwen_22khz_interruptions_only_4nodes_FT_cont_22khz_12.5FPS_sync_duplex_data_interruption_only_fix/checkpoints/step\=24005-last.ckpt" \
# && pip install nv-one-logger-core==2.1.0 nv-one-logger-pytorch-lightning-integration==2.1.0 nv-one-logger-training-telemetry==2.1.0 kaldialign==0.9.1 lhotse==1.31.1 \

# audio_path="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/scripts/source_test_sample.wav"
# audio_path='/lustre/fsw/portfolios/convai/users/ecasanova/edresson_internal_eval_dataset_samples/edresson_buying_house.wav'
audio_path='/lustre/fsw/portfolios/llmservice/users/kevinhu/data/kevin_17oct/kevin_game2_pm17_wic_vm.m4a'

# MODEL_PATH=/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Nemotron-VoiceChat-november/duplex-eartts-fp16-stt-21-november/
# MODEL_PATH="/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Nemotron-VoiceChat-november/duplex-eartts-fp16-stt-22-november_stt_eos_4/"
MODEL_PATH=/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Nemotron-VoiceChat-november/duplex-eartts-fp16-stt-22-november_stt_v2_fp32/

audio_name=$(basename ${audio_path} | sed 's/\.[^.]*$//')

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
python /lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/scripts/eartts/inference_streaming_realtime_updated.py \
    --model_path="${MODEL_PATH}" \
    --llm_checkpoint_path="${MODEL_PATH}" \
    --audio_path="${audio_path}" \
    --decode_audio \
    --output_text="${RESULTS_DIR}/output_${audio_name}.txt" \
    --output_asr_text="${RESULTS_DIR}/output_asr_text_${audio_name}.txt" \
    --output_audio="${RESULTS_DIR}/output_audio_${audio_name}.wav"
EOF




# fp16 eartts model
# ++model.speech_generation.model.pretrained_model="/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Duplex_EARTTS/eartts_rvq_cont_task_nanov2_tts_pretraining_wordlist_duplex_data_4nodes_FT_cont_22khz_12.5FPS_nanov2_no_llm_emb_subword_emb_bos_eos_emb_wordlist_data_mask_prompt_loss_no_text_norm_2_delayfp32_3rd_stage_wd_gat_fp16_tdf_step_74011-last.ckpt" \
# fp32 eartts model
# ++model.speech_generation.model.pretrained_model="/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Duplex_EARTTS/eartts_rvq_cont_task_nanov2_tts_pretraining_wordlist_duplex_data_4nodes_FT_cont_22khz_12.5FPS_nanov2_no_llm_emb_subword_emb_bos_eos_emb_wordlist_data_mask_prompt_loss_no_text_norm_2_delay_fp32_3rd_stage_wd_gat_tts_data_fix_step_64010.ckpt" \

#     data.validation_ds.seed=$SEED
#     ++model.inference_eos_boost=2 \
# for debug: ++trainer.val_check_interval=1 \
#trainer.strategy.data_parallel_size=${TOTAL_NUM_GPUS} \
#     ++model.perception.modality_adapter_quantizer_levels=[8,8,8,6,5] \
#  ++model.debug_dataloader_audios_path=/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/debug_samples/ \
#  ++hf_export_dir="/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/DuplexS2S/Duplex_S2S_Nanov2_30_set_hf/" \ export

OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

CNAME=staszek


bash -c "${cmd}"


# srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" --container-name=$CNAME $MOUNTS bash -c "${cmd}"



# To run it uses+
# bash /lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/scripts/eartts/run_streaming_real_time.sh 42