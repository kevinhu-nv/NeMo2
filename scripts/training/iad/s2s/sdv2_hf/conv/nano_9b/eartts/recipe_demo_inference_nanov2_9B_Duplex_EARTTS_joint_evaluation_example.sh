#!/bin/bash
#SBATCH -A convai_convaird_nemo-speech
#SBATCH -J "convai_convaird_nemo-speech:duplex_demo_zhehuai_without_david_ed_scale"
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1  # number of nodes
#SBATCH -t 4:00:00              # wall time
#SBATCH --time-min 04:00:00  
#SBATCH --ntasks-per-node=8    # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0

set -x

if [ -z "$1" ]; then
    echo "First argument (random seed) is missing"
    exit 1
fi
SEED="${1}"

GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
TOTAL_NUM_GPUS=`expr $GPUS_PER_NODE \* $SLURM_JOB_NUM_NODES`

WANDB="edfa402d01b985ab9ce1f53e7f63a1f7cbb1a3d5" # replace with your own WandB API key

CONTAINER=/lustre/fsw/portfolios/convai/users/ecasanova/docker_images/nemo_duplex_november_eartts.sqsh
CODE_DIR=/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/branches/NeMo-release_not_rebased

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


EXP_NAME="${CONFIG_NAME}_${SLURM_JOB_NUM_NODES}demo_qwen_8B_model_duplex_eartts_nanov2_eartts_new_special_tokens_no_context_test_eartts_fp32_kevin_ckpt"

RESULTS_DIR="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/results/inferences/1.78kbps/${EXP_NAME}"
mkdir -p ${RESULTS_DIR}

PROJECT_NAME="salm_s2s_speech_decoder_v2_new_codebase_chars_emb"

# 20 nov
# ++model.stt.model.pretrained_s2s_model="/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/IAD_Nano-9B_SFT_Parakeet600m_asr_sp_from_checkpoints_hf_32002_64gpu_PT0.65_SFT0.05_QA0.02_TEXT0.1_MCQ0.03_ASR0.01_SP0.05-b200_LR5e-5_na0.5-30-60_ci_sa15_la2_tls10.0_facuda_val200.v2/checkpoints_hf_6354/" \
# 21 nov
#  ++model.stt.model.pretrained_s2s_model="/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge//exp_SFT_9b/IAD_Nano-9B_SFT_Parakeet600m_asr_sp_from_checkpoints_hf_32002_64gpu_PT0.7_SFT0.05_QA0.02_TEXT0.1_MCQ0.03_ASR0.01_SP0.01_LR5e-5_na0.5-30-60_ci_sa15_la2_tls10.0_facuda_val200.v2/checkpoints_hf_4208/" \

# 22 nov
# eos 4
# ++model.stt.model.pretrained_s2s_model="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/results/exp/nov25/exp_SFT_9b/IAD_Nano-9B_SFT_Parakeet600m_asr_sp_from_checkpoints_hf_32002_64gpu_PT0.65_SFT0.05_QA0.02_TEXT0.1_MCQ0.03_ASR0.01_SP0.01_LR5e-5_na0.5-30-60_ci_sa15_la2_tls10.0_facuda_val200.v2_delay_eos_4/checkpoints_hf_5779/" \
# kevin checkpoint
# ++model.stt.model.pretrained_s2s_model="/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/IAD_Nano-9B_SFT_Parakeet600m_asr_sp_from_checkpoints_hf_32002_64gpu_PT0.7_SFT0.05_QA0.02_TEXT0.1_MCQ0.03_ASR0.01_SP0.01_LR5e-5_na0.5-30-60_ci_sa15_la2_tls10.0_facuda_val200.v2/checkpoints_hf_11127/" \

# EARTTS fp16
#  ++model.speech_generation.model.pretrained_model="/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Duplex_EARTTS/eartts_rvq_cont_task_nanov2_tts_pretraining_wordlist_duplex_data_4nodes_FT_cont_22khz_12.5FPS_nanov2_no_llm_emb_subword_emb_bos_eos_emb_wordlist_data_mask_prompt_loss_no_text_norm_2_delayfp32_3rd_stage_wd_gat_fp16_tdf_step_74011-last.ckpt" \

# EARTTS fp32
#  ++model.speech_generation.model.pretrained_model="/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Duplex_EARTTS/eartts_rvq_cont_task_nanov2_tts_pretraining_wordlist_duplex_data_4nodes_FT_cont_22khz_12.5FPS_nanov2_no_llm_emb_subword_emb_bos_eos_emb_wordlist_data_mask_prompt_loss_no_text_norm_2_delay_fp32_3rd_stage_wd_gat_tts_data_fix_step_64010.ckpt" \

# 24 nov - ankita
# ++model.stt.model.pretrained_s2s_model="/lustre/fsw/portfolios/llmservice/users/apasad/projects/nemo_s2s_interim_merged_cc_nov/exp_SFT-nano9b/IAD_nano9b_parakeet600m_from_32k_64gpu_5e-5_PT0.55_SFT0.1_QA0.02_TEXT0.1_loss0.5_MCQ0.03_ASR0.01_sysp0.2_NoiseProb0.5_SNR-30-60_asr_dtc0_dst15_loss_text5.0_bos10.0_eos5.0_pad1.0_fixed_mantis_gretel_2511_v2.1_approved_SFT_data//checkpoints_hf_8202/" \

# && pip install nv-one-logger-core==2.1.0 nv-one-logger-pytorch-lightning-integration==2.1.0 nv-one-logger-training-telemetry==2.1.0 kaldialign==0.9.1 lhotse==1.31.1 \
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
python ${CODE_DIR}/examples/speechlm2/nemotron_voicechat_infer.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    exp_manager.name=${EXP_NAME} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++model.stt.model.eval_text_turn_taking=True \
     ++model.stt.model.pretrained_s2s_model="/lustre/fsw/portfolios/llmservice/users/kevinhu/code/s2s_pretrain_20251022_merge/exp_SFT_9b/IAD_Nano-9B_SFT_Parakeet600m_asr_sp_from_checkpoints_hf_32002_64gpu_PT0.7_SFT0.05_QA0.02_TEXT0.1_MCQ0.03_ASR0.01_SP0.01_LR5e-5_na0.5-30-60_ci_sa15_la2_tls10.0_facuda_val200.v2/checkpoints_hf_11127/" \
    ++model.stt.model.pretrained_asr=/lustre/fsw/portfolios/llmservice/users/kdhawan/models/cache-aware/oci-N-8_G-8_cacheaware-600M_granary-gsc_PC_multilookahead.nemo \
    ++model.speech_generation.model.pretrained_model="/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Duplex_EARTTS/eartts_rvq_cont_task_nanov2_tts_pretraining_wordlist_duplex_data_4nodes_FT_cont_22khz_12.5FPS_nanov2_no_llm_emb_subword_emb_bos_eos_emb_wordlist_data_mask_prompt_loss_no_text_norm_2_delay_fp32_3rd_stage_wd_gat_tts_data_fix_step_64010.ckpt" \
    ++model.inference_speaker_reference="/lustre/fsw/portfolios/convai/users/ecasanova/S2S-full-duplex/inference_references/Emma_S3_A1_SC7_singleturntarget_21_channel_1_audio_in.wav" \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    exp_manager.explicit_log_dir=${RESULTS_DIR} \
    data.train_ds.seed=$SEED \
    data.train_ds.batch_duration=100 \
    data.validation_ds.batch_size=2 \
    ++trainer.limit_val_batches=1 \
    ++trainer.precision=32 \
    ++trainer.max_steps=1 \
    ++trainer.val_check_interval=1 \
    data.validation_ds.seed=$SEED
EOF


# To export the checkpoint to HF format use:
# ++hf_export_dir="/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Nemotron-VoiceChat-november/duplex-eartts-fp16-stt-22-november_stt_eos_4_fp16_eartts/" \
# fp16 eartts model
# ++model.speech_generation.model.pretrained_model="/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Duplex_EARTTS/eartts_rvq_cont_task_nanov2_tts_pretraining_wordlist_duplex_data_4nodes_FT_cont_22khz_12.5FPS_nanov2_no_llm_emb_subword_emb_bos_eos_emb_wordlist_data_mask_prompt_loss_no_text_norm_2_delayfp32_3rd_stage_wd_gat_fp16_tdf_step_74011-last.ckpt" \
# fp32 eartts model
# ++model.speech_generation.model.pretrained_model="/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Duplex_EARTTS/eartts_rvq_cont_task_nanov2_tts_pretraining_wordlist_duplex_data_4nodes_FT_cont_22khz_12.5FPS_nanov2_no_llm_emb_subword_emb_bos_eos_emb_wordlist_data_mask_prompt_loss_no_text_norm_2_delay_fp32_3rd_stage_wd_gat_tts_data_fix_step_64010.ckpt" \

# export huggingface model for inference
#  ++hf_export_dir="/lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Nemotron-VoiceChat-november/duplex-eartts-fp16-stt-20-november/" \
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

CNAME=staszek


bash -c "${cmd}"


# srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" --container-name=$CNAME $MOUNTS bash -c "${cmd}"



# To run it uses+
# bash /lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/scripts/eartts/recipe_demo_inference_nanov2_9B_Duplex_EARTTS_joint_evaluation_example.sh 42