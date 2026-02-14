#!/bin/sh

ckpt_dir=${1:-"/lustre/fsw/portfolios/llmservice/users/kevinhu/projects/duplex-stt/result/EA_release/candidate_3/IAD_nano9b_parakeet600m_from_PT_32k_64gpu_5e-5_PT0.7_SFT0.05_QA0.02_TEXT0.1_loss0.5_MCQ0.03_ASR0.01_sysp0.03_NoiseProb0.5_SNR-30-60_asr_dtc2_dst15_loss_text5.0_bos10.0_eos5.0_pad1.0_eosplacementsfix_nospecaug_ei0.1_ot8_TN_all_data_v3.2_ir/checkpoints_hf_step-12556-last/eartts_rvq_cont_task_nanov2_tts_pretraining_wordlist_duplex_data_new_branch_4nodes_duplex_eartts_2_delay_4rd_stage_fp32_wd_1500_et_eos_dp_eos_dup_step_10004.ckpt"}

inf_dir=$ckpt_dir/fdb/pad0_bos0_eos0/validation_logs

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/branches/NeMo-release_not_rebased

# FDB
pred_wav_dir=${inf_dir}/pred_wavs
output_dir=${inf_dir}/metric
bash ${CODE_DIR}/scripts/speech_eval/run_full_duplex_bench_eval.sh ${pred_wav_dir} ${output_dir}

# VB
inf_dir=${ckpt_dir}/vb/validation_logs
pred_text_dir=${inf_dir}/metadatas
bash ${CODE_DIR}/scripts/speech_eval/eval_intel.sh ${pred_text_dir}

set +x