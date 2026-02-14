#!/bin/bash

# This is full-duplex bench eval based on agent text transcribed from speech. The results should be similar to the FDB eval in eval_all_s2s.sh

ckpt_dir=${1:-"/lustre/fsw/portfolios/llmservice/users/kevinhu/projects/duplex-stt/result/EA_release/candidate_3/IAD_nano9b_parakeet600m_from_PT_32k_64gpu_5e-5_PT0.7_SFT0.05_QA0.02_TEXT0.1_loss0.5_MCQ0.03_ASR0.01_sysp0.03_NoiseProb0.5_SNR-30-60_asr_dtc2_dst15_loss_text5.0_bos10.0_eos5.0_pad1.0_eosplacementsfix_nospecaug_ei0.1_ot8_TN_all_data_v3.2_ir/checkpoints_hf_step-12556-last"}
inf_dir=${ckpt_dir}/fdb/validation_logs
pred_text_dir=${inf_dir}/metadatas

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/branches/NeMo-release_not_rebased

bash ${CODE_DIR}/scripts/speech_eval/eval_candor_turn_taking.sh ${pred_text_dir}
bash ${CODE_DIR}/scripts/speech_eval/eval_user_interruption.sh ${pred_text_dir}
bash ${CODE_DIR}/scripts/speech_eval/eval_pause_handling.sh ${pred_text_dir}