#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_all.sh all [inference_out_dir] [full_duplex_bench_result_dir] [deployment_name] [original_data_base_path]
#   ./run_all.sh <single_dataset_name> [inference_out_dir] [full_duplex_bench_result_dir] [deployment_name] [original_data_base_path]
#
# INFERENCE_OUT_DIR: directory containing inference outputs from running the model on Full-Duplex-Bench (validation_logs/agent)
# FULL_DUPLEX_BENCH_RESULT_DIR: destination where the reorganized dataset and Full-Duplex-Bench metrics will be saved
#
# Supported dataset names:
#   candor_turn_taking | candor_pause_handling | synthetic_pause_handling | synthetic_user_interruption

export TORCH_HOME="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"
export HF_HOME="/lustre/fsw/portfolios/llmservice/users/kevinhu/results/HFCACHE"

# Inference outputs: where the S2S model predictions on Full-Duplex-Bench are saved
INFERENCE_OUT_DIR=${1:-"/lustre/fsw/portfolios/llmservice/users/kevinhu/projects/duplex-stt/result/EA_release/candidate_3/IAD_nano9b_parakeet600m_from_PT_32k_64gpu_5e-5_PT0.7_SFT0.05_QA0.02_TEXT0.1_loss0.5_MCQ0.03_ASR0.01_sysp0.03_NoiseProb0.5_SNR-30-60_asr_dtc2_dst15_loss_text5.0_bos10.0_eos5.0_pad1.0_eosplacementsfix_nospecaug_ei0.1_ot8_TN_all_data_v3.2_ir/checkpoints_hf_step-12556-last/eartts_rvq_cont_task_nanov2_tts_pretraining_wordlist_duplex_data_new_branch_4nodes_duplex_eartts_2_delay_4rd_stage_fp32_wd_1500_et_eos_dp_eos_dup_step_10004.ckpt/fdb/pad0_bos0_eos0/validation_logs/pred_wavs/"}  
FULL_DUPLEX_BENCH_RESULT_DIR=${2:-"/lustre/fsw/portfolios/llmservice/users/kevinhu/projects/duplex-stt/result/EA_release/candidate_3/IAD_nano9b_parakeet600m_from_PT_32k_64gpu_5e-5_PT0.7_SFT0.05_QA0.02_TEXT0.1_loss0.5_MCQ0.03_ASR0.01_sysp0.03_NoiseProb0.5_SNR-30-60_asr_dtc2_dst15_loss_text5.0_bos10.0_eos5.0_pad1.0_eosplacementsfix_nospecaug_ei0.1_ot8_TN_all_data_v3.2_ir/checkpoints_hf_step-12556-last/eartts_rvq_cont_task_nanov2_tts_pretraining_wordlist_duplex_data_new_branch_4nodes_duplex_eartts_2_delay_4rd_stage_fp32_wd_1500_et_eos_dp_eos_dup_step_10004.ckpt/fdb/pad0_bos0_eos0/metric"}  


REQUESTED=${3:-synthetic_pause_handling}
DEPLOYMENT_NAME=${4:-gpt-4o}
ORIGINAL_DATA_BASE_PATH=${5:-"/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/full-duplex/benchmarks/full_duplex_bench/original_data/v1.0"}
CLIENT_KEY_JSONL=${6:-"/lustre/fsw/portfolios/llmservice/users/kevinhu/HFCACHE/zh_openai_key.txt"}

if [[ "$REQUESTED" == "all" ]]; then
  DATASETS=(
    synthetic_user_interruption
    candor_turn_taking
    candor_pause_handling
    synthetic_pause_handling
  )
else
  DATASETS=("$REQUESTED")
fi

for DATASET_NAME in "${DATASETS[@]}"; do
  case "$DATASET_NAME" in
    candor_turn_taking)
      ORIGINAL_DATA_PATH="${ORIGINAL_DATA_BASE_PATH}/candor_turn_taking"
      EVAL_TASK="smooth_turn_taking"
      ;;
    candor_pause_handling)
      ORIGINAL_DATA_PATH="${ORIGINAL_DATA_BASE_PATH}/candor_pause_handling"
      EVAL_TASK="pause_handling"
      ;;
    synthetic_pause_handling)
      ORIGINAL_DATA_PATH="${ORIGINAL_DATA_BASE_PATH}/synthetic_pause_handling"
      EVAL_TASK="pause_handling"
      ;;
    synthetic_user_interruption)
      ORIGINAL_DATA_PATH="${ORIGINAL_DATA_BASE_PATH}/synthetic_user_interruption"
      EVAL_TASK="user_interruption"
      ;;
    *)
      echo "Unsupported dataset_name: $DATASET_NAME" >&2
      exit 1
      ;;
  esac

  VERSION_DIR=$(basename "$(dirname "$ORIGINAL_DATA_PATH")")
  DEST_DATASET_DIR="$FULL_DUPLEX_BENCH_RESULT_DIR/$VERSION_DIR/$DATASET_NAME"

  CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/branches/NeMo-release_not_rebased
  echo "[1/3][$DATASET_NAME] Reorganizing outputs -> $DEST_DATASET_DIR"
  python ${CODE_DIR}/scripts/speech_eval/reorganize_candor_outputs.py \
    --output_path "$INFERENCE_OUT_DIR" \
    --original_data_path "$ORIGINAL_DATA_PATH" \
    --revised_output_path "$FULL_DUPLEX_BENCH_RESULT_DIR" \
    --dataset_name "$DATASET_NAME" \
    --strict_ids \
    --clean_destination

  # Ensure destination dataset directory exists even if no files were copied (so downstream steps can still log)
  mkdir -p "$DEST_DATASET_DIR"

  CODE_DIR=/lustre/fsw/portfolios/llmservice/users/kevinhu/Full-Duplex-Bench-NV
  echo "[2/3][$DATASET_NAME] Transcribing with ASR for $DEST_DATASET_DIR"
  OUTPUT_FILE="$FULL_DUPLEX_BENCH_RESULT_DIR/$VERSION_DIR/asr_${DATASET_NAME}.log"
  # Use specialized ASR task for user_interruption; otherwise use full
  if [[ "$EVAL_TASK" == "user_interruption" ]]; then
    ASR_TASK="user_interruption"
  else
    ASR_TASK="full"
  fi
  python ${CODE_DIR}/get_transcript/asr.py --root_dir "$DEST_DATASET_DIR" --stereo --task "$ASR_TASK" | tee "$OUTPUT_FILE"
  echo "ASR done, outputting to $OUTPUT_FILE"

  echo "[3/3][$DATASET_NAME] Running evaluation task '$EVAL_TASK'"
  EVAL_ARGS=(
    --task "$EVAL_TASK"
    --root_dir "$DEST_DATASET_DIR"
    --deployment_name "$DEPLOYMENT_NAME"
  )
  if [[ "$EVAL_TASK" == "user_interruption" ]]; then
    if [[ -z "$CLIENT_KEY_JSONL" ]]; then
      echo "CLIENT_KEY_JSONL must be provided for task '$EVAL_TASK'." >&2
      exit 1
    fi
    EVAL_ARGS+=(--client_key_jsonl "$CLIENT_KEY_JSONL")
  fi

  python ${CODE_DIR}/evaluation/evaluate.py "${EVAL_ARGS[@]}" | tee "$FULL_DUPLEX_BENCH_RESULT_DIR/$VERSION_DIR/eval_${DATASET_NAME}.log"
done

echo "All requested datasets completed: ${DATASETS[*]}"
