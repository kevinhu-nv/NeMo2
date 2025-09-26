#!/bin/sh

# Define experiment configurations
EXPERIMENTS=(
    # "DFW_qwen_1b_convasr_joint_4nodes_left2_asrl3_txtl3_et12_eb2:step-11896-last"
    # "DFW_qwen_1b_convasr_joint_4nodes_left4_asrl3_txtl3_et12_eb2:step-11790-last"
    # "DFW_qwen_1b_convasr_joint_4nodes_left8_asrl3_txtl3_et12_eb2:step-3967-last"
    # "DFW_qwen_1b_convasr_joint_4nodes_sa_left2_asrl3_txtl3_et12_eb2:step-11002-last"
    # "DFW_qwen_1b_convasr_joint_4nodes_sa_left4_asrl3_txtl3_et12_eb2.fixval:step-19087-last"
    "DFW_qwen_1b_convasr_joint_4nodes_sa_left8_asrl3_txtl3_et12_eb2:step-11490-last"
)

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo

# Loop through all experiments
for exp_config in "${EXPERIMENTS[@]}"; do
    # Skip commented lines
    if [[ $exp_config == \#* ]]; then
        continue
    fi
    
    # Split experiment name and checkpoint
    IFS=':' read -r EXP_NAME CKPT_NAME <<< "$exp_config"
    
    echo "=========================================="
    echo "Running evaluation for: ${EXP_NAME}"
    echo "Checkpoint: ${CKPT_NAME}"
    echo "=========================================="
    
    # Run evaluations
    bash ${CODE_DIR}/scripts/speech_eval/eval_candor_turn_taking_text.sh ${EXP_NAME} ${CKPT_NAME}
    sleep 5
    bash ${CODE_DIR}/scripts/speech_eval/eval_conv.sh ${EXP_NAME} ${CKPT_NAME}
    sleep 5
    # bash ${CODE_DIR}/scripts/speech_eval/eval_intel.sh ${EXP_NAME} ${CKPT_NAME}
    
    echo "Completed evaluation for: ${EXP_NAME}"
    echo ""
done

echo "All evaluations completed!"
