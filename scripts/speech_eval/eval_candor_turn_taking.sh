#!/bin/bash

export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo

#####################
# Run one example
python $CODE_DIR/scripts/speech_eval/Full-Duplex-Bench/evaluation/eval_smooth_turn_taking.py \
  --root_dir $CODE_DIR/scripts/speech_eval/Full-Duplex-Bench/evaluation/smooth_turn_taking_example/