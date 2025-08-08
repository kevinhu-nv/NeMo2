#!/bin/bash

export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo

#####################
# Run ASR first
# Demo ckpt from recipe 1: https://docs.google.com/document/d/17z9JeCK8Ht-ojLp1H8ynJqr8Pz7aGdohy0C0PYIvXvY/edit?pli=1&tab=t.0#bookmark=id.7im551u1k7e0
pred_audio_dir=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/results/f2f-test/validation_logs/pred_wavs/
# python $CODE_DIR/scripts/speech_eval/Full-Duplex-Bench/get_transcript/asr.py --root_dir $pred_audio_dir --task full --stereo

#########################
# Prepare json files
# Copy all candor dataset to pred_audio_dir
candor_dataset_dir="/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking"
cp -r ${candor_dataset_dir} ${pred_audio_dir}
# Copy output jsons to the above dir
dataset_name="candor"
new_candor_dataset_dir="${pred_audio_dir}/candor_turn_taking"
# For each JSON file matching candor_*.json in $pred_audio_dir,
# remove the prefix, get the number, and copy/rename to the corresponding subdir as output.json
for json_file in "$pred_audio_dir"/${dataset_name}_*.json; do
  # Get the base filename
  base_json_file=$(basename "$json_file")
  # Remove the prefix and suffix to get the number (e.g., candor_123.json -> 123)
  number="${base_json_file#${dataset_name}_}"
  number="${number%.json}"
  # Target directory
  target_dir="${new_candor_dataset_dir}/${number}"
  # Only copy if the target directory exists
  if [ -d "$target_dir" ]; then
    cp "$json_file" "${target_dir}/output.json"
  else
    echo "Warning: target directory $target_dir does not exist, skipping $json_file"
  fi
done


#####################
# Evaluate turn taking
python $CODE_DIR/scripts/speech_eval/Full-Duplex-Bench/evaluation/eval_smooth_turn_taking.py \
  --root_dir ${new_candor_dataset_dir}



# python $CODE_DIR/scripts/speech_eval/Full-Duplex-Bench/evaluation/eval_smooth_turn_taking.py \
#   --root_dir $CODE_DIR/scripts/speech_eval/Full-Duplex-Bench/evaluation/smooth_turn_taking_example/