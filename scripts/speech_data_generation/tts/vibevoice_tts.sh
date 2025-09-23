#! /bin/bash

git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice/
pip install -e .

export HF_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"

# python demo/inference_from_file.py --model_path microsoft/VibeVoice-Large --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice

cd /workspace/VibeVoice

# python demo/inference_from_file.py \
#   --model_path microsoft/VibeVoice-1.5B \
#   --txt_path /lustre/fsw/portfolios/convai/users/kevinhu/debug/vv_script.txt \
#   --speaker_names Alice Frank
# cp outputs/*_generated.wav /lustre/fsw/portfolios/convai/users/kevinhu/debug/

# CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo/scripts/speech_data_generation
# python ${CODE_DIR}/vibevoice_tts.py

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo/scripts/speech_data_generation
BASE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_vv
# OUT_DIR=$BASE_DIR/shar_duplex_transcribed
WAV_DIR=$BASE_DIR
python ${CODE_DIR}/vibevoice_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR}

exit 0
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_paraphrase
python ${CODE_DIR}/vibevoice_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR} --paraphrase_user_text --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct
