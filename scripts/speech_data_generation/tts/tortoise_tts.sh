#!/usr/bin/env bash
set -euo pipefail

echo "Setting up Tortoise TTS..."

# Update pip and install build tools
echo "Updating pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install tokenizers separately to avoid build issues
echo "Installing tokenizers..."
pip install tokenizers

# Install additional dependencies
echo "Installing additional dependencies..."
pip install torchaudio
pip install transformers

# Install Tortoise TTS
echo "Installing Tortoise TTS..."
pip install tortoise-tts

echo "Tortoise TTS setup complete!"
echo ""
echo "Usage:"
echo "python tortoise_tts.py --wav_dir /path/to/wav/files --out_dir /path/to/output --transcribe --paraphrase_user_text"
echo ""
echo "Available voices: angie, daniel, emma, geralt, halle, jlaw, lj, mol, pat, william"

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo/scripts/speech_data_generation
BASE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_tortoise
WAV_DIR=$BASE_DIR
# python ${CODE_DIR}/tortoise_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR}

OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_paraphrase_tortoise
python ${CODE_DIR}/tortoise_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR} --paraphrase_user_text --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct

