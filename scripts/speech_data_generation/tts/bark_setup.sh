#!/usr/bin/env bash
set -euo pipefail

echo "Setting up Bark TTS..."

# Update pip and install build tools
echo "Updating pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install Bark TTS (much easier than Tortoise)
echo "Installing Bark TTS..."
pip install bark

# Install additional dependencies
echo "Installing additional dependencies..."
pip install soundfile
pip install torchaudio
pip install transformers

echo "Bark TTS setup complete!"
echo ""
echo "Usage:"
echo "python bark_tts.py --wav_dir /path/to/wav/files --out_dir /path/to/output --transcribe --paraphrase_user_text"
echo ""
echo "Available voices: v2/en_speaker_0 through v2/en_speaker_9"

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo/scripts/speech_data_generation
BASE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_bark
WAV_DIR=$BASE_DIR

# Run with paraphrasing
python ${CODE_DIR}/bark_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR} --paraphrase_user_text --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct
