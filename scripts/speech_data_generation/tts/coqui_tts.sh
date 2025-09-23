#!/usr/bin/env bash
set -euo pipefail

echo "Setting up Coqui TTS..."

# Update pip and install build tools
echo "Updating pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install Coqui TTS (much easier than Tortoise)
echo "Installing Coqui TTS..."
pip install TTS

# Install additional dependencies
echo "Installing additional dependencies..."
pip install torchaudio
pip install soundfile

echo "Coqui TTS setup complete!"
echo ""
echo "Usage:"
echo "python coqui_tts.py --wav_dir /path/to/wav/files --out_dir /path/to/output --transcribe --paraphrase_user_text"
echo ""
echo "Available voices: tts_models/en/ljspeech/tacotron2-DDC, tts_models/en/ljspeech/fast_pitch, tts_models/en/vctk/vits"

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo/scripts/speech_data_generation
BASE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_coqui
WAV_DIR=$BASE_DIR

# Run without paraphrasing
# python ${CODE_DIR}/coqui_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR}

# Run with paraphrasing
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_paraphrase_coqui
python ${CODE_DIR}/coqui_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR} --paraphrase_user_text --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct
