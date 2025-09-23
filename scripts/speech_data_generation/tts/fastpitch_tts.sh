#!/usr/bin/env bash
set -euo pipefail

echo "Setting up FastPitch TTS..."

# Check if FastPitch (nemo_toolkit[tts]) is already installed
if python -c "import nemo.collections.tts" &> /dev/null; then
    echo "FastPitch (nemo_toolkit[tts]) is already installed. Skipping installation."
else
    # Update pip and install build tools
    echo "Updating pip and installing build tools..."
    pip install --upgrade pip

    # Install NeMo TTS (which includes FastPitch)
    echo "Installing NeMo TTS..."
    pip install nemo_toolkit[tts]

    # Install additional dependencies
    echo "Installing additional dependencies..."
    pip install torchaudio
    pip install soundfile

    echo "FastPitch TTS setup complete!"
    echo ""
    echo "Usage:"
    echo "python fastpitch_tts.py --wav_dir /path/to/wav/files --out_dir /path/to/output --transcribe --paraphrase_user_text"
    echo ""
    echo "FastPitch is a high-quality, fast TTS model from NVIDIA."
fi

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo/scripts/speech_data_generation
BASE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_fastpitch
WAV_DIR=$BASE_DIR

export HF_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/torch"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/torch/nemo"
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/datasets"
export TRANSFORMERS_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/models"

# Run without paraphrasing
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_fastpitch
python ${CODE_DIR}/fastpitch_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR}

# Run with paraphrasing
# OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_paraphrase_fastpitch
# python ${CODE_DIR}/fastpitch_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR} --paraphrase_user_text --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct
