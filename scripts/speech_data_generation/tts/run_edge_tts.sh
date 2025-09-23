#!/usr/bin/env bash
set -euo pipefail

echo "Setting up Edge TTS..."

# Check if Edge TTS and Whisper are already installed
if python -c "import edge_tts, whisper" &> /dev/null; then
    echo "Edge TTS and Whisper are already installed. Skipping installation."
else
    # Update pip and install build tools
    echo "Updating pip and installing build tools..."
    pip install --upgrade pip

    # Install Edge TTS and dependencies
    echo "Installing Edge TTS and dependencies..."
    pip install edge-tts
    pip install openai-whisper
    pip install torchaudio
    pip install soundfile

    echo "Edge TTS setup complete!"
    echo ""
    echo "Usage:"
    echo "python run_edge_tts.py --wav_dir /path/to/wav/files --out_dir /path/to/output --transcribe --paraphrase_user_text"
    echo ""
    echo "Edge TTS is Microsoft's cloud-based text-to-speech service."
fi

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo/scripts/speech_data_generation
BASE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_edge_tts
WAV_DIR=$BASE_DIR

export HF_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HUGGINGFACE_HUB_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/torch"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/torch/nemo"
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/datasets"
export TRANSFORMERS_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/models"

# Run without paraphrasing
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_edge_tts
python ${CODE_DIR}/run_edge_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR} --randomize_voice

# Run with paraphrasing and voice randomization
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_paraphrase_edge_tts
python ${CODE_DIR}/run_edge_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR} --paraphrase_user_text --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct --randomize_voice

