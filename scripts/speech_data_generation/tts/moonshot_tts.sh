#!/usr/bin/env bash
set -euo pipefail

echo "Setting up Moonshot TTS..."

# Check if Kimi-Audio is already installed
if python -c "from kimia_infer.api.kimia import KimiAudio" &> /dev/null; then
    echo "Kimi-Audio is already installed. Skipping installation."
else
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1

    # 1) New venv
    python -m venv --clear .venv-mooncast
    source .venv-mooncast/bin/activate
    python -m pip install -U pip setuptools wheel

    # 2) Install a CUDA-matched PyTorch (choose ONE index-url based on your CUDA runtime)
    # If your box has CUDA 12.4:
    pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
    # If you prefer CPU-only (no GPU):
    # pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

    # 3) Keep the numeric stack simple (avoid NumPy 2.x churn)
    pip install "numpy<2" soundfile librosa onnxruntime  # add onnxruntime-gpu if repo suggests

    # 4) Get MoonCast source
    git clone https://github.com/<ORG_OR_USER>/MoonCast.git
    cd MoonCast

    # 5) Install MoonCast as a package (this is the key to fix 'modules.*' import errors)
    pip install -e .

    pip install -r requirements.txt
    pip install huggingface_hub

    # 4) Download pretrained weights
    python download_pretrain.py

    pip install "setuptools<81" -U

    # b) make sure core audio libs are present
    pip install soxr soundfile librosa torchaudio

    # c) pin transformers to a version that has both Wav2Vec2BertModel and SeamlessM4TFeatureExtractor
    pip install 'transformers[audio]==4.47.0' accelerate sentencepiece

    # d) if you installed flash-attn earlier and it’s causing trouble, skip it for now
    pip uninstall -y flash-attn || true
    export USE_FLASH_ATTENTION=0

    pip uninstall -y flash-attn torch torchvision torchaudio

    # install a wheel-friendly combo
    pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

    # now install flash-attn as a wheel (no build)
    pip install --only-binary=:all: "flash-attn==2.6.3"

    sudo apt-get update && sudo apt-get install -y build-essential ninja-build

    # 2) Environment to speed up/ensure correct build on H100
    export PYTHONNOUSERSITE=1
    export CUDA_HOME=/usr/local/cuda             # adjust if your CUDA is elsewhere
    export TORCH_CUDA_ARCH_LIST="90"             # H100 = sm_90 (build only this arch)
    export MAX_JOBS=$(nproc)

    # 3) Build/install from source against *your* torch
    pip install -U packaging
    pip install -v --no-build-isolation --no-cache-dir flash-attn

    pip install pydub
    # and since pydub shells out to ffmpeg for reading/writing formats:
    sudo apt-get update && sudo apt-get install -y ffmpeg



    echo "Moonshot TTS setup complete!"
    echo ""
    echo "Usage:"
    echo "python moonshot_tts.py --wav_dir /path/to/wav/files --out_dir /path/to/output --transcribe --paraphrase_user_text"
    echo ""
    echo "Moonshot Kimi-Audio is a high-quality TTS model from Moonshot AI."
fi

exit 0

CODE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/s2s/NeMo/scripts/speech_data_generation
BASE_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_moonshot
WAV_DIR=$BASE_DIR

export HF_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HUGGINGFACE_HUB_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/torch"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/torch/nemo"
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/datasets"
export TRANSFORMERS_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/models"

# Run without paraphrasing
OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_moonshot
python ${CODE_DIR}/moonshot_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR}

# Run with paraphrasing
# OUT_DIR=/lustre/fsw/portfolios/convai/users/kevinhu/data/candor_turn_taking_paraphrase_moonshot
# python ${CODE_DIR}/moonshot_tts.py --wav_dir ${WAV_DIR} --transcribe --out_dir ${OUT_DIR} --paraphrase_user_text --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct
