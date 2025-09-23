#!/usr/bin/env bash

set -euo pipefail

export HF_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export HUGGINGFACE_HUB_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
export TORCH_HOME="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/torch"
export NEMO_CACHE_DIR="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/torch/nemo"
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/datasets"
export TRANSFORMERS_CACHE="/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE/models"

echo "Setting up MoonCast TTS..."

unset PYTHONPATH
export PYTHONNOUSERSITE=1

# 1) New venv
python -m venv --clear .venv-mooncast
source .venv-mooncast/bin/activate

python -m pip install -U pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install "numpy<2" soundfile librosa onnxruntime

git clone https://github.com/jzq2000/MoonCast.git
cd MoonCast
pip install -r requirements.txt

pip install huggingface_hub
python download_pretrain.py

# from the MoonCast venv
pip uninstall -y flash-attn || true
export USE_FLASH_ATTENTION=0 
pip install "setuptools<81" -U
# b) make sure core audio libs are present
pip install soxr soundfile librosa torchaudio
# c) pin transformers to a version that has both Wav2Vec2BertModel and SeamlessM4TFeatureExtractor
pip install 'transformers[audio]==4.47.0' accelerate sentencepiece
# d) if you installed flash-attn earlier and it’s causing trouble, skip it for now
pip uninstall -y flash-attn || true
export USE_FLASH_ATTENTION=0
# pip uninstall -y flash-attn torch torchvision torchaudio
# install a wheel-friendly combo
pip install --index-url https://download.pytorch.org/whl/cu121 \
torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
# 2) Environment to speed up/ensure correct build on H100
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="90"
export MAX_JOBS=$(nproc)
# 3) Build/install from source against *your* torch
pip install -U packaging
pip install -v --no-build-isolation --no-cache-dir flash-attn
pip install pydub


echo "Moonshot TTS setup complete!"
echo ""
echo "Usage:"
echo "python moonshot_tts.py --wav_dir /path/to/wav/files --out_dir /path/to/output --transcribe --paraphrase_user_text"
echo ""
echo "Moonshot Kimi-Audio is a high-quality TTS model from Moonshot AI."


# Test run: simple one-liner: synthesize a single sentence -> hello.mp3
python - <<'PY'
import base64
from inference import Model
m = Model()
js = {"dialogue":[{"role":"0","text":"Hello MoonCast, this is a quick test."}]}
b64 = m.inference(js)  # non-streaming, no prompt
open("hello.mp3","wb").write(base64.b64decode(b64))
print("wrote hello.mp3")
PY