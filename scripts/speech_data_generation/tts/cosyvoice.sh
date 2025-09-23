#! /bin/bash

test -d CosyVoice || git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -q -r requirements.txt
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("model-scope/CosyVoice-300M-SFT",
                  local_dir="pretrained_models/CosyVoice-300M-SFT")
PY
# CosyVoice expects these on PYTHONPATH:
export PYTHONPATH=third_party/AcademiCodec:third_party/Matcha-TTS