#! /bin/bash

# 1) Get Python 3.10 on Ubuntu 24.04 (more compatible with pkuseg)
apt-get update && apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.10 python3.10-venv python3.10-distutils python3.10-dev build-essential

# 2) Fresh venv
python3.10 -m venv /opt/chatterbox310
source /opt/chatterbox310/bin/activate
python -m pip install -U pip setuptools wheel
pip install numpy
pip install --use-pep517 --no-build-isolation pkuseg==0.0.25

# 3) Install with deps (no hacks needed)
pip install chatterbox-tts

# 4) Smoke test
python - <<'PY'
import torch, torchaudio as ta
from chatterbox.tts import ChatterboxTTS
m = ChatterboxTTS.from_pretrained(device='cuda' if torch.cuda.is_available() else 'cpu')
w = m.generate('Hello from Chatterbox')
ta.save('out.wav', w.cpu(), m.sr)
print('Wrote out.wav')
PY
