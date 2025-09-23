# from cosyvoice import CosyVoice
from cosyvoice.cli.cosyvoice import CosyVoice
import torchaudio, os

# Download weights from Hugging Face / ModelScope automatically
cv = CosyVoice("FunAudioLLM/CosyVoice-300M-SFT", fp16=True)

print("Speakers:", cv.list_avaliable_spks())
speaker = "英文女"   # or any printed English speaker

texts = [
    "Hello! This is a quick CosyVoice test using pip install.",
    "It generates English WAVs with one line of code."
]

os.makedirs("wavs_en", exist_ok=True)
for i, t in enumerate(texts, 1):
    out = cv.inference_sft(t, speaker)
    torchaudio.save(f"wavs_en/{i:02d}.wav", out["tts_speech"], out.get("sample_rate", 22050))
