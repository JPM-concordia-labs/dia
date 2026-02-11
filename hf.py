import os

import torch
import torchaudio
from transformers import AutoProcessor, DiaForConditionalGeneration


torch_device = "cuda" if torch.cuda.is_available() else "cpu"
model_checkpoint = "./models/dia-hf"
output_path = "same_voice.mp3"

# Use a fixed seed for repeatability.
seed = 333

# Keep this transcript exactly aligned with the reference audio.
reference_audio_path = "simple.mp3"
reference_transcript = (
    "[S1] Dia is an open weights text to dialogue model. "
    "[S2] You get full control over scripts and voices. "
    "[S1] Wow. Amazing. (laughs) "
    "[S2] Try it now on Git hub or Hugging Face."
)
text_to_generate = (
    "[S1] Hi, this is a same-voice consistency test. "
    "[S2] Great, I will keep the same voice and speaking style throughout this clip. "
    "[S1] Perfect, please continue with one more complete sentence so we can verify both speakers. "
    "[S2] Absolutely, this second speaker line should be clearly present in the final audio. "
    "[S1]"
)

processor = AutoProcessor.from_pretrained(model_checkpoint)
model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

generation_kwargs = dict(
    max_new_tokens=3072,
#    min_new_tokens=700,
    guidance_scale=3.0,
    temperature=2.0,
    top_p=0.90,
    top_k=40,
)

if os.path.exists(reference_audio_path):
    wav, sr = torchaudio.load(reference_audio_path)
    wav = wav.mean(dim=0)  # force mono
    if sr != 44100:
        wav = torchaudio.functional.resample(wav, sr, 44100)
    reference_audio = wav.numpy()

    conditioned_text = [f"{reference_transcript} {text_to_generate}"]
    inputs = processor(
        text=conditioned_text,
        audio=reference_audio,
        sampling_rate=44100,
        return_tensors="pt",
    ).to(torch_device)
    audio_prompt_len = processor.get_audio_prompt_len(inputs["decoder_attention_mask"])
    outputs = model.generate(**inputs, **generation_kwargs)
    audios = processor.batch_decode(outputs, audio_prompt_len=audio_prompt_len)
else:
    inputs = processor(text=[text_to_generate], padding=True, return_tensors="pt").to(torch_device)
    outputs = model.generate(**inputs, **generation_kwargs)
    audios = processor.batch_decode(outputs)

processor.save_audio(audios, output_path)
print(f"Saved: {output_path}")
