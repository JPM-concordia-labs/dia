import os

import torch
import torchaudio
from transformers import AutoProcessor, DiaForConditionalGeneration


torch_device = "cuda" if torch.cuda.is_available() else "cpu"
model_checkpoint = "./models/dia-hf"
output_path = "same_voice.mp3"

# Use a fixed seed for repeatability.
seed = 1234

# Keep this transcript exactly aligned with the reference audio.
reference_audio_path = "simple.mp3"
reference_transcript = (
    "[S1] Dia is an open weights text to dialogue model. "
    "[S2] You get full control over scripts and voices. "
    "[S1] Wow. Amazing. (laughs) "
    "[S2] Try it now on Git hub or Hugging Face."
)
text_to_generate = (
    "[S1] Hi, this is a same-voice test. "
    "[S2] Great, let's keep the same speaker style in this output."
)

processor = AutoProcessor.from_pretrained(model_checkpoint)
model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
generator = torch.Generator(device=torch_device).manual_seed(seed)

generation_kwargs = dict(
    max_new_tokens=3072,
    guidance_scale=3.0,
    temperature=1.0,
    top_p=0.90,
    top_k=45,
    generator=generator,
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
