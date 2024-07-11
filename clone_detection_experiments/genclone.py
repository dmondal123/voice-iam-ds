import torch
import torchaudio
from allaudio import AudioSeal

# Load the audio file
def load_audio(file_path, target_sample_rate=16000):
    wav, sr = torchaudio.load(file_path)
    if sr != target_sample_rate:
        wav = torchaudio.transforms.Resample(sr, target_sample_rate)(wav)
        sr = target_sample_rate
    return wav, sr

# Save the audio file
def save_audio(wav, sr, output_path):
    # Ensure the wav tensor has the shape (channels, samples)
    if wav.dim() == 3:
        wav = wav.squeeze(0)  # remove batch dimension if present
    torchaudio.save(output_path, wav, sr)

# Function to generate a clone of the voice
def generate_clone(audio_path, output_path):
    # Load the audio file
    wav, sr = load_audio(audio_path)

    # Ensure the audio tensor has the right shape (batch, channels, samples)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # add batch dimension if not present
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)  # add channel dimension if not present

    # Load the watermark generator model
    model = AudioSeal.load_generator("audioseal_wm_16bits")

    # Generate the watermark
    watermark = model.get_watermark(wav, sr)

    # Optional: add a 16-bit message to embed in the watermark
    # msg = torch.randint(0, 2, (wav.shape(0), model.msg_processor.nbits), device=wav.device)
    # watermark = model.get_watermark(wav, message=msg)

    # Create the watermarked (cloned) audio
    watermarked_audio = wav + watermark

    # Save the cloned audio to a file
    save_audio(watermarked_audio, sr, output_path)
    print(f"Cloned audio saved to {output_path}")

# Path to the original audio file
audio_path = "/Users/dmondal/Documents/iam/voice_based_iam/Diya/001.wav"
# Path to save the cloned audio file
output_path = "cloned.wav"

# Generate the cloned audio
generate_clone(audio_path, output_path)
