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

# Function to detect watermark
def detect_watermark(audio_path):
    # Load the audio file
    wav, sr = load_audio(audio_path)

    # Ensure the audio tensor has the right shape (batch, channels, samples)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # add batch dimension if not present
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)  # add channel dimension if not present

    # Load the detector model
    detector = AudioSeal.load_detector("audioseal_detector_16bits")

    # Detect watermark
    result, message = detector.detect_watermark(wav, sr)

    # Output the results
    print("Probability of watermark presence:", result)
    print("Extracted message (if any):", message)

    # For more detailed detection at frame level
    result_tensor, message_tensor = detector(wav, sr)
    print("Frame-level watermark probabilities:", result_tensor[:, 1, :])
    print("Bit-level probabilities:", message_tensor)

    # Determine if the audio is likely cloned
    if result > 0.5:
        print("The audio is likely watermarked (cloned).")
    else:
        print("The audio is unlikely to be watermarked (cloned).")

# Path to the user-input audio file
audio_path = "/Users/dmondal/Documents/iam/cloned.wav"

# Detect if the audio file is cloned
detect_watermark(audio_path)
