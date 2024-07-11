import os
import torch
import torchaudio
from audioseal import AudioSeal

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

    # Create the output directory if it does not exist
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the audio file
    torchaudio.save(output_path, wav, sr)

# Generate a watermark in the audio
def generate_watermark(audio_path, output_path):
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

    # Create the watermarked audio
    watermarked_audio = wav + watermark

    # Save the watermarked audio to a file
    save_audio(watermarked_audio, sr, output_path)
    #print(f"Watermarked audio saved to {output_path}")
    return watermarked_audio, sr

# Detect if the audio is watermarked
def detect_watermark(audio_path):
    # Load the audio file
    wav, sr = load_audio(audio_path)

    # Ensure the audio tensor has the right shape (batch, channels, samples)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # add batch dimension if not present
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)  # add channel dimension if not present

    # Load the watermark detector model
    detector = AudioSeal.load_detector("audioseal_detector_16bits")

    # Detect the watermark
    result, message = detector.detect_watermark(wav, sr)

    # Print the detection results
    print(f"Detection result: {result}")
    #print(f"Detected message: {message}")

    # To detect the messages in the low-level
    result, message = detector(wav, sr)
    #print(f"Frame-wise detection result: {result[:, 1 , :]}")
    #print(f"Frame-wise detected message: {message}")

    return result, message

# Main function to generate a watermark and then detect it
def main(input_audio_path, output_audio_path):
    # Generate the watermark
    watermarked_audio, sr = generate_watermark(input_audio_path, output_audio_path)

    # Detect the watermark
    detect_watermark(output_audio_path)

# Paths for the input and output audio files
input_audio_path = "/Users/dmondal/Documents/iam/testing/Anushka/fake/1.wav"  # Replace with the path to your input audio file
output_audio_path = "output_audio.wav"  # Replace with the path to your output watermarked audio file

# Run the main function
main(input_audio_path, output_audio_path)
