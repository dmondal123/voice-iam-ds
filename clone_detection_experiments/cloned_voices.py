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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the audio file
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

# Main function to clone all voices
def clone_all_voices(input_base_dir, output_base_dir):
    # List of subfolders
    subfolders = ['diya', 'anushka', 'saqib', 'harsh', 'sneha']
    
    for subfolder in subfolders:
        input_folder = os.path.join(input_base_dir, subfolder)
        output_folder = os.path.join(output_base_dir, subfolder)
        
        # Iterate through all audio files in the input folder
        for i in range(1, 51):  # Assuming files are named 001.wav to 050.wav
            input_audio_path = os.path.join(input_folder, f"{i:03d}.wav")
            output_audio_path = os.path.join(output_folder, f"clone_{i}.wav")
            
            if os.path.exists(input_audio_path):
                generate_clone(input_audio_path, output_audio_path)
            else:
                print(f"File {input_audio_path} does not exist")

# Base directory containing the original audio files
input_base_dir = "voice_based_iam"
# Base directory to save the cloned audio files
output_base_dir = "cloned_voices"

# Clone all voices
clone_all_voices(input_base_dir, output_base_dir)
