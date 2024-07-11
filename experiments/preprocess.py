import os
from datasets import Dataset, DatasetDict, load_dataset
import torchaudio
from transformers import Wav2Vec2Processor

data_path = "./voice_based_iam"

# Load processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Function to load audio files and labels
def load_data(data_path):
    data = []
    labels = []
    for speaker in os.listdir(data_path):
        speaker_path = os.path.join(data_path, speaker)
        if os.path.isdir(speaker_path):
            for file_name in os.listdir(speaker_path):
                if file_name.endswith(".wav"):
                    file_path = os.path.join(speaker_path, file_name)
                    data.append(file_path)
                    labels.append(speaker)
    return data, labels

file_paths, labels = load_data(data_path)

# Create a dataset
dataset = Dataset.from_dict({"file_path": file_paths, "label": labels})

# Split the dataset
dataset = dataset.train_test_split(test_size=0.2)

# Preprocess the dataset
from torchaudio.transforms import Resample

# Specify the target sampling rate
target_sampling_rate = 16000

# Resample audio files
def resample_audio(audio_path):
    audio, sr = torchaudio.load(audio_path)
    resampler = Resample(orig_freq=sr, new_freq=target_sampling_rate)
    resampled_audio = resampler(audio)
    return resampled_audio, target_sampling_rate

# Update the preprocess function
def preprocess(batch):
    audio_path = batch["file_path"]
    audio, sr = resample_audio(audio_path)
    # Make sure audio is a single-channel waveform
    if len(audio.shape) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    batch["input_values"] = inputs.input_values.squeeze(0)  # Squeeze to remove batch dimension
    return batch


dataset = dataset.map(preprocess, remove_columns=["file_path"])

# Define a label map
label_list = dataset["train"].unique("label")
label_list.sort()
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Map labels to IDs
def encode_labels(batch):
    batch["label"] = label2id[batch["label"]]
    return batch

dataset = dataset.map(encode_labels)

import csv

# Save label2id to CSV
with open("label2id.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for label, idx in label2id.items():
        writer.writerow([label, idx])

# Save id2label to CSV
with open("id2label.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for idx, label in id2label.items():
        writer.writerow([idx, label])
