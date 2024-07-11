import os
import torch
from datasets import Dataset, load_dataset
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Check and set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = torch.device("mps")
else:
    device = torch.device("cpu")

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

data_path = "./voice_based_iam"
file_paths, labels = load_data(data_path)

# Create a dataset
dataset = Dataset.from_dict({"file_path": file_paths, "label": labels})

# Split the dataset
dataset = dataset.train_test_split(test_size=0.2)

# Resample audio files
def resample_audio(audio_path):
    audio, sr = torchaudio.load(audio_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    resampled_audio = resampler(audio)
    return resampled_audio, 16000

import csv

# Read the CSV file and create the label2id mapping dictionary
label2id = {}
with open("label2id.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        label, idx = row
        label2id[label.strip()] = int(idx)

# Define the preprocess function with label2id as an argument
def preprocess(batch, label2id):
    audio, sr = resample_audio(batch["file_path"])
    # Tokenize inputs with padding
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    batch["input_values"] = inputs.input_values.squeeze(1)  # Remove unnecessary dimension
    batch["label"] = label2id[batch["label"]]
    return batch

# Usage:
dataset = dataset.map(lambda x: preprocess(x, label2id), remove_columns=["file_path"])

# Define a label map
label2id = {label: i for i, label in enumerate(sorted(set(labels)))}
id2label = {i: label for label, i in label2id.items()}

# Map labels to IDs
def encode_labels(batch):
    batch["label"] = label2id[batch["label"]]
    return batch

dataset = dataset.map(encode_labels)

# Initialize the model
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
).to(device)

# Define the data collator
def data_collator(features):
    input_values = [feature["input_values"] for feature in features]
    labels = [feature["label"] for feature in features]
    batch = processor.pad({"input_values": input_values}, padding=True, return_tensors="pt")
    batch["labels"] = torch.tensor(labels, dtype=torch.long)
    return batch

# Define the compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=2,  # Reduce the batch size if necessary
    per_device_eval_batch_size=2,   # Reduce the batch size if necessary
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
