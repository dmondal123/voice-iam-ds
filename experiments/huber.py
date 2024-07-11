import os
import glob
import torchaudio
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification, TrainingArguments, Trainer
import torch
import joblib
from sklearn.metrics import classification_report

# Dataset and output directory
dataset_path = "voice_based_iam"
output_duration = 5  # seconds
output_sr = 16000  # target sample rate

# List speaker folders
speaker_folders = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
print(f"Speaker folders: {speaker_folders}")

# Load and preprocess audio files
def load_and_preprocess_audio(file_name, target_sr, duration):
    waveform, sr = torchaudio.load(file_name)
    waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.size(1) > target_sr * duration:
        waveform = waveform[:, :target_sr * duration]
    elif waveform.size(1) < target_sr * duration:
        waveform = torch.nn.functional.pad(waveform, (0, target_sr * duration - waveform.size(1)))
    return waveform

# Extracting features and labels
features = []
labels = []

for folder in speaker_folders:
    folder_path = os.path.join(dataset_path, folder)
    print(f"Processing folder: {folder}")
    for file in glob.glob(os.path.join(folder_path, "*.wav")):
        waveform = load_and_preprocess_audio(file, target_sr=output_sr, duration=output_duration)
        features.append(waveform.squeeze().numpy())
        labels.append(folder)

print(f"Number of extracted features: {len(features)}")
print(f"Number of labels: {len(labels)}")

# Pad sequences to the same length
max_length = max(len(feature) for feature in features)
features = np.array([np.pad(feature, (0, max_length - len(feature)), 'constant') for feature in features])

# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Save the label encoder
joblib.dump(le, 'label_encoder.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Create Hugging Face datasets
train_data_dict = {'input_values': list(X_train), 'label': y_train}
test_data_dict = {'input_values': list(X_test), 'label': y_test}
train_dataset = Dataset.from_dict(train_data_dict)
test_dataset = Dataset.from_dict(test_data_dict)
dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})

# Feature Extraction
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")

# Model and Training Setup
model = HubertForSequenceClassification.from_pretrained("facebook/hubert-large-ls960-ft", num_labels=len(le.classes_))

# Define Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=200,
)

# Define Trainer
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    accuracy = (preds == p.label_ids).mean()
    return {'accuracy': accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['test'],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)

# Training
trainer.train()

# Save the model
model.save_pretrained("./fine_tuned_hubert")
feature_extractor.save_pretrained("./fine_tuned_hubert")

# Evaluation
train_results = trainer.evaluate(eval_dataset=dataset_dict['train'])
test_results = trainer.evaluate(eval_dataset=dataset_dict['test'])

print(f"Train Accuracy: {train_results['eval_accuracy']}")
print(f"Test Accuracy: {test_results['eval_accuracy']}")

# Generate classification report
y_true = np.concatenate([x['label'] for x in dataset_dict['test']])
y_pred = np.argmax(trainer.predict(dataset_dict['test']).predictions, axis=1)
report = classification_report(y_true, y_pred, target_names=le.classes_)
print(report)