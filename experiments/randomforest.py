import os
import librosa
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pickle

# Initialize wav2vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

def load_audio(file_path, target_sr=16000):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_features(audio):
    # Extract features using wav2vec2
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the last hidden state as the embedding
    embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()
    return embeddings

import random

def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_level * noise
    return augmented_audio

def shift_audio(audio, shift_max=0.2):
    shift = np.random.randint(len(audio) * shift_max)
    augmented_audio = np.roll(audio, shift)
    return augmented_audio

def augment_audio(audio):
    audio_aug = add_noise(audio)
    audio_aug = shift_audio(audio_aug)
    return audio_aug

def load_dataset(data_dir, augment=False):
    X, y = [], []
    labels = [label for label in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, label))]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.wav', '.mp3', '.flac')):
                audio = load_audio(file_path)
                if audio is not None:
                    features = extract_features(audio)
                    X.append(features)
                    y.append(label_to_idx[label])
                    if augment:
                        # Add augmented data
                        augmented_audio = augment_audio(audio)
                        augmented_features = extract_features(augmented_audio)
                        X.append(augmented_features)
                        y.append(label_to_idx[label])

    return np.array(X), np.array(y)

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset with augmentation
data_dir = 'voice_based_iam'
X, y = load_dataset(data_dir, augment=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print precision, recall, and F1 score
print(classification_report(y_test, y_pred, target_names=[label for label in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, label))]))

model_filename = 'speaker_identification_model_random.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(clf, f)

print("Random Forest model saved successfully!")

labels = [label for label in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, label))]
label_to_idx = {label: idx for idx, label in enumerate(labels)}



