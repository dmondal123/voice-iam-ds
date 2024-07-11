import os
import argparse
import pickle
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch

def load_audio(file_path, target_sr=16000):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_features(audio):
    # Initialize wav2vec2 model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    # Extract features using wav2vec2
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the last hidden state as the embedding
    embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()
    return embeddings

def predict_speaker(file_path, model_filename='speaker_identification_model_random.pkl', label_mapping=None):
    # Load the saved model
    with open(model_filename, 'rb') as f:
        clf = pickle.load(f)

    # Load the audio file
    audio = load_audio(file_path)
    if audio is not None:
        # Extract features (placeholder for now)
        features = extract_features(audio)
        features = np.reshape(features, (1, -1))
        # Predict speaker
        prediction = clf.predict(features)
        # Map integer label to user name
        if label_mapping is not None:
            predicted_label = label_mapping[prediction[0]]
        else:
            predicted_label = prediction[0]
        return predicted_label
    else:
        return "Error: Unable to load audio"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Identification")
    parser.add_argument("audio_path", type=str, help="Path to the audio file for speaker identification")
    args = parser.parse_args()

    # Define the label mapping dictionary
    label_mapping = {0: "Anushka", 1: "Diya", 2: "Harsh", 3: "Saqib", 4: "Sneha"}

    predicted_speaker = predict_speaker(args.audio_path, label_mapping=label_mapping)
    print(f"Predicted Speaker: {predicted_speaker}")
