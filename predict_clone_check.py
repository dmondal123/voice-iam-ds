import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

model_1 = load_model('voice_cloning_detection_model.h5')
label_encoder_1 = joblib.load('label_encoder_3.pkl')

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def preprocess_input(file_path):
    features = extract_features(file_path)
    if features is not None and features.shape == (40,):
        features = features.reshape(1, 40)  
        features = features[..., np.newaxis, np.newaxis] 
        return features
    else:
        print(f"Feature extraction failed or feature shape mismatch for {file_path}")
        return None
    

def predict(file_path):
    features = preprocess_input(file_path)
    if features is not None:
        prediction = model.predict(features)
        predicted_label = (prediction > 0.5).astype(int).flatten()[0]
        predicted_class = label_encoder.inverse_transform([predicted_label])
        return predicted_class[0]
    else:
        return None
    

file_path = '/Users/msaqib/capstone/SpeakerPredection/Test/Saqib/cloned_voices/cloned_audio - 2024-06-20T001813.054.wav'
predicted_class = predict(file_path)
if predicted_class:
    print(f"The predicted class for the audio file is: {predicted_class}")
else:
    print("Prediction failed.")