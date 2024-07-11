import numpy as np
import tensorflow as tf
import joblib
import librosa
import os
from glob import glob

model = tf.keras.models.load_model('speaker_recognition_model_3.h5')
le = joblib.load('label_encoder_3.pkl')
scaler = joblib.load('scaler_3.pkl')


# Function to extract features
def extract_features(file_name):
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}. Error: {e}")
        return None
    return mfccs_scaled


def predict_speaker(file_path, threshold=0.95):
    features = extract_features(file_path)
    if features is not None:
        features = scaler.transform([features])
        predictions = model.predict(features)
        prediction_confidence = np.max(predictions)
        print(predictions)
        print(prediction_confidence)
        if prediction_confidence >= threshold:
            prediction = np.argmax(predictions, axis=1)
            speaker = le.inverse_transform(prediction)
            return speaker[0]
        else:
            return "other"
    else:
        return None
    
def convert_to_wav(audio_path):
    if not audio_path.endswith('.wav'):
        new_path = audio_path.rsplit('.', 1)[0] + '.wav'
        try:
            audio = AudioSegment.from_file(audio_path)
            audio.export(new_path, format='wav')
            print(f"Converted: {audio_path} to {new_path}")
            return new_path
        except CouldntDecodeError as e:
            print(f"Error decoding {audio_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {audio_path}: {e}")
            return None
    return audio_path


# Example usage with threshold
file_path = '/Users/msaqib/Downloads/audio (3).webm'
file_path = convert_to_wav(file_path)
predicted_speaker = predict_speaker(file_path)
if predicted_speaker:
    print(f"Predicted Speaker: {predicted_speaker}")
else:
    print("Could not extract features from the audio file.")
