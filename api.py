from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
import numpy as np
import tensorflow as tf
import joblib
import librosa
import os
from glob import glob
from pydub import AudioSegment
import torch
import torchaudio
from audioseal import AudioSeal
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
# Initialize FastAPI app
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# Load the model
model = tf.keras.models.load_model('speaker_recognition_model_5.h5')
# Load the label encoder and scaler
le = joblib.load('label_encoder_5.pkl')
scaler = joblib.load('scaler_5.pkl')
model_1 = load_model('voice_cloning_detection_model.h5')
label_encoder_1 = joblib.load('label_encoder_3.pkl')
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
def extract_features_1(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
def preprocess_input(file_path):
    features = extract_features_1(file_path)
    if features is not None and features.shape == (40,):
        features = features.reshape(1, 40)
        features = features[..., np.newaxis, np.newaxis]
        return features
    else:
        print(f"Feature extraction failed or feature shape mismatch for {file_path}")
        return None
def predict_1(file_path):
    features = preprocess_input(file_path)
    if features is not None:
        prediction = model_1.predict(features)
        predicted_label = (prediction > 0.5).astype(int).flatten()[0]
        predicted_class = label_encoder_1.inverse_transform([predicted_label])
        return predicted_class[0]
    else:
        return None
# Function to predict speaker with confidence threshold
def predict_speaker(file_path, threshold=0.80):
    features = extract_features(file_path)
    if features is not None:
        features = scaler.transform([features])
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        predictions = model(features)
        prediction_confidence = tf.reduce_max(predictions).numpy()
        print(predictions)
        print(prediction_confidence)
        if prediction_confidence >= threshold:
            prediction = tf.argmax(predictions, axis=1).numpy()
            speaker = le.inverse_transform(prediction)
            return speaker[0]
        else:
            return "Other"
    else:
        return None
def convert_to_wav(audio_path):
    if not audio_path.endswith('.wav'):
        new_path = audio_path.rsplit('.', 1)[0] + '.wav'
        audio = AudioSegment.from_file(audio_path)
        audio.export(new_path, format='wav')
        return new_path
    return audio_path
def load_audio(file_path, target_sample_rate=16000):
    file_path=convert_to_wav(file_path)
    wav,sr = torchaudio.load(file_path)
    wav,sr=torchaudio.load(file_path)
    if sr != target_sample_rate:
        wav = torchaudio.transforms.Resample(sr, target_sample_rate)(wav)
        sr = target_sample_rate
    return wav, sr
# def detect_watermark(audio_path):
#     # Load the audio file
#     wav, sr = load_audio(audio_path)
#     # Ensure the audio tensor has the right shape (batch, channels, samples)
#     if wav.dim() == 1:
#         wav = wav.unsqueeze(0)  # add batch dimension if not present
#     if wav.dim() == 2:
#         wav = wav.unsqueeze(0)  # add channel dimension if not present
#     # Load the detector model
#     detector = AudioSeal.load_detector("audioseal_detector_16bits")
#     # Detect watermark
#     result, message = detector.detect_watermark(wav, sr)
#     # Determine if the audio is likely cloned
#     if result > 0.5:
#         return True, "Voice clone detected. Authorization Failure"
#     else:
#         return False, "The audio is unlikely to be watermarked (cloned)."
def is_silent(audio_data, threshold=0.01):
    # Calculate the root mean square (RMS) energy of the audio data
    rms = librosa.feature.rms(y=audio_data).mean()
    return rms < threshold
# Define FastAPI endpoint for speaker prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...), threshold: Optional[float] = 0.80):
    # Save the uploaded file temporarily
    with open(file.filename, "wb") as buffer:
        buffer.write(file.file.read())
        file_path = file.filename
    audio_dat, _ = librosa.load(file_path, sr=None)
    if is_silent(audio_dat):
        predicted_speaker="No Voice"
        return {"predicted_speaker": predicted_speaker}
    # is_cloned, message = detect_watermark(file_path)
    # if is_cloned:
    #     predicted_speaker = "Cloned"
    #     os.remove(file_path)  # Clean up the saved file
    #     return {"predicted_speaker": predicted_speaker}
    cloned_voice=predict_1(file_path)
    if cloned_voice == "fake" :
        predicted_speaker="Cloned"
        return {"predicted_speaker": predicted_speaker}
    # Perform speaker prediction
    predicted_speaker = predict_speaker(file_path, threshold)
    if predicted_speaker:
        return {"predicted_speaker": predicted_speaker}
    else:
        raise HTTPException(status_code=404, detail=f"Could not predict speaker for {file.filename}")
# Run the FastAPI application with Uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)





