from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import numpy as np
import tensorflow as tf
import joblib
import librosa
import os
from pydub import AudioSegment

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

# Load the model, label encoder, and scaler
model = tf.keras.models.load_model('best_model_1.keras')
label_encoder = joblib.load('label_encoder_5.pkl')
scaler = joblib.load('scaler_5.pkl')

# Function to extract MFCC features
def extract_features(file_name):
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}. Error: {e}")
        return None
    return mfccs_scaled

# Function to preprocess input and extract features
def preprocess_input(file_path):
    features = extract_features(file_path)
    if features is not None:
        features = scaler.transform([features])
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        return features
    else:
        return None

# Function to predict the speaker
def predict_speaker(file_path, threshold=0.80):
    features = preprocess_input(file_path)
    if features is not None:
        predictions = model(features)
        prediction_confidence = tf.reduce_max(predictions).numpy()
        print(predictions)
        print(prediction_confidence)
        if prediction_confidence >= threshold:
            prediction = tf.argmax(predictions, axis=1).numpy()
            speaker = label_encoder.inverse_transform(prediction)
            return speaker[0]
        else:
            return "Other"
    else:
        return None

# Convert non-WAV files to WAV format
def convert_to_wav(audio_path):
    if not audio_path.endswith('.wav'):
        new_path = audio_path.rsplit('.', 1)[0] + '.wav'
        audio = AudioSegment.from_file(audio_path)
        audio.export(new_path, format='wav')
        return new_path
    return audio_path

# Check if audio is silent
def is_silent(audio_data, threshold=0.01):
    rms = librosa.feature.rms(y=audio_data).mean()
    return rms < threshold

# Define FastAPI endpoint for speaker prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...), threshold: Optional[float] = 0.80):
    # Save the uploaded file temporarily
    file_path = f"temp/{file.filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Convert to WAV if necessary
    file_path = convert_to_wav(file_path)
    
    # Load audio and check if silent
    audio_data, _ = librosa.load(file_path, sr=None)
    if is_silent(audio_data):
        predicted_speaker = "No Voice"
        os.remove(file_path)  # Clean up the saved file
        return {"predicted_speaker": predicted_speaker}
    
    # Perform speaker prediction
    predicted_speaker = predict_speaker(file_path, threshold)
    os.remove(file_path)  # Clean up the saved file
    
    if predicted_speaker:
        return {"predicted_speaker": predicted_speaker}
    else:
        raise HTTPException(status_code=404, detail=f"Could not predict speaker for {file.filename}")

# Run the FastAPI application with Uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
