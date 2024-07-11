from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware 
import torch
import torchaudio
from transformers import Wav2Vec2Processor
import torch.nn as nn
import torch.nn.functional as F
import os
from audioseal import AudioSeal
from torch.utils.data import Dataset, DataLoader
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

class VoiceDataset(Dataset):
    def __init__(self, directory, processor, target_sr=16000, max_length=16000*5):  # 5 seconds max length
        self.directory = directory
        self.processor = processor
        self.target_sr = target_sr
        self.max_length = max_length
        self.user_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        self.audio_files = [(user, os.path.join(directory, user, f)) for user in self.user_dirs for f in os.listdir(os.path.join(directory, user)) if f.endswith('.wav')]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        user, file_path = self.audio_files[idx]
        audio, sr = torchaudio.load(file_path)
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(audio)
        audio = audio.squeeze()[:self.max_length]  # Trim or pad to max_length
        if audio.shape[0] < self.max_length:
            padding = self.max_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))
        inputs = self.processor(audio, sampling_rate=self.target_sr, return_tensors="pt", padding=True)
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}  # Remove batch dimension
        label = self.user_dirs.index(user)
        return inputs, label

    def __getclasses__(self):
        return self.user_dirs

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 20000, 128)  # Adjusted for input size 80000
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 20000)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_model(model_path, num_classes):
    model = CNNModel(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

from pydub import AudioSegment

def convert_to_wav(audio_path):
    if not audio_path.endswith('.wav'):
        new_path = audio_path.rsplit('.', 1)[0] + '.wav'
        audio = AudioSegment.from_file(audio_path)
        audio.export(new_path, format='wav')
        return new_path
    return audio_path


def predict_user(audio_file, model, processor, target_sr=16000, max_length=16000 * 5):
    audio_file = convert_to_wav(audio_file)
    audio, sr = torchaudio.load(audio_file)
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(audio)
    audio = audio.squeeze()[:max_length]
    if audio.shape[0] < max_length:
        padding = max_length - audio.shape[0]
        audio = torch.nn.functional.pad(audio, (0, padding))

    inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True)
    input_values = inputs['input_values'].squeeze(0)
    input_values = input_values.unsqueeze(0)
    input_values = input_values.reshape(input_values.shape[0], -1)

    with torch.no_grad():
        outputs = model(input_values)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
    user_dirs = [d for d in os.listdir('voice_based_iam') if os.path.isdir(os.path.join('voice_based_iam', d))]
    predicted_user = user_dirs[predicted.item()]
    
    user_dirs = [d for d in os.listdir('voice_based_iam') if os.path.isdir(os.path.join('voice_based_iam', d))]
    predicted_user = user_dirs[predicted.item()]

    if predicted_user == 'Other' or probabilities[0][predicted].item() < 0.5:  # Example threshold
        return "User not in database"
    return predicted_user


def load_audio(file_path, target_sample_rate=16000):
    file_path=convert_to_wav(file_path)
    print("working1")
    wav,sr = torchaudio.load(file_path)
    print("working2")
    

    wav,sr=torchaudio.load(file_path)

    if sr != target_sample_rate:
        wav = torchaudio.transforms.Resample(sr, target_sample_rate)(wav)
        sr = target_sample_rate
    return wav, sr

# Function to detect watermark
def detect_watermark(audio_path):
    # Load the audio file 
    wav, sr = load_audio(audio_path)
    
    # Ensure the audio tensor has the right shape (batch, channels, samples)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # add batch dimension if not present
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)  # add channel dimension if not present

    # Load the detector model
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    
    # Detect watermark
    result, message = detector.detect_watermark(wav, sr)

    # Determine if the audio is likely cloned
    if result > 0.5:
        return True, "Voice clone detected. Authorization Failure"
    else:
        return False, "The audio is unlikely to be watermarked (cloned)."
'''
# Prediction function
def predict_user(audio_file, model, processor, target_sr=16000, max_length=16000*5):
    audio, sr = torchaudio.load(audio_file)
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(audio)
    audio = audio.squeeze()[:max_length]
    if audio.shape[0] < max_length:
        padding = max_length - audio.shape[0]
        audio = torch.nn.functional.pad(audio, (0, padding))

    inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True)
    input_values = inputs['input_values'].squeeze(0)
    input_values = input_values.unsqueeze(0)
    input_values = input_values.reshape(input_values.shape[0], -1)

    with torch.no_grad():
        outputs = model(input_values)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    user_dirs = [d for d in os.listdir('voice_based_iam') if os.path.isdir(os.path.join('voice_based_iam', d))]
    user_dirs.append('Unknown')
    predicted_user = user_dirs[predicted.item()]

    return predicted_user, confidence.item()
'''
model_path = "model_checkpoint_updated.pth"  # Path to the saved model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
num_classes = len([d for d in os.listdir('voice_based_iam') if os.path.isdir(os.path.join('voice_based_iam', d))])  # Adding 1 for "Unknown"

# Load the model
model = load_model(model_path, num_classes)

@app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         audio_path = f"/tmp/{file.filename}"
#         print(audio_path)
#         print("working")
#         with open(audio_path, "wb") as buffer:
#              buffer.write(await file.read())
        
#         is_cloned, message = detect_watermark(audio_path)
        
#         if is_cloned:
#             return JSONResponse(content={"Cloned Voice detected. Authorization Failure": message})
        
#         predicted_user, confidence = predict_user(audio_path, model, processor)
#         os.remove(audio_path)  # Clean up the saved file

#         if predicted_user == 'Other' or confidence < 0.5:  # Adjust the confidence threshold as needed
#             return JSONResponse(content={"message": "User not in database"})

#         return JSONResponse(content={"predicted_user": predicted_user})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

async def predict(file: UploadFile = File(...)):
    try:
        audio_path = f"/tmp/{file.filename}"
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())

        is_cloned, message = detect_watermark(audio_path)
        if is_cloned:
            os.remove(audio_path)  # Clean up the saved file
            return JSONResponse(content={"Cloned Voice detected . Authorization Failure": message})

        predicted_user = predict_user(audio_path, model, processor)
        os.remove(audio_path)  # Clean up the saved file
        return JSONResponse(content={"predicted_user": predicted_user})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0')
