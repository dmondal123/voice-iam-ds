import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import torchaudio

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


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels=5,  # Set the number of labels for your classification task
    ignore_mismatched_sizes=True  # Ignore size mismatches to handle newly initialized weights
)

def collate_fn(batch):
    inputs = [item[0]['input_values'] for item in batch]
    labels = [item[1] for item in batch]
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return {'input_values': inputs}, labels
# Dataset and DataLoader with smaller batch size
dataset = VoiceDataset('voice_based_iam', processor)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
print(train_dataloader.__len__(), val_dataloader.__len__())

# Extract embeddings using wav2vec2
def extract_embeddings(dataloader):
    embeddings = []
    labels = []
    wav2vec2_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, label = batch
            input_values = inputs['input_values']
            input_values = input_values.reshape(input_values.shape[0], -1)  # Reshape input to 2D
            outputs = wav2vec2_model(input_values)
            hidden_states = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(hidden_states)
            labels.append(label)
    return torch.cat(embeddings), torch.cat(labels)

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

input_size = 80000
hidden_size = 128
num_layers = 4
num_classes = len(dataset.user_dirs)
model = CNNModel(num_classes)

def train_model(model, train_dataloader, val_dataloader, num_epochs=7, learning_rate=0.001, accumulation_steps=4, save_path="model_checkpoint.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(train_dataloader):
            input_values = inputs['input_values']
            if inputs["input_values"].shape == (1, 80000):
                #print(inputs["input_values"].shape)
                outputs = model(input_values)
                #print(outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                total_loss += loss.item()
                if i % 10 == 0:
                    torch.cuda.empty_cache()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader):.4f}')

        # Save the model checkpoint
        torch.save(model.state_dict(), save_path)
        print(f'Model saved to {save_path}')

        # Evaluate the model on the validation set
        val_accuracy = evaluate_model(model, val_dataloader)
        print(f'Validation Accuracy: {val_accuracy:.4f}')

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            input_values = inputs['input_values']
            if input_values.shape == (1, 80000):
                outputs = model(input_values)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    model.train()
    return correct / total

# Train the model and save
train_model(model, train_dataloader, val_dataloader)

def load_model(model_path, num_classes):
    model = CNNModel(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Prediction function
def predict_user(audio_file, model, processor, target_sr=16000, max_length=16000*5):
    # Load and preprocess the audio file
    audio, sr = torchaudio.load(audio_file)
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(audio)
    audio = audio.squeeze()[:max_length]  # Trim or pad to max_length
    if audio.shape[0] < max_length:
        padding = max_length - audio.shape[0]
        audio = torch.nn.functional.pad(audio, (0, padding))
    
    # Extract features using Wav2Vec2
    inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True)
    input_values = inputs['input_values'].squeeze(0)
    
    # Ensure the input is in the expected shape
    input_values = input_values.unsqueeze(0)  # Add batch dimension
    input_values = input_values.reshape(input_values.shape[0], -1)  # Reshape to 2D

    # Pass through the trained model to get the prediction
    with torch.no_grad():
        outputs = model(input_values)
        _, predicted = torch.max(outputs.data, 1)
    
    # Map the predicted label back to the user
    user_dirs = [d for d in os.listdir('voice_based_iam') if os.path.isdir(os.path.join('voice_based_iam', d))]
    predicted_user = user_dirs[predicted.item()]
    
    return predicted_user

# Example usage:
model_path = "model_checkpoint.pth"  # Path to the saved model
audio_file = "/Users/dmondal/Documents/iam/testing/testing_anushka_001.wav"  # Path to the audio file for prediction
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
num_classes = len([d for d in os.listdir('voice_based_iam') if os.path.isdir(os.path.join('voice_based_iam', d))])

# Load the model
model = load_model(model_path, num_classes)

# Predict the user
predicted_user = predict_user(audio_file, model, processor)
print(f"Predicted User: {predicted_user}")
