import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn
import torch.optim as optim

# Dataset definition with sequence length limit
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

# Load processor and model for wav2vec2
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Custom collate function for padding
def collate_fn(batch):
    inputs = [item[0]['input_values'] for item in batch]
    labels = [item[1] for item in batch]
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return {'input_values': inputs}, labels

# Dataset and DataLoader with smaller batch size
dataset = VoiceDataset('voice_based_iam', processor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)  # Smaller batch size

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

embeddings, labels = extract_embeddings(dataloader)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


input_size = embeddings.size(1)
hidden_size = 128
num_layers = 2
num_classes = len(dataset.user_dirs)
model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

# Train LSTM model with gradient accumulation
def train_model(model, dataloader, num_epochs=10, learning_rate=0.001, accumulation_steps=4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs['input_values'])
            loss = criterion(outputs, labels)
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
            if i % 10 == 0:
                torch.cuda.empty_cache()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

train_model(model, dataloader)


