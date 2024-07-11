import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import torch.nn.functional as F

def load_audio(file_path, target_sr=16000):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
# Define a dataset class
class VoiceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.users = [user for user in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, user))]

    def __len__(self):
        return sum(len(os.listdir(os.path.join(self.data_dir, user))) for user in self.users)

    def __getitem__(self, idx):
        user_idx = 0
        while idx >= len(os.listdir(os.path.join(self.data_dir, self.users[user_idx]))):
            idx -= len(os.listdir(os.path.join(self.data_dir, self.users[user_idx])))
            user_idx += 1
        user_dir = os.path.join(self.data_dir, self.users[user_idx])
        audio_files = [file for file in os.listdir(user_dir) if file.endswith('.wav') or file.endswith('.mp3')]

        file_name = audio_files[idx]
        file_path = os.path.join(user_dir, file_name)
        
        try:
            audio, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

        return audio, user_idx


# Define the RNN model
class VoiceRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(VoiceRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state
        h_0 = torch.zeros(1, x.size(0), hidden_size)

        # Forward pass through RNN
        out, _ = self.rnn(x, h_0)

        # Only take the output from the final time step
        out = self.fc(out[:, -1, :])
        return out


# Define hyperparameters
input_size = 1  # Assuming single-channel audio
hidden_size = 128
num_classes = len(os.listdir('voice_based_iam'))

# Define hyperparameters
num_epochs = 10
log_interval = 10
input_size = 1  # Assuming single-channel audio
hidden_size = 128
num_classes = len(os.listdir('voice_based_iam'))

def collate_fn(batch):
    # Sort the batch by sequence length (descending order)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    # Get the maximum sequence length in the batch
    max_seq_len = batch[0][0].shape[0]
    
    # Pad sequences to have equal lengths
    padded_data = [F.pad(torch.tensor(data), pad=(0, max_seq_len - data.shape[0]), mode='constant', value=0) for data, label in batch]
    
    # Convert list of padded sequences to a tensor
    padded_data = torch.stack(padded_data)
    
    # Convert labels to tensor
    labels = torch.tensor([label for data, label in batch])
    
    return padded_data, labels



# Create dataset and dataloader
dataset = VoiceDataset('voice_based_iam')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Initialize the model
model = VoiceRNN(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.unsqueeze(1)  # Add channel dimension
        data, target = data.float(), target.long()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

