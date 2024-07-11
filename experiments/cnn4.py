import os
import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import Wav2Vec2Processor
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForSequenceClassification, AdamW, get_scheduler
import torch.nn.functional as F
import torch


class AudioDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.file_paths = []
        self.labels = []
        self.label_to_idx = {}

        self._prepare_dataset()

    def _prepare_dataset(self):
        for label_idx, label_name in enumerate(os.listdir(self.root_dir)):
            label_dir = os.path.join(self.root_dir, label_name)
            if os.path.isdir(label_dir):
                self.label_to_idx[label_name] = label_idx
                for file_name in os.listdir(label_dir):
                    if file_name.endswith(".wav"):
                        self.file_paths.append(os.path.join(label_dir, file_name))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0)  # Convert to mono
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform).squeeze().numpy()  # Resample to 16000 Hz

        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs["labels"] = torch.tensor(label, dtype=torch.long)

        return inputs


# Initialize the processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Create the dataset
dataset = AudioDataset(root_dir="data", processor=processor)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: processor.pad(x, return_tensors="pt"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels=len(dataset.label_to_idx),  # Number of classes
    ignore_mismatched_sizes=True
).to(device)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
print(train_dataloader.__len__(), val_dataloader.__len__())
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 10
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
save_path="model_checkpoint.pth"
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
# Training loop
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in train_loader:
        inputs = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    # Save the model checkpoint
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    # Evaluate the model on the validation set
    val_accuracy = evaluate_model(model, val_dataloader)
    print(f'Validation Accuracy: {val_accuracy:.4f}')


print("Training completed.")


model.save_pretrained("pretrained-2.pth")
processor.save_pretrained("pretrained-2.1.pth")

