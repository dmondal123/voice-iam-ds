import os
import pandas as pd
import soundfile as sf
import numpy as np
from datasets import Dataset, load_metric
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# Step 1: Prepare Data
data = []
data_dir = 'voice_based_iam'

# Collect data into a pandas DataFrame
for user_id in os.listdir('voice_based_iam'):
    if user_id == '.DS_Store':
        continue
    for file_name in os.listdir(f'voice_based_iam/{user_id}'):
        if file_name == '.DS_Store':
            continue
        file_path = f'voice_based_iam/{user_id}/{file_name}'
        data.append({'path': file_path, 'label': user_id})

df = pd.DataFrame(data)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split dataset into train and test
dataset = dataset.train_test_split(test_size=0.2)

# Step 2: Create Label Mapping
label_to_id = {label: idx for idx, label in enumerate(dataset['train'].unique('label'))}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# Step 3: Preprocess Data
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def preprocess_function(batch):
    audio, _ = sf.read(batch['path'])
    input_values = processor(audio, sampling_rate=16000).input_values[0]
    batch['input_values'] = input_values
    batch['labels'] = label_to_id[batch['label']]
    
    # Debug prints
    #print(f"Processed {batch['path']}:")
    #print(f"  input_values type: {type(input_values)}")
    #print(f"  input_values dtype: {input_values.dtype if isinstance(input_values, np.ndarray) else 'N/A'}")
    #print(f"  input_values shape: {input_values.shape if isinstance(input_values, np.ndarray) else 'N/A'}")
    #print(f"  labels: {batch['labels']}")
    
    return batch


# Apply preprocessing
dataset = dataset.map(preprocess_function)

# Step 4: Fine-tune Model
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=len(label_to_id))

from transformers import DataCollatorWithPadding
import torch

class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        # Ensure the input_values are numpy arrays and pad them correctly
        input_features = [feature["input_values"] for feature in features]
        
        # Padding input features manually
        max_length = max(len(f) for f in input_features)
        padded_features = [
            np.pad(f, (0, max_length - len(f)), mode='constant') for f in input_features
        ]
        
        batch = {
            "input_values": torch.tensor(padded_features, dtype=torch.float32),
            "labels": torch.tensor([feature["labels"] for feature in features], dtype=torch.long),
        }
        
        return batch

data_collator = CustomDataCollatorWithPadding(tokenizer=processor)

# Define training arguments
training_args = TrainingArguments(
  output_dir="./results",
  evaluation_strategy="epoch",
  learning_rate=2e-5,
  per_device_train_batch_size=4,
  per_device_eval_batch_size=4,
  num_train_epochs=3,
  weight_decay=0.01,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=processor,
    data_collator=data_collator,
)

# Fine-tune model
trainer.train()

# Step 5: Evaluate the Model
results = trainer.evaluate()
print(f"Evaluation results: {results}")