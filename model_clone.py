import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib
from tensorflow.keras import layers, models


def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data_from_directory(data_dir):
    supported_formats = ('.wav', '.mp3', '.flac')
    features = []
    labels = []
    for label in ['fake', 'real']:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            print(f"Directory {label_dir} not found.")
            continue
        for person in os.listdir(label_dir):
            person_dir = os.path.join(label_dir, person)
            if not os.path.isdir(person_dir):
                continue
            for filename in os.listdir(person_dir):
                if filename.endswith(supported_formats):
                    file_path = os.path.join(person_dir, filename)
                    print(f"Processing file: {file_path}")
                    feature = extract_features(file_path)
                    if feature is not None and feature.shape == (40,): 
                        features.append(feature)
                        labels.append(label)
                    else:
                        print(f"Feature extraction failed for {file_path}")
    return np.array(features), np.array(labels)

data_dir = '/Users/msaqib/capstone/SpeakerPredection/voice_based_iam'

# Load data
X, y = load_data_from_directory(data_dir)

print(f"Loaded {len(X)} samples")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(40, 1, 1)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(1, activation='sigmoid')  # Binary classification output
# ])

model = models.Sequential([
    layers.Conv2D(32, (3, 1), activation='relu', input_shape=(40, 1, 1)),
    layers.MaxPooling2D((2, 1)),
    layers.Conv2D(64, (3, 1), activation='relu'),
    layers.MaxPooling2D((2, 1)),
    layers.Conv2D(128, (3, 1), activation='relu'),
    layers.MaxPooling2D((2, 1)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')

# Save the model
model.save('voice_cloning_detection_model.h5')

 # Evaluate the model on the test data
y_pred = np.argmax(model.predict(X_test), axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(label_encoder, 'label_encoder_3.pkl')
