import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import joblib
import librosa.display
import soundfile as sf
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import tensorflow as tf
from tensorflow import keras

# Dataset and output directory
dataset_path = "voice_based_iam"

# List speaker folders
speaker_folders = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
print(f"Speaker folders: {speaker_folders}")

# Function to merge audio files
def merge_audio_files(input_folder, output_file):
    full_path = os.path.join(dataset_path, input_folder)
    files = os.listdir(full_path)
    merged = AudioSegment.empty()
    for file in files:
        if file.endswith('.wav'):
            audio = AudioSegment.from_wav(os.path.join(full_path, file))
            merged += audio
    merged.export(output_file, format='wav')

# Function to extract features
def extract_features(file_name):
    print(f"Extracting features from: {file_name}")
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        if np.isnan(mfccs_scaled).any():
            print(f"NaN values found in the features extracted from {file_name}")
            return None
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}. Error: {e}")
        return None
    return mfccs_scaled

# Extracting features and labels
features = []
labels = []

for folder in speaker_folders:
    folder_path = os.path.join(dataset_path, folder)
    print(f"Processing folder: {folder}")
    for file in glob.glob(os.path.join(folder_path, "*.wav")):
        data = extract_features(file)
        if data is not None:
            features.append(data)
            labels.append(folder)

print(f"Number of extracted features: {len(features)}")
print(f"Number of labels: {len(labels)}")

# Convert features and labels to numpy arrays
X = np.array(features)
y = np.array(labels)

# Check if features and labels are correctly extracted
if X.shape[0] == 0:
    print("No features were extracted. Please check the audio files and the feature extraction process.")
else:
    # Encode the labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape the data for CNN input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build the CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Save the model
    model.save('speaker_recognition_cnn_model.h5')

    # Evaluate the model on the test data
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Calculate and print train accuracy
    y_train_pred = np.argmax(model.predict(X_train), axis=1)
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))

    # Save the label encoder and scaler
    joblib.dump(le, 'label_encoder_6.pkl')
    joblib.dump(scaler, 'scaler_6.pkl')
