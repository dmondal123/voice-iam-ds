import os
import glob
import numpy as np
import librosa
import joblib
import soundfile as sf
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import tensorflow as tf
from tensorflow import keras

# Dataset and output directory
dataset_path = "voice_based_iam"

# List speaker folders
speaker_folders = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
print(f"Speaker folders: {speaker_folders}")

# Function to extract features
def extract_features(file_name):
    print(f"Extracting features from: {file_name}")
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data), sr=sample_rate)

        features = np.concatenate((np.mean(mfccs.T, axis=0), np.mean(chroma.T, axis=0),
                                   np.mean(mel.T, axis=0), np.mean(contrast.T, axis=0),
                                   np.mean(tonnetz.T, axis=0)))
        if np.isnan(features).any():
            print(f"NaN values found in the features extracted from {file_name}")
            return None
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}. Error: {e}")
        return None
    return features

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

    # Stratified train-test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping, checkpoint])

    # Load the best model
    model = tf.keras.models.load_model('best_model.keras')

    # Evaluate the model on the test data
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the label encoder and scaler
    joblib.dump(le, 'label_encoder_8.pkl')
    joblib.dump(scaler, 'scaler_8.pkl')
