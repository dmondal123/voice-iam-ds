import os
import glob
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import tensorflow as tf
from tensorflow import keras

# Dataset and output directory
dataset_path = "voice_based_iam"
output_duration = 5  # seconds
output_sr = 16000  # target sample rate

# List speaker folders
speaker_folders = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
print(f"Speaker folders: {speaker_folders}")

# Function to load and preprocess audio files
def load_and_preprocess_audio(file_name, target_sr, duration):
    print(f"Loading and preprocessing: {file_name}")
    try:
        # Load audio file
        audio, sr = librosa.load(file_name, sr=target_sr, mono=True)

        # Pad or truncate audio to the target duration
        target_length = target_sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
        else:
            audio = audio[:target_length]

        return audio
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

# Function to augment audio data
def augment_audio(audio, sr):
    # Time shift
    shift_max = 0.2 * sr
    shift = int(np.random.uniform(-shift_max, shift_max))
    audio = np.roll(audio, shift)

    # Pitch shift
    pitch_factor = np.random.uniform(-5, 5)
    audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=pitch_factor)

    # Speed change
    speed_factor = np.random.uniform(0.9, 1.1)
    audio = librosa.effects.time_stretch(audio, rate=speed_factor)

    return audio

# Function to convert audio to Mel Spectrogram
def audio_to_mel_spectrogram(audio, sr, n_mels=128):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

# Function to convert Mel Spectrogram to MFCC
def mel_spectrogram_to_mfcc(mel_spectrogram, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(S=mel_spectrogram, sr=sr, n_mfcc=n_mfcc)
    return mfcc

# Function to apply SpecAugment
def apply_spec_augment(mel_spectrogram, num_masks=1, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    mel_spectrogram = mel_spectrogram.copy()
    for _ in range(num_masks):
        # Frequency masking
        num_mel_channels = mel_spectrogram.shape[0]
        num_freqs_to_mask = int(np.random.uniform(0, freq_masking_max_percentage) * num_mel_channels)
        f = np.random.randint(0, num_mel_channels - num_freqs_to_mask)
        mel_spectrogram[f:f + num_freqs_to_mask, :] = 0

        # Time masking
        num_frames = mel_spectrogram.shape[1]
        num_times_to_mask = int(np.random.uniform(0, time_masking_max_percentage) * num_frames)
        t = np.random.randint(0, num_frames - num_times_to_mask)
        mel_spectrogram[:, t:t + num_times_to_mask] = 0

    return mel_spectrogram

# Function to pad or truncate MFCC to a fixed length
def pad_or_truncate_mfcc(mfcc, fixed_length):
    if mfcc.shape[1] < fixed_length:
        pad_width = fixed_length - mfcc.shape[1]
        mfcc_padded = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc_padded = mfcc[:, :fixed_length]
    return mfcc_padded

# Extracting features and labels
features = []
labels = []
fixed_length = 500

for folder in speaker_folders:
    folder_path = os.path.join(dataset_path, folder)
    print(f"Processing folder: {folder}")
    for file in glob.glob(os.path.join(folder_path, "*.wav")):
        audio = load_and_preprocess_audio(file, target_sr=output_sr, duration=output_duration)
        if audio is not None:
            audio_augmented = augment_audio(audio, sr=output_sr)
            mel_spectrogram = audio_to_mel_spectrogram(audio_augmented, sr=output_sr)
            mfcc = mel_spectrogram_to_mfcc(mel_spectrogram, sr=output_sr)
            mfcc_augmented = apply_spec_augment(mfcc)
            mfcc_padded = pad_or_truncate_mfcc(mfcc_augmented, fixed_length)
            features.append(mfcc_padded.flatten())
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
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    # Build the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True), input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(le.classes_), activation='softmax')
    ])



    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = keras.callbacks.ModelCheckpoint('best_model_2.keras', monitor='val_loss', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping, checkpoint])

    # Load the best model
    model = tf.keras.models.load_model('best_model_2.keras')

    # Evaluate the model on the test data
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the label encoder and scaler
    joblib.dump(le, 'label_encoder_9.pkl')
    joblib.dump(scaler, 'scaler_9.pkl')
