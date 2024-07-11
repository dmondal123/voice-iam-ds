import os
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def convert_to_wav(input_path, output_path):
    try:
        print(f"Converting {input_path} to {output_path}...")
        audio = AudioSegment.from_file(input_path, format="webm")
        audio.export(output_path, format="wav")
        print("Conversion to .wav completed.")
    except Exception as e:
        print(f"Error converting {input_path} to .wav: {str(e)}")
        return None
    return output_path

def clone_voice(audio_path, output_folder, pitch_shift_steps=0.5, time_stretch_rate=1.05):
    try:
        
        
        audio, sr = librosa.load(audio_path, sr=None)
        
        D_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift_steps)
        
        y_stretched = librosa.effects.time_stretch(D_shifted, rate=time_stretch_rate)
        
        S = librosa.feature.melspectrogram(y=y_stretched, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        formants = np.mean(S_db, axis=1)
        formant_filter = np.exp(formants / 20)
        formant_filter = np.tile(formant_filter, (S.shape[1], 1)).T
        S_filtered = S * formant_filter
        y_formant_preserved = librosa.feature.inverse.mel_to_audio(S_filtered, sr=sr)
        
        y_formant_preserved = librosa.util.normalize(y_formant_preserved)
        
        output_filename = f"cloned_{os.path.basename(audio_path)}"
        output_path = os.path.join(output_folder, output_filename)
        
        sf.write(output_path, y_formant_preserved, sr)
        
        return output_path
    except Exception as e:
        print(f"Error cloning voice from {audio_path}: {str(e)}")
        return None
    
audio_path = '/path/to/your/audio/file.webm' 

output_folder = os.path.join(os.path.dirname(audio_path), 'cloned_voices')
os.makedirs(output_folder, exist_ok=True)


if audio_path.lower().endswith('.webm'):
    wav_path = os.path.splitext(audio_path)[0] + '.wav'
    converted_path = convert_to_wav(audio_path, wav_path)
    if converted_path is not None:
        audio_path = converted_path
    else:
        print(f"Failed to convert {audio_path} to .wav format")
        exit(1)

print(f"\nProcessing file: {audio_path}")
cloned_path = clone_voice(audio_path, output_folder)
if cloned_path:
    print(f"Successfully cloned voice to: {cloned_path}")
else:
    print(f"Failed to clone voice from {audio_path}")






# import os
# import glob
# import librosa
# import numpy as np
# import soundfile as sf
# from scipy.signal import resample
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.neighbors import NearestNeighbors

# def clone_voice(wav_path, output_folder, pitch_shift_steps=0.5, time_stretch_rate=1.05):
#     try:
#         print(f"Attempting to clone voice from: {wav_path}")
        
#         # Load the audio file
#         audio, sr = librosa.load(wav_path, sr=None)
#         print(f"Audio loaded. Duration: {librosa.get_duration(y=audio, sr=sr):.2f} seconds")
        
#         # Apply pitch shifting
#         D = librosa.stft(audio)
#         D_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift_steps)
#         print("Pitch shifting applied")
        
#         # Apply time stretching
#         y_stretched = librosa.effects.time_stretch(D_shifted, rate=time_stretch_rate)
#         print("Time stretching applied")
        
#         # Formant preservation (basic implementation)
#         S = librosa.feature.melspectrogram(y=y_stretched, sr=sr)
#         S_db = librosa.power_to_db(S, ref=np.max)
#         formants = np.mean(S_db, axis=1)
#         formant_filter = np.exp(formants / 20)
#         formant_filter = np.tile(formant_filter, (S.shape[1], 1)).T
#         S_filtered = S * formant_filter
#         y_formant_preserved = librosa.feature.inverse.mel_to_audio(S_filtered, sr=sr)
#         print("Formant preservation applied")
        
#         # Normalize audio
#         y_formant_preserved = librosa.util.normalize(y_formant_preserved)
#         print("Audio normalized")
        
#         # Create output filename
#         output_filename = f"cloned_{os.path.basename(wav_path)}"
#         output_path = os.path.join(output_folder, output_filename)
        
#         # Save the cloned voice
#         sf.write(output_path, y_formant_preserved, sr)
#         print(f"Cloned voice saved to: {output_path}")
        
#         return output_path
#     except Exception as e:
#         print(f"Error cloning voice from {wav_path}: {str(e)}")
#         return None

# # Path to the folder containing .wav files
# folder_path = '/Users/msaqib/capstone/SpeakerPredection/voice_based_iam/Aryan'

# # List all .wav files in the folder
# wav_files = glob.glob(os.path.join(folder_path, '*.wav'))
# print(f"Number of .wav files found: {len(wav_files)}")

# if not wav_files:
#     raise ValueError("No .wav files found in the specified folder")

# # Create output directory if it doesn't exist
# output_folder = os.path.join(folder_path, 'cloned_voices')
# os.makedirs(output_folder, exist_ok=True)

# # Clone all voices
# cloned_voices = []
# for wav_file in wav_files:
#     print(f"\nProcessing file: {wav_file}")
#     cloned_path = clone_voice(wav_file, output_folder)
#     if cloned_path:
#         cloned_voices.append(cloned_path)
#     else:
#         print(f"Failed to clone voice from {wav_file}")

# print(f"\nTotal voices cloned: {len(cloned_voices)}")
# if len(cloned_voices) == 0:
#     print("No voices were successfully cloned. Please check the error messages above.")







