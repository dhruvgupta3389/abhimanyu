import librosa
import noisereduce as nr

def preprocess_audio(filename):
    data, sr = librosa.load(filename)
    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=sr)
    return reduced_noise, sr

def extract_features(audio_data, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    return mfccs.T

# Example usage
filename = "sample_audio.wav"
preprocessed_audio, sr = preprocess_audio(filename)
features = extract_features(preprocessed_audio, sr)
