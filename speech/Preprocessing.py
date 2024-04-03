import noisereduce as nr
import librosa

def preprocess_audio(filename):
    data, sr = librosa.load(filename)
    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=sr)
    return reduced_noise

# Example usage
preprocessed_audio = preprocess_audio("sample_audio.wav")
