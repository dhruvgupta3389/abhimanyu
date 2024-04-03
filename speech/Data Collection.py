import sounddevice as sd
import soundfile as sf

def record_audio(duration, filename):
    fs = 44100  # Sample rate
    print("Recording...")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, myrecording, fs)

# Example usage
record_audio(5, "sample_audio.wav")
