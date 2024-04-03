import sounddevice as sd
import soundfile as sf
import noisereduce as nr
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models

def record_audio(duration, filename):
    fs = 44100  # Sample rate
    print("Recording...")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, myrecording, fs)

def preprocess_audio(filename):
    data, sr = librosa.load(filename)
    # Apply noise reduction
    noisy_part = data
    reduced_noise = nr.reduce_noise(y=noisy_part, sr=sr)
    return reduced_noise, sr

def extract_features(audio_data, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    return mfccs.T

def build_model(input_shape, output_classes):
    model = models.Sequential([
        layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),  # Increased number of filters
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_classes, activation='softmax')
    ])
    return model

def recognize_speech(features):
    # Placeholder for speech recognition
    # You can implement your speech recognition logic here
    recognized_text = "This is a placeholder for recognized speech"
    return recognized_text

def main():
    # Step 1: Record audio
    record_audio(5, "sample_audio.wav")
    
    # Step 2: Preprocess audio
    preprocessed_audio, sr = preprocess_audio("sample_audio.wav")
    
    # Step 3: Extract features
    features = extract_features(preprocessed_audio, sr)
    
    # Step 4: Model setup and training (not included here for brevity)
    input_shape = features.shape
    num_classes = 10  # Replace 10 with the actual number of classes in your dataset
    model = build_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.fit(features, labels, epochs=num_epochs, batch_size=batch_size, validation_split=0.2)
    
    # Step 5: Recognize speech
    recognized_text = recognize_speech(features)
    print(recognized_text)

if __name__ == "__main__":
    main()
