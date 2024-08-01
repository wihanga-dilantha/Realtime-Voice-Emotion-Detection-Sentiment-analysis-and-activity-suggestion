import pyaudio
import numpy as np
import pandas as pd
import webrtcvad
import librosa
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import load_model
import streamlit as st

st.set_page_config(layout='wide')
st.header("Voice Emotion Detection", divider=True)

def emotion_detection():
    # Enable eager execution
    #tf.disable_v2_behavior()
    tf.compat.v1.disable_v2_behavior()
    #tf.enable_eager_execution()
    tf.compat.v1.enable_eager_execution()

    # Function to check if audio data contains voice
    def contains_voice(audio_data, sr=16000, frame_duration=30, silence_threshold=0.01):
        vad = webrtcvad.Vad()
        vad.set_mode(2)  # Set aggressiveness mode (0-3, stricter with higher values)

        # Convert audio data to 16-bit PCM (if needed)
        if not isinstance(audio_data, np.ndarray):
            audio_data = (audio_data * 32768).astype(np.int16)

        # Frame duration in milliseconds (10, 20, or 30 ms)
        num_samples_per_frame = int(sr * (frame_duration / 1000.0))
        frames = [audio_data[i:i + num_samples_per_frame * 2] for i in range(0, len(audio_data), num_samples_per_frame * 2)]

        # Check each frame for voice activity
        for frame in frames:
            if len(frame) < num_samples_per_frame * 2:
                continue

            # Calculate maximum absolute value in the frame
            max_abs_value = np.abs(frame).max()

            # Skip frame if silence threshold is exceeded
            if max_abs_value < silence_threshold:
                continue

            if vad.is_speech(frame, sr):
                return True
        return False

    # Function to prepare input data from live audio
    def prepare_input_data_live(audio_data, sr=16000, chunk_duration=2, n_mfcc=40):
        # Reshape audio data into 2-second chunks
        audio_chunks = [audio_data[i:i+sr*chunk_duration] for i in range(0, len(audio_data), sr*chunk_duration)]

        # Extract MFCC features for each chunk
        mfcc_chunks = []
        for chunk in audio_chunks:
            y = chunk.astype(np.float32)
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
            mfcc_chunks.append(mfcc)

        # Expand dimensions for model input
        input_data = np.expand_dims(mfcc_chunks, axis=-1)
        return input_data

    # Function to make prediction using the provided model and audio data
    def make_prediction_live(model, audio_data):
        input_data = prepare_input_data_live(audio_data)
        prediction = model.predict(input_data)
        predicted_classes = np.argmax(prediction, axis=1)
        return predicted_classes

    # Load the trained model
    model_path = 'emotionDetection.h5'
    model = load_model(model_path)

    # Initialize PyAudio
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # We need 16000 Hz for webrtcvad
    CHUNK = 1024

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening for voice and predicting emotions...")

    while True:
        # Read audio data from the stream
        data = stream.read(CHUNK)

        # Convert data to NumPy array (if needed)
        if not isinstance(data, np.ndarray):
            data_np = np.frombuffer(data, dtype=np.int16)
        else:
            data_np = data

        # Check for voice activity
        if contains_voice(data_np):
            # Make emotion prediction
            prediction = make_prediction_live(model, data_np)

            # Mapping numerical labels to emotion names
            emotion_mapping = {
                0: 'angry',
                1: 'neutral',
                2: 'calm',
                3: 'happy',
                4: 'neutral',
                5: 'surprise',
                6: 'sad',
                7: 'disgust'
            }

            

            # Get the predicted emotion labels
            predicted_emotions = [emotion_mapping[label] for label in prediction]

            # Print the predicted emotions
            live_emotion.markdown(f"## {predicted_emotions[0]}")
            
            # Initialize the progress bars for each emotion

            detected_emotion = (predicted_emotions[0])
            intensity = 100
            update_emotion_progress(detected_emotion, intensity)

    # Close the stream when finished
    stream.stop_stream()
    stream.close()
    p.terminate()
# 3 columns
left, middle, right = st.columns([10, 1, 1], vertical_alignment="bottom")
start_button = middle.button("Start")

live_emotion = st.empty()

emotions = ['neutral', 'calm', 'happy', 'sad', 'angry','fear', 'disgust', 'surprise']
emotion_bars = {emotion: st.progress(0, text=emotion.capitalize()) for emotion in emotions}

def update_emotion_progress(emotion, intensity):
    for key in emotion_bars.keys():
        if key == emotion:
            emotion_bars[key].progress(intensity, text=f"{key.capitalize()} - {intensity}%")
        else:
            emotion_bars[key].progress(0, text=f"{key.capitalize()} - 0%")


if start_button:
    emotion_detection()