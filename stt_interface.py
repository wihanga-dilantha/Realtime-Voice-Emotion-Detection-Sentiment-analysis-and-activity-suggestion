import pyaudio
import whisper
import numpy as np
import warnings
import streamlit as st

st.set_page_config(layout='wide')
st.header("Speech to Text", divider=True)

def write_text(txt):
    st.write(txt) 

    #start listning
def start():
    write_text("Listning..")
    #stop listning
def stop():
    write_text("Stopped")

# Suppress specific warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load the base Whisper model
whisper_model = whisper.load_model("base")

# Audio recording parameters
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate
CHUNK = 1024  # Buffer size

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# print("Recording...")

# Buffer to hold audio data
buffer = []
#bottom 3 columns
left, middle, right = st.columns([10,1,1], vertical_alignment="bottom")

start_button = middle.button("Start", on_click=start)
stop_button = right.button("Stop", on_click=stop)

txtbox = st.empty()

if start_button:
    try:
        with open("transcriptions.txt", "a", encoding='utf-8') as file:  # Open file in append mode
            while not stop_button:
                # Read audio data from the microphone
                data = stream.read(CHUNK)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Append the chunk to the buffer
                buffer.append(audio_data)
                
                # Accumulate approximately 5 seconds of audio before transcribing
                if len(buffer) >= int(RATE / CHUNK * 5):
                    # Concatenate buffer into a single array
                    audio_buffer = np.concatenate(buffer, axis=0)
                    
                    # Convert audio data to Whisper format
                    audio_buffer = audio_buffer.astype(np.float32) / 32768.0
                    
                    # Transcribe audio data
                    result = whisper_model.transcribe(audio_buffer)
                    transcription = result['text']
                    txtbox.markdown(f"**Transcription:** {transcription}")
                    
                    # Save transcription to file
                    file.write(transcription + "\n")
                    file.flush()  # Flush the buffer to ensure immediate write
                    
                    # Clear the buffer
                    buffer = []

    except KeyboardInterrupt:
        print("Recording stopped")
        stream.stop_stream()
        stream.close()
        audio.terminate()

