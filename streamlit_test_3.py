import streamlit as st
import sounddevice as sd
import wave
import tempfile
import keyboard
import numpy as np
from groq import Groq

# --- Audio Capture Function (using sounddevice) ---
def record_audio_interactive(samplerate=16000):
    """
    Records audio interactively, starting and stopping based on user actions.
    """
    st.write("Press 'r' to start recording and 's' to stop.")  # Streamlit output

    try:
        audio_data = []
        recording = False

        while True:
            if keyboard.is_pressed('r') and not recording:
                st.write("Recording started. Press 's' to stop.")  # Streamlit output
                recording = True
                audio_data = []
                stream = sd.InputStream(samplerate=samplerate, channels=1, dtype="int16")
                stream.start()

            if recording:
                data, _ = stream.read(1024)
                audio_data.append(data)

            if keyboard.is_pressed('s') and recording:
                st.write("Recording stopped.")  # Streamlit output
                recording = False
                stream.stop()
                stream.close()
                break

        audio_data = np.concatenate(audio_data)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_file.name, 'w') as wav_file:
            wav_file.setparams((1, 2, samplerate, 0, 'NONE', 'not compressed'))
            wav_file.writeframes(audio_data.tobytes())

        st.write(f"Audio saved: {temp_file.name}")  # Streamlit output
        return temp_file.name

    except Exception as e:
        st.write(f"Error: {e}")  # Streamlit output
        return None

# --- Groq API Transcription Function ---
client = Groq(api_key="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")  # Replace with your actual API key

def transcribe_audio(audio_file_path):
    """Transcribes speech from an audio file using the Groq API."""

    try:
        with open(audio_file_path, "rb") as file:
            translation = client.audio.translations.create(
                file=(audio_file_path, file.read()),
                model="whisper-large-v3",
                response_format="json",
            )
        return translation.text

    except FileNotFoundError:
        st.write(f"Error: File not found at {audio_file_path}")  # Streamlit output
        return None
    except Exception as e:
        st.write(f"An error occurred: {e}")  # Streamlit output
        return None

# --- Streamlit UI ---
if __name__ == "__main__":  # Add this conditional block
    st.write("Speech-to-Text App")

    if st.button("Start Recording and Transcribe"):
        with st.spinner("Recording and Transcribing..."):
            file_path = record_audio_interactive()
            if file_path:
                transcript = transcribe_audio(file_path)
                if transcript:
                    st.write("Transcript:")
                    st.write(transcript)
                else:
                    st.write("Transcription failed.")
            else:
                st.write("Recording failed.")