import streamlit as st
import os
import requests
import time
import base64
from backend.record_voice import record_voice

st.title("🎙️ Voice Input Interface")

API_URL = "http://127.0.0.1:8000/upload-voice/"

if st.button("🎤 Speak Now"):
    filename = "streamlit_voice.wav"

    with st.spinner("Recording for 10 seconds..."):
        record_path = record_voice(filename=filename, duration=10, sample_rate=16000)

    time.sleep(1)

    if os.path.exists(record_path):
        st.success(f"✅ Recording saved: {record_path}")
        st.audio(record_path, format="audio/wav")

        # === Send to backend ===
        with open(record_path, "rb") as f:
            files = {"file": (filename, f, "audio/wav")}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            data = response.json()

            st.subheader("🧠 AI Response")
            st.write(data.get("ai_response", "No response text."))

            # Display all response data
            st.json(data)

            # === Play TTS output if available ===
            tts_audio_path = data.get("tts_audio_path")
            if tts_audio_path and os.path.exists(tts_audio_path):
                st.success("🔊 Playing AI voice response...")

                # Load audio file and convert to base64
                with open(tts_audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode()

                # Invisible auto-play audio
                audio_html = f"""
                    <audio autoplay>
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)

                # Also show a visible player (optional)
                st.audio(tts_audio_path, format="audio/mp3")

            else:
                st.warning("⚠️ TTS audio not available or file missing.")

        else:
            st.error(f"❌ Upload failed: {response.text}")

    else:
        st.error("❌ Recording file was not created. Please try again.")
