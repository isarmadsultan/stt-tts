import streamlit as st
import os
import requests
import time
import base64
from backend.record_voice import record_voice
from backend.config import API_BASE_URL

API_URL = f"http://127.0.0.1:8000/upload-voice/"


st.title("🎙️ Voice Input Interface")



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

            try:
                response = requests.post(API_URL, files=files, timeout=180)
                response.raise_for_status()
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the backend API. Please make sure FastAPI is running on port 8000.")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("⚠️ Request timed out. The backend took too long to respond.")
                st.stop()

        # === Parse response ===
        data = response.json()

        st.subheader("🧠 AI Response")
        st.write(data.get("ai_response", "No response text."))

        # === Display Timing Breakdown ===
        timing_info = data.get("timing_breakdown")
        if timing_info:
            st.subheader("⏱️ Processing Time Breakdown")
            st.json(timing_info)
        elif "processing_time_seconds" in data:
            st.subheader("⏱️ Total Processing Time")
            st.write(f"{data['processing_time_seconds']} seconds")

        # === Display All Raw Data (for debugging) ===
        with st.expander("📦 Full Response JSON"):
            st.json(data)

        # === Play TTS output if available ===
        tts_audio_path = data.get("tts_audio_path")
        if tts_audio_path and os.path.exists(tts_audio_path):
            st.success("🔊 Playing AI voice response...")

            with open(tts_audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode()

            # Auto-play hidden player
            audio_html = f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)

            # Optional visible player
            st.audio(tts_audio_path, format="audio/mp3")

        else:
            st.warning("⚠️ TTS audio not available or file missing.")

    else:
        st.error("❌ Recording file was not created. Please try again.")
