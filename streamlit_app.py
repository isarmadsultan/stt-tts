import streamlit as st
import os
import requests
import time
import base64
from pathlib import Path
from backend.record_voice import record_voice

API_URL = "http://127.0.0.1:8000/upload-voice/"

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
        with st.spinner("🔄 Processing your voice..."):
            with open(record_path, "rb") as f:
                files = {"file": (filename, f, "audio/wav")}

                try:
                    response = requests.post(API_URL, files=files, timeout=180)
                    response.raise_for_status()
                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Could not connect to the backend API. Please make sure FastAPI is running on port 8000."
                    )
                    st.stop()
                except requests.exceptions.Timeout:
                    st.error(
                        "⚠️ Request timed out. The backend took too long to respond."
                    )
                    st.stop()
                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ Server error: {e}")
                    if response.text:
                        st.error(f"Details: {response.text}")
                    st.stop()

        # === Parse response ===
        data = response.json()

        # Check for error status
        if data.get("status") == "error":
            st.error(f"❌ Processing failed: {data.get('error', 'Unknown error')}")
            st.error(f"Error type: {data.get('error_type', 'N/A')}")
            with st.expander("📦 Error Details"):
                st.json(data)
            st.stop()

        # === Display Transcription ===
        transcription = data.get("transcription", "")
        if transcription:
            st.subheader("📝 Your Question")
            st.info(transcription)

        # === Display AI Response ===
        st.subheader("🧠 AI Response")
        ai_response = data.get("ai_response", "No response text.")
        st.write(ai_response)

        # === Display Audio Chunk Info ===
        num_chunks = data.get("num_audio_chunks", 0)
        total_duration = data.get("total_audio_duration", 0.0)

        if num_chunks > 0:
            st.info(
                f"🎵 Generated {num_chunks} audio chunks (≈{total_duration:.1f}s total)"
            )

        # === Display Timing Breakdown ===
        timing_info = data.get("timing")
        if timing_info:
            st.subheader("⏱️ Processing Time Breakdown")

            # Create a nice formatted display with 4 columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Transcription", f"{timing_info.get('transcription', 0)}s")
            with col2:
                # Highlight the Agent Response Time (time to first audio)
                agent_response_time = timing_info.get("time_to_first_audio", 0)
                st.metric(
                    "🎯 Agent Response",
                    f"{agent_response_time}s",
                    help="Time from STT completion to first audio playback",
                )
            with col3:
                st.metric("RAG + TTS", f"{timing_info.get('parallel_rag_tts', 0)}s")
            with col4:
                st.metric("Total", f"{timing_info.get('total_time', 0)}s")

            with st.expander(" Detailed Timing"):
                st.json(timing_info)

        # === Play TTS Audio Chunks ===
        tts_audio_paths = data.get("tts_audio_paths", [])

        if tts_audio_paths:
            # Concatenate audio chunks for seamless automatic playback
            combined_audio = b""
            valid_paths = []

            for audio_path in tts_audio_paths:
                if os.path.exists(audio_path):
                    valid_paths.append(audio_path)
                    with open(audio_path, "rb") as f:
                        combined_audio += f.read()

            if combined_audio:
                st.success(
                    f"🔊 Playing AI voice response ({len(valid_paths)} chunks)..."
                )
                st.subheader("🔈 Audio Response")
                # Autoplay the combined audio
                st.audio(combined_audio, format="audio/mp3", autoplay=True)

            # Show individual chunks in expander for debugging
            with st.expander("🎵 Individual Audio Chunks"):
                for idx, audio_path in enumerate(tts_audio_paths, 1):
                    if os.path.exists(audio_path):
                        st.caption(f"Chunk {idx}")
                        st.audio(audio_path, format="audio/mp3")
                    else:
                        st.warning(f"⚠️ Audio chunk {idx} not found: {audio_path}")
        else:
            st.warning("⚠️ No TTS audio chunks were generated.")

        # === Display All Raw Data (for debugging) ===
        with st.expander("📦 Full Response JSON"):
            st.json(data)

    else:
        st.error("❌ Recording file was not created. Please try again.")


# === Sidebar Info ===
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This interface uses:
    - **Whisper** for speech-to-text
    - **RAG** for AI responses
    - **Streaming TTS** for voice output
    
    The streaming TTS generates multiple audio chunks in parallel for faster response times.
    
    **Key Metrics:**
    - **Agent Response Time**: Time from STT completion to first audio playback
    - **Total Time**: Complete end-to-end processing time
    """)

    # Health check
    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=5).json()
        st.success("✅ Backend is healthy")

        with st.expander("🔍 Service Status"):
            for service, status in health.get("services", {}).items():
                icon = "✅" if status else "❌"
                st.text(f"{icon} {service.capitalize()}")
    except:
        st.error("❌ Cannot connect to backend")
