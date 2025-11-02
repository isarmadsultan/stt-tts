# backend/app.py
import os
import tempfile
import wave
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal
import time
import chromadb
from dotenv import load_dotenv
load_dotenv()
from voice_input import VoiceInputConfig
from ai_agent import generate_answer
from tts_service import generate_tts_audio

app = FastAPI(title="Voice API with Whisper + RAG Agent")
@app.on_event("startup")
def load_resources():
    """
    Load Whisper model and ChromaDB once when the server starts.
    """
    # === Load Whisper ===
    if not getattr(app.state, "whisper_model_loaded", False):
        import whisper
        model_name = os.getenv("WHISPER_MODEL", "medium")
        print(f"🔄 Loading Whisper model '{model_name}'...")
        app.state.whisper_model = whisper.load_model(model_name)
        app.state.whisper_model_loaded = True
        print("✅ Whisper model loaded.")

    # === Initialize ChromaDB ===
    if not getattr(app.state, "chroma_loaded", False):
        try:
            # Use relative path from current file location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, "chroma_db_openai")
            
            print(f"🔄 Initializing Chroma from {db_path}...")
            chroma_client = chromadb.PersistentClient(path=db_path)
            
            # List collections first
            collections = chroma_client.list_collections()
            print("📁 Available Chroma collections:", [c.name for c in collections])
            
            if not collections:
                raise Exception("No collections found in ChromaDB")
            
            # Get the actual collection name (adjust as needed)
            collection_name = os.getenv("CHROMA_COLLECTION_NAME", "langchain")
            app.state.chroma_collection = chroma_client.get_collection(collection_name)
            
            # Verify collection has data
            count = app.state.chroma_collection.count()
            print(f"✅ ChromaDB collection '{collection_name}' loaded with {count} documents.")
            
            app.state.chroma_loaded = True
            
        except Exception as e:
            print(f"❌ Failed to load ChromaDB: {e}")
            raise

def validate_wav_bytes(file_bytes: bytes):
    try:
        with wave.open(BytesIO(file_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            duration = wav_file.getnframes() / float(sample_rate)
    except wave.Error as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {e}")

    config = VoiceInputConfig(sample_rate=sample_rate, format="wav")
    return {
        "sample_rate": sample_rate,
        "channels": num_channels,
        "sample_width_bytes": sample_width,
        "duration_seconds": round(duration, 2),
        "config": config.dict(),
    }


def transcribe_with_whisper_bytes(file_bytes: bytes) -> str:
    if not getattr(app.state, "whisper_model_loaded", False):
        raise HTTPException(status_code=500, detail="Whisper model not loaded")

    model = app.state.whisper_model
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        result = model.transcribe(tmp.name)
        return result.get("text", "").strip()


@app.post("/upload-voice/")
async def upload_voice(file: UploadFile = File(...)):
    if file.content_type not in {"audio/wav", "audio/x-wav"}:
        raise HTTPException(status_code=400, detail="Only WAV files are allowed")

    start_time = time.time()
    file_bytes = await file.read()

    metadata = validate_wav_bytes(file_bytes)
    transcription = transcribe_with_whisper_bytes(file_bytes)

    # Retrieve AI answer using preloaded Chroma
    collection = app.state.chroma_collection
    ai_response = generate_answer(transcription, collection)

   
    try:
        tts_path = generate_tts_audio(ai_response, output_filename="ai_response_tts.mp3")
    # Generate TTS response from AI text
    except Exception as e:
        tts_path = None
        print(f"[WARN] Could not generate TTS: {e}")

    total_time = round(time.time() - start_time, 2)

    return JSONResponse(
        content={
            "filename": file.filename,
            "content_type": file.content_type,
            **metadata,
            "transcription": transcription,
            "ai_response": ai_response,
            "tts_audio_path": tts_path,
            "processing_time_seconds": total_time,
        }
    )

