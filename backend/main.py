import os
import tempfile
import wave
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import time
import torch
import chromadb
from config import API_BASE_URL
from dotenv import load_dotenv
from voice_input import VoiceInputConfig
from ai_agent import generate_answer
from tts_service import generate_tts_audio
import uvicorn
from config import API_HOST, API_PORT, API_PROTOCOL
load_dotenv()

app = FastAPI(title="Voice API with Whisper + RAG Agent")
history = []

@app.on_event("startup")
def load_resources():
    """
    Load Whisper model (including large models) and ChromaDB once when the server starts.
    """
    # === Load Whisper ===
    if not getattr(app.state, "whisper_model_loaded", False):
        import whisper

        model_name = "medium"  # Fixed to medium model
        gpu_index = int(os.getenv("CUDA_DEVICE", "0"))
        device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"

        print(f"\n🔄 Loading Whisper model '{model_name}' on device={device} ...")

        if device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            # Load model on device
            model = whisper.load_model(model_name, device=device)
            app.state.whisper_model = model
            app.state.use_fp16 = device.startswith("cuda") and not model_name.startswith("tiny")

            app.state.whisper_model_loaded = True

            gpu_name = torch.cuda.get_device_name(gpu_index) if device.startswith("cuda") else "CPU"
            mem_alloc = round(torch.cuda.memory_allocated(gpu_index) / 1024**3, 2) if device.startswith("cuda") else 0
            mem_total = round(torch.cuda.get_device_properties(gpu_index).total_memory / 1024**3, 2) if device.startswith("cuda") else 0

            print(f"✅ Whisper model loaded successfully.")
            print(f"   → Device: {device}")
            print(f"   → GPU: {gpu_name}")
            print(f"   → GPU Memory: {mem_alloc}/{mem_total} GB used")
            print(f"   → FP16 Enabled: {app.state.use_fp16}")
            print(f"   → Model Size: {model_name}")
        except RuntimeError as e:
            print(f"❌ Failed to load Whisper model: {e}")
            raise e

    # === Initialize ChromaDB ===
    if not getattr(app.state, "chroma_loaded", False):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, "chroma_db_openai")

            print(f"🔄 Initializing Chroma from {db_path}...")
            chroma_client = chromadb.PersistentClient(path=db_path)

            collections = chroma_client.list_collections()
            print("📁 Available Chroma collections:", [c.name for c in collections])

            if not collections:
                raise Exception("No collections found in ChromaDB")

            collection_name = os.getenv("CHROMA_COLLECTION_NAME", "langchain")
            app.state.chroma_collection = chroma_client.get_collection(collection_name)

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
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")

    try:
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file.flush()

        # Large models can benefit from temperature fallback
        result = model.transcribe(
            tmp_path,
            fp16=bool(getattr(app.state, "use_fp16", False)),
            language="en",
            task="transcribe",
            temperature=0.0,
            best_of=3,
            beam_size=5,
            condition_on_previous_text=True,
            verbose=False,
        )
        return result.get("text", "").strip()

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.post("/upload-voice/")
async def upload_voice(file: UploadFile = File(...)):
    global history  # Declare that we're using the global history variable
    
    if file.content_type not in {"audio/wav", "audio/x-wav"}:
        raise HTTPException(status_code=400, detail="Only WAV files are allowed")

    # === TOTAL START ===
    total_start = time.time()

    # === Step 1: Read and validate audio ===
    read_start = time.time()
    file_bytes = await file.read()
    metadata = validate_wav_bytes(file_bytes)
    read_end = time.time()

    # === Step 2: Speech-to-Text (Whisper STT) ===
    stt_start = time.time()
    transcription = transcribe_with_whisper_bytes(file_bytes)
    stt_end = time.time()

    # === Step 3: Retrieve documents and generate AI response (RAG) ===
    rag_start = time.time()
    collection = app.state.chroma_collection
    ai_response, history = generate_answer(transcription, collection, history)
    rag_end = time.time()
    print(f'History is: {history}')
    
    # === Step 4: Text-to-Speech (TTS) generation ===
    tts_start = time.time()
    try:
        tts_path = generate_tts_audio(ai_response, output_filename="ai_response_tts.mp3")
    except Exception as e:
        tts_path = None
        print(f"[WARN] Could not generate TTS: {e}")
    tts_end = time.time()

    # === TOTAL END ===
    total_end = time.time()

    # === Timing Summary ===
    timing_info = {
        "audio_read_validation_time": round(read_end - read_start, 2),
        "stt_time": round(stt_end - stt_start, 2),
        "rag_query_time": round(rag_end - rag_start, 2),
        "tts_generation_time": round(tts_end - tts_start, 2),
        "total_processing_time": round(total_end - total_start, 2),
    }

    # === Response ===
    return JSONResponse(
        content={
            "filname": file.filename,
            "content_type": file.content_type,
            **metadata,
            "transcription": transcription,
            "ai_response": ai_response,
            "tts_audio_path": tts_path,
            "timing_breakdown": timing_info,
        }
    )