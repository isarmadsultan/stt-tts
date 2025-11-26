import os
import tempfile
import time
import torch
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import whisper

from ai_agent import RAGAgent
from tts_service import TTSService

load_dotenv()


# ============================================================
# 1. STARTUP CLASS (Loads Whisper + ChromaDB)
# ============================================================

class Startup:
    whisper_model = None
    whisper_fp16 = False
    chroma_collection = None

    @classmethod
    def initialize(cls):
        """
        Load Whisper model and ChromaDB one time when app starts.
        """
        cls._load_whisper()
        cls._load_chroma()

    @classmethod
    def _load_whisper(cls):
        if cls.whisper_model:
            return  # already loaded

        print("\n🔄 Loading Whisper model...")

        model_name = "medium"
        gpu_index = int(os.getenv("CUDA_DEVICE", "0"))
        device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"

        if device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            cls.whisper_model = whisper.load_model(model_name, device=device)
            cls.whisper_fp16 = device.startswith("cuda") and not model_name.startswith("tiny")

            print(f"✅ Whisper loaded on {device}")
        except Exception as e:
            print(f"❌ Whisper failed to load: {e}")
            raise

    @classmethod
    def _load_chroma(cls):
        if cls.chroma_collection:
            return

        print("\n🔄 Initializing ChromaDB...")

        try:
            current = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current, "chroma_db_openai")

            client = chromadb.PersistentClient(path=db_path)

            collection_name = os.getenv("CHROMA_COLLECTION_NAME", "langchain")
            cls.chroma_collection = client.get_collection(collection_name)

            count = cls.chroma_collection.count()
            print(f"✅ ChromaDB collection loaded with {count} docs")

        except Exception as e:
            print(f"❌ ChromaDB load failed: {e}")
            raise


# ============================================================
# 2. TRANSCRIBER CLASS (Uses Startup resources)
# ============================================================

class Transcriber:

    def transcribe_bytes(self, file_bytes: bytes) -> str:
        """
        Save audio bytes to temp file → Whisper → text.
        """

        if not Startup.whisper_model:
            raise HTTPException(status_code=500, detail="Whisper not loaded")

        model = Startup.whisper_model
        fp16 = Startup.whisper_fp16

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")

        try:
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file.flush()

            result = model.transcribe(
                tmp_path,
                fp16=fp16,
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


# ============================================================
# 3. UPLOAD VOICE HANDLER (Inherits Transcriber)
# ============================================================

class UploadVoiceHandler(Transcriber):

    def __init__(self):
        self.history = []  # conversational history
        self.tts_service = TTSService()
        self.rag_agent = None

    async def process_voice_upload(self, file: UploadFile):
        """
        Full pipeline: read → transcribe → AI answer → TTS → JSON
        """

        if file.content_type not in {"audio/wav", "audio/x-wav"}:
            raise HTTPException(status_code=400, detail="Only WAV files allowed")

        total_start = time.time()

        # Step 1: Read Audio ====================================================
        read_start = time.time()
        file_bytes = await file.read()
        read_end = time.time()

        # Step 2: Transcribe ====================================================
        stt_start = time.time()
        text = self.transcribe_bytes(file_bytes)
        stt_end = time.time()

        # Step 3: RAG Agent =====================================================
        rag_start = time.time()
        
        if not self.rag_agent:
            if not Startup.chroma_collection:
                raise HTTPException(status_code=500, detail="ChromaDB not initialized")
            self.rag_agent = RAGAgent(Startup.chroma_collection)

        ai_response, self.history = self.rag_agent.answer(text, self.history)
        rag_end = time.time()

        # Step 4: TTS ===========================================================
        tts_start = time.time()
        try:
            tts_path = self.tts_service.generate(ai_response, "ai_response_tts.mp3")
        except:
            tts_path = None
        tts_end = time.time()

        # Step 5: Build Response JSON ===========================================
        total_end = time.time()

        timing = {
            "audio_read_time": round(read_end - read_start, 2),
            "stt_time": round(stt_end - stt_start, 2),
            "rag_time": round(rag_end - rag_start, 2),
            "tts_time": round(tts_end - tts_start, 2),
            "total_time": round(total_end - total_start, 2),
        }

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "transcription": text,
            "ai_response": ai_response,
            "tts_audio_path": tts_path,
            "timing": timing,
        }


# ============================================================
# 4. FASTAPI APP (Uses The OOP Classes)
# ============================================================

app = FastAPI(title="OOP Voice API")

handler = UploadVoiceHandler()


@app.on_event("startup")
def startup_event():
    Startup.initialize()


@app.post("/upload-voice/")
async def upload_voice_route(file: UploadFile = File(...)):
    response = await handler.process_voice_upload(file)
    return JSONResponse(content=response)
