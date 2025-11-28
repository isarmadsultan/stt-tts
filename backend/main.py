import os
import tempfile
import time
from typing import Dict, List, Tuple, Optional
import torch
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import whisper
from contextlib import asynccontextmanager
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from backend.ai_agent import RAGAgent
from backend.tts_service import TTSService

load_dotenv()


# ============================================================
# 1. INFRASTRUCTURE LAYER - Model & DB Initialization
# ============================================================

class WhisperModelLoader:
    """Single Responsibility: Load and manage Whisper model lifecycle"""
    
    def __init__(self, model_name: str = "medium", gpu_index: int = 0):
        self.model_name = model_name
        self.gpu_index = gpu_index
        self.model = None
        self.fp16 = False
        self.device = None
    
    def load(self) -> Tuple[whisper.Whisper, bool]:
        """Load Whisper model and return model + fp16 flag"""
        if self.model:
            return self.model, self.fp16
        
        print(f"\n🔄 Loading Whisper model '{self.model_name}'...")
        
        self.device = f"cuda:{self.gpu_index}" if torch.cuda.is_available() else "cpu"
        
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            self.fp16 = self.device.startswith("cuda") and not self.model_name.startswith("tiny")
            
            print(f"✅ Whisper loaded on {self.device}")
            return self.model, self.fp16
        
        except Exception as e:
            print(f"❌ Whisper failed to load: {e}")
            raise


# Replace ChromaDBInitializer with WeaviateInitializer
class WeaviateInitializer:
    """Single Responsibility: Initialize and manage Weaviate connection"""
    
    def __init__(self, host: str = "localhost", port: int = 9000, grpc_port: int = 50051):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.client = None
        self.collection_name = os.getenv("WEAVIATE_COLLECTION_NAME", "Document")
    
    def initialize(self) -> weaviate.WeaviateClient:
        """Initialize Weaviate and return client"""
        if self.client:
            return self.client
        
        print("\n🔄 Initializing Weaviate...")
        
        try:
            self.client = weaviate.connect_to_local(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                additional_config=AdditionalConfig(
                    timeout=Timeout(init=30, query=60, insert=120)
                )
            )
            
            # Check if collection exists
            collections = self.client.collections.list_all()
            if self.collection_name in collections:
                print(f"✅ Weaviate collection '{self.collection_name}' loaded")
            else:
                print(f"⚠️ Collection '{self.collection_name}' not found in Weaviate")
            
            return self.client
        
        except Exception as e:
            print(f"❌ Weaviate initialization failed: {e}")
            raise


# Update ServiceContainer
class ServiceContainer:
    """Single Responsibility: Manage application-level service instances"""
    
    def __init__(self):
        self.whisper_loader = None
        self.weaviate_initializer = None  # Changed from chroma_initializer
        self.whisper_model = None
        self.whisper_fp16 = False
        self.weaviate_client = None  # Changed from chroma_collection
    
    def initialize_all(self):
        """Initialize all required services"""
        # Load Whisper
        model_name = os.getenv("WHISPER_MODEL", "medium")
        gpu_index = int(os.getenv("CUDA_DEVICE", "0"))
        
        self.whisper_loader = WhisperModelLoader(model_name, gpu_index)
        self.whisper_model, self.whisper_fp16 = self.whisper_loader.load()
        
        # Initialize Weaviate
        self.weaviate_initializer = WeaviateInitializer()
        self.weaviate_client = self.weaviate_initializer.initialize()

# ============================================================
# 2. AUDIO PROCESSING LAYER
# ============================================================

class AudioFileHandler:
    """Single Responsibility: Manage temporary audio file operations"""
    
    @staticmethod
    def save_to_temp(file_bytes: bytes, suffix: str = ".wav") -> str:
        """Save bytes to temporary file and return path"""
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        
        try:
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file.flush()
            return tmp_path
        except Exception as e:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise e
    
    @staticmethod
    def cleanup(file_path: str) -> None:
        """Remove temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass


class WhisperTranscriber:
    """Single Responsibility: Transcribe audio using Whisper model"""
    
    def __init__(self, model: whisper.Whisper, fp16: bool = False):
        self.model = model
        self.fp16 = fp16
    
    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """Transcribe audio file to text"""
        try:
            result = self.model.transcribe(
                audio_path,
                fp16=self.fp16,
                language=language,
                task="transcribe",
                temperature=0.0,
                best_of=3,
                beam_size=5,
                condition_on_previous_text=True,
                verbose=False,
            )
            return result.get("text", "").strip()
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {str(e)}"
            )


# ============================================================
# 3. BUSINESS LOGIC LAYER
# ============================================================

class ConversationManager:
    """Single Responsibility: Manage conversation history"""
    
    def __init__(self):
        self.history: List = []
    
    def get_history(self) -> List:
        """Return current conversation history"""
        return self.history
    
    def update_history(self, new_history: List) -> None:
        """Update conversation history"""
        self.history = new_history
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.history = []

# Update RAGService
class RAGService:
    """Single Responsibility: Handle RAG-based question answering"""
    
    def __init__(self, weaviate_client: weaviate.WeaviateClient):  # Changed parameter type
        self.rag_agent = RAGAgent(weaviate_client)  # Pass weaviate_client instead
    
    def get_answer(self, question: str, history: List) -> Tuple[str, List]:
        """Get AI answer and updated history"""
        return self.rag_agent.answer(question, history)


class TTSServiceWrapper:
    """Single Responsibility: Handle Text-to-Speech generation"""
    
    def __init__(self):
        self.tts_service = TTSService()
    
    def generate_speech(self, text: str, filename: str = "ai_response_tts.mp3") -> Optional[str]:
        """Generate speech audio from text"""
        try:
            return self.tts_service.generate(text, filename)
        except Exception as e:
            print(f"⚠️ TTS generation failed: {e}")
            return None


# ============================================================
# 4. ORCHESTRATION LAYER
# ============================================================

class TimingMetricsCollector:
    """Single Responsibility: Collect and format timing metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
    
    def start(self, operation: str) -> None:
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end(self, operation: str) -> None:
        """End timing an operation"""
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            self.metrics[operation] = round(elapsed, 2)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all collected metrics"""
        return self.metrics.copy()


class VoicePipelineOrchestrator:
    """Single Responsibility: Coordinate the voice processing pipeline"""
    
    def __init__(
        self,
        transcriber: WhisperTranscriber,
        conversation_manager: ConversationManager,
        rag_service: RAGService,
        tts_service: TTSServiceWrapper,
        file_handler: AudioFileHandler
    ):
        self.transcriber = transcriber
        self.conversation_manager = conversation_manager
        self.rag_service = rag_service
        self.tts_service = tts_service
        self.file_handler = file_handler
        self.metrics_collector = TimingMetricsCollector()
    
    async def process(self, file_bytes: bytes) -> Dict:
        """Execute the complete voice processing pipeline"""
        temp_path = None
        
        try:
            # Step 1: Save audio to temp file
            self.metrics_collector.start("file_save")
            temp_path = self.file_handler.save_to_temp(file_bytes)
            self.metrics_collector.end("file_save")
            
            # Step 2: Transcribe
            self.metrics_collector.start("transcription")
            transcription = self.transcriber.transcribe(temp_path)
            self.metrics_collector.end("transcription")
            
            # Step 3: Get AI response
            self.metrics_collector.start("rag_processing")
            history = self.conversation_manager.get_history()
            ai_response, updated_history = self.rag_service.get_answer(transcription, history)
            self.conversation_manager.update_history(updated_history)
            self.metrics_collector.end("rag_processing")
            
            # Step 4: Generate TTS
            self.metrics_collector.start("tts_generation")
            tts_path = self.tts_service.generate_speech(ai_response)
            self.metrics_collector.end("tts_generation")
            
            return {
                "transcription": transcription,
                "ai_response": ai_response,
                "tts_audio_path": tts_path,
            }
        
        finally:
            if temp_path:
                self.file_handler.cleanup(temp_path)


class ResponseFormatter:
    """Single Responsibility: Format API responses"""
    
    @staticmethod
    def format_voice_response(
        filename: str,
        content_type: str,
        pipeline_result: Dict,
        metrics: Dict[str, float]
    ) -> Dict:
        """Format the final API response"""
        return {
            "filename": filename,
            "content_type": content_type,
            "transcription": pipeline_result["transcription"],
            "ai_response": pipeline_result["ai_response"],
            "tts_audio_path": pipeline_result["tts_audio_path"],
            "timing": metrics,
        }


# ============================================================
# 5. API LAYER (Controller)
# ============================================================

class VoiceController:
    """Single Responsibility: Handle HTTP requests and coordinate response"""
    
    def __init__(self, orchestrator: VoicePipelineOrchestrator):
        self.orchestrator = orchestrator
        self.formatter = ResponseFormatter()
    
    async def handle_upload(self, file: UploadFile) -> Dict:
        """Handle voice file upload request"""
        
        # Start total timing
        total_start = time.time()
        
        # Read file
        read_start = time.time()
        file_bytes = await file.read()
        read_time = round(time.time() - read_start, 2)
        
        # Process through pipeline
        pipeline_result = await self.orchestrator.process(file_bytes)
        
        # Get metrics and add read time
        metrics = self.orchestrator.metrics_collector.get_metrics()
        metrics["audio_read_time"] = read_time
        metrics["total_time"] = round(time.time() - total_start, 2)
        
        # Format response
        return self.formatter.format_voice_response(
            filename=file.filename,
            content_type=file.content_type,
            pipeline_result=pipeline_result,
            metrics=metrics
        )


# ============================================================
# 6. FASTAPI APPLICATION SETUP
# ============================================================

app = FastAPI(title="SRP-Compliant Voice API")

# Global service container
service_container = ServiceContainer()

# Global orchestrator (initialized after startup)
orchestrator: Optional[VoicePipelineOrchestrator] = None
controller: Optional[VoiceController] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global orchestrator, controller
    
    # Initialize infrastructure
    service_container.initialize_all()
    
    # Create service instances
    transcriber = WhisperTranscriber(
        service_container.whisper_model,
        service_container.whisper_fp16
    )
    
    conversation_manager = ConversationManager()
    rag_service = RAGService(service_container.weaviate_client)
    tts_service = TTSServiceWrapper()
    file_handler = AudioFileHandler()
    
    # Create orchestrator
    orchestrator = VoicePipelineOrchestrator(
        transcriber=transcriber,
        conversation_manager=conversation_manager,
        rag_service=rag_service,
        tts_service=tts_service,
        file_handler=file_handler
    )
    
    # Create controller
    controller = VoiceController(orchestrator)
    
    yield  # Application runs here
    
    # Shutdown (optional cleanup)
    if service_container.weaviate_client:
        service_container.weaviate_client.close()
        print("🔌 Closed Weaviate connection")

# Update FastAPI app initialization
app = FastAPI(title="SRP-Compliant Voice API", lifespan=lifespan)


@app.post("/upload-voice/")
async def upload_voice_route(file: UploadFile = File(...)):
    """Voice upload endpoint"""
    if not controller:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    response = await controller.handle_upload(file)
    return JSONResponse(content=response)


# Update health_check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_loaded": service_container.whisper_model is not None,
        "weaviate_loaded": service_container.weaviate_client is not None,  # Changed from chroma_loaded
    }