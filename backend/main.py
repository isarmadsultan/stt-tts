"""
Updated Main Application with SOLID TTS Integration
====================================================

Key Changes:
1. Integrated new SOLID-compliant TTS service
2. Added proper dependency injection
3. Enhanced error handling and cleanup
4. Improved streaming pipeline coordination
5. Better separation of concerns
"""

import os
import tempfile
import time
import threading
import queue
import pygame
from typing import Dict, List, Tuple, Optional
import torch
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import whisper
from contextlib import asynccontextmanager
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from pathlib import Path

from backend.ai_agent import StreamingRAGAgent

# Import new SOLID TTS components
from backend.tts_service import (
    StreamingTTSService,
    TTSServiceFactory,
    TTSConfig,
    AudioChunk,
    ITTSProvider,
    IFileStorage,
    OpenAITTSProvider,
    LocalFileStorage
)

load_dotenv()


# ============================================================
# 1. INFRASTRUCTURE LAYER - Model & DB Initialization
# ============================================================

class WhisperModelLoader:
    """Single Responsibility: Load and manage Whisper model lifecycle"""
    
    def __init__(self, model_name: str = "large", gpu_index: int = 0):
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


class TTSServiceInitializer:
    """
    Single Responsibility: Initialize TTS service with proper configuration
    D - Dependency Inversion: Can inject different providers/storage
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[ITTSProvider] = None,
        storage: Optional[IFileStorage] = None,
        config: Optional[TTSConfig] = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.provider = provider
        self.storage = storage
        self.config = config or self._load_config_from_env()
    
    def _load_config_from_env(self) -> TTSConfig:
        """Load TTS configuration from environment variables"""
        return TTSConfig(
            voice=os.getenv("TTS_VOICE", "alloy"),
            model=os.getenv("TTS_MODEL", "tts-1"),
            max_concurrent_tasks=int(os.getenv("TTS_MAX_CONCURRENT", "5")),
            max_retries=int(os.getenv("TTS_MAX_RETRIES", "2")),
            output_dir=Path(os.getenv("TTS_OUTPUT_DIR", tempfile.gettempdir())) / "tts_audio"
        )
    
    def initialize(self) -> StreamingTTSService:
        """Initialize and return TTS service"""
        print("\n🔄 Initializing TTS Service...")
        
        try:
            # Use factory if no custom components provided
            if self.provider is None and self.storage is None:
                # Check environment for optimization preference
                optimization = os.getenv("TTS_OPTIMIZATION", "default")
                
                if optimization == "throughput":
                    service = TTSServiceFactory.create_high_throughput(self.api_key)
                    print("✅ TTS Service initialized (High Throughput mode)")
                elif optimization == "quality":
                    service = TTSServiceFactory.create_high_quality(self.api_key)
                    print("✅ TTS Service initialized (High Quality mode)")
                else:
                    service = StreamingTTSService(
                        api_key=self.api_key,
                        config=self.config
                    )
                    print("✅ TTS Service initialized (Default mode)")
            else:
                # Use custom components
                service = StreamingTTSService(
                    tts_provider=self.provider,
                    file_storage=self.storage,
                    api_key=self.api_key,
                    config=self.config
                )
                print("✅ TTS Service initialized (Custom configuration)")
            
            return service
        
        except Exception as e:
            print(f"❌ TTS Service initialization failed: {e}")
            raise


class ServiceContainer:
    """Single Responsibility: Manage application-level service instances"""
    
    def __init__(self):
        self.whisper_loader = None
        self.weaviate_initializer = None
        self.tts_initializer = None
        
        self.whisper_model = None
        self.whisper_fp16 = False
        self.weaviate_client = None
        self.tts_service = None
    
    def initialize_all(self):
        """Initialize all required services"""
        # Load Whisper with LARGE model
        model_name = os.getenv("WHISPER_MODEL", "large")
        gpu_index = int(os.getenv("CUDA_DEVICE", "0"))
        
        self.whisper_loader = WhisperModelLoader(model_name, gpu_index)
        self.whisper_model, self.whisper_fp16 = self.whisper_loader.load()
        
        # Initialize Weaviate
        self.weaviate_initializer = WeaviateInitializer()
        self.weaviate_client = self.weaviate_initializer.initialize()
        
        # Initialize TTS
        self.tts_initializer = TTSServiceInitializer()
        self.tts_service = self.tts_initializer.initialize()



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


class StreamingRAGService:
    """Single Responsibility: Handle RAG-based question answering with streaming"""
    
    def __init__(self, weaviate_client: weaviate.WeaviateClient):
        self.rag_agent = StreamingRAGAgent(weaviate_client)
    
    async def get_answer_streaming(self, question: str, history: List):
        """Get AI answer stream"""
        async for chunk in self.rag_agent.answer_streaming(question, history):
            yield chunk
    
    async def get_full_answer(self, question: str, history: List) -> Tuple[str, List]:
        """Get complete AI answer and updated history"""
        return await self.rag_agent.answer(question, history)


class AudioChunkCollector:
    """
    Single Responsibility: Collect and manage audio chunks during streaming.
    Maintains order and provides access to chunk metadata.
    """
    
    def __init__(self):
        self.chunks: List[AudioChunk] = []
        self.chunk_count = 0
    
    def add_chunk(self, chunk: AudioChunk) -> None:
        """Add audio chunk to collection"""
        self.chunks.append(chunk)
        self.chunk_count += 1
    
    def get_ordered_paths(self) -> List[str]:
        """Get audio file paths in order"""
        # Chunks are already ordered by design
        return [str(chunk.file_path) for chunk in self.chunks]
    
    def get_total_duration(self) -> float:
        """Get estimated total duration of all chunks"""
        return sum(chunk.duration_estimate for chunk in self.chunks)
    
    def get_count(self) -> int:
        """Get number of chunks"""
        return self.chunk_count


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


class AudioPlayer:
    """
    Single Responsibility: Play audio files sequentially from a queue.
    Thread-safe audio playback manager.
    """
    
    def __init__(self, first_audio_callback=None):
        self.queue = queue.Queue()
        self.playback_enabled = False
        self.first_audio_callback = first_audio_callback
        self.first_audio_played = False
        
        try:
            pygame.mixer.init()
            self.playback_enabled = True
            print("✅ Audio playback enabled")
        except pygame.error as e:
            print(f"⚠️ Audio device not found. Playback disabled: {e}")
        
        self.thread = threading.Thread(target=self._play_worker, daemon=True)
        self.thread.start()
    
    def play(self, file_path: str) -> None:
        """Add file to playback queue"""
        if self.playback_enabled:
            self.queue.put(file_path)
    
    def _play_worker(self) -> None:
        """Worker thread to play audio sequentially"""
        while True:
            file_path = self.queue.get()
            if file_path is None:
                break
            
            try:
                if self.playback_enabled and pygame.mixer.get_init():
                    pygame.mixer.music.load(file_path)
                    pygame.mixer.music.play()
                    
                    # Trigger callback when first audio starts playing
                    if not self.first_audio_played and self.first_audio_callback:
                        self.first_audio_callback()
                        self.first_audio_played = True
                    
                    # Wait until playback finishes
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ Error playing {file_path}: {e}")
            finally:
                self.queue.task_done()
    
    def stop(self) -> None:
        """Stop playback and cleanup"""
        self.queue.put(None)
        if self.playback_enabled:
            pygame.mixer.music.stop()
    
    def reset_first_audio_flag(self) -> None:
        """Reset the first audio flag for next request"""
        self.first_audio_played = False


class StreamingVoicePipelineOrchestrator:
    """
    Single Responsibility: Coordinate the streaming voice processing pipeline.
    
    Key Improvements:
    - Uses new SOLID TTS service with proper cleanup
    - Better error handling throughout pipeline
    - Automatic resource cleanup via context manager
    - Detailed metrics collection
    """
    
    def __init__(
        self,
        transcriber: WhisperTranscriber,
        conversation_manager: ConversationManager,
        rag_service: StreamingRAGService,
        tts_service: StreamingTTSService,
        file_handler: AudioFileHandler,
        enable_playback: bool = True
    ):
        self.transcriber = transcriber
        self.conversation_manager = conversation_manager
        self.rag_service = rag_service
        self.tts_service = tts_service
        self.file_handler = file_handler
        self.metrics_collector = TimingMetricsCollector()
        self.audio_player = AudioPlayer() if enable_playback else None
    
    async def process_streaming(self, file_bytes: bytes) -> Dict:
        """
        Execute the streaming voice processing pipeline with parallelization.
        
        Pipeline:
        1. Save & Transcribe audio (sequential)
        2. RAG + TTS streaming (parallel)
        3. Playback (async)
        4. History update (sequential)
        5. Cleanup (automatic)
        
        Tracks two key metrics:
        - time_to_first_audio: Time from transcription end to first audio playback (agent response time)
        - total_time: Time for entire pipeline completion
        """
        temp_path = None
        chunk_collector = AudioChunkCollector()
        first_audio_callback_triggered = False
        
        # Define callback for first audio playback
        def on_first_audio_start():
            nonlocal first_audio_callback_triggered
            if not first_audio_callback_triggered:
                self.metrics_collector.end("time_to_first_audio")
                first_audio_callback_triggered = True
                print("🎯 First audio started playing!")
        
        # Reset audio player flag for this request
        if self.audio_player:
            self.audio_player.reset_first_audio_flag()
            # Update callback
            self.audio_player.first_audio_callback = on_first_audio_start
        
        try:
            # Step 1: Save audio to temp file
            self.metrics_collector.start("file_save")
            temp_path = self.file_handler.save_to_temp(file_bytes)
            self.metrics_collector.end("file_save")
            
            # Step 2: Transcribe
            self.metrics_collector.start("transcription")
            transcription = self.transcriber.transcribe(temp_path)
            self.metrics_collector.end("transcription")
            
            if not transcription:
                raise HTTPException(
                    status_code=400,
                    detail="No transcription generated from audio"
                )
            
            # Step 3: Start tracking time to first audio (from end of transcription)
            self.metrics_collector.start("time_to_first_audio")
            
            # Step 4: Parallel RAG + TTS pipeline with automatic cleanup
            self.metrics_collector.start("parallel_rag_tts")
            
            history = self.conversation_manager.get_history()
            
            # Create async generator for RAG streaming
            text_stream = self.rag_service.get_answer_streaming(transcription, history)
            
            # Use context manager for automatic TTS cleanup
            async with self.tts_service.streaming_session(text_stream) as tts_stream:
                async for chunk in tts_stream:
                    # Collect chunk metadata
                    chunk_collector.add_chunk(chunk)
                    
                    # Play immediately for real-time experience
                    if self.audio_player:
                        self.audio_player.play(str(chunk.file_path))
                    
                    print(f"🎵 Chunk {chunk.chunk_id} ready: {chunk.text[:50]}...")
            
            self.metrics_collector.end("parallel_rag_tts")
            
            # If no audio player or playback disabled, manually end the timer
            if not self.audio_player and not first_audio_callback_triggered:
                self.metrics_collector.end("time_to_first_audio")
            
            # Step 5: Get full answer for history update
            # (This is cached in the RAG agent, so it's fast)
            self.metrics_collector.start("history_update")
            full_answer, updated_history = await self.rag_service.get_full_answer(
                transcription, 
                history
            )
            self.conversation_manager.update_history(updated_history)
            self.metrics_collector.end("history_update")
            
            # Wait a bit for audio callback to trigger if it hasn't yet
            if self.audio_player and not first_audio_callback_triggered:
                await asyncio.sleep(0.2)
                if not first_audio_callback_triggered:
                    # Fallback: manually end the timer
                    self.metrics_collector.end("time_to_first_audio")
            
            # Return results with metrics
            return {
                "transcription": transcription,
                "ai_response": full_answer,
                "tts_audio_paths": chunk_collector.get_ordered_paths(),
                "num_audio_chunks": chunk_collector.get_count(),
                "total_audio_duration": chunk_collector.get_total_duration(),
            }
        
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline processing failed: {str(e)}"
            )
        
        finally:
            # Cleanup input audio file
            if temp_path:
                self.file_handler.cleanup(temp_path)
            
            # Note: TTS chunk cleanup is handled automatically by context manager


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
            "tts_audio_paths": pipeline_result["tts_audio_paths"],
            "num_audio_chunks": pipeline_result["num_audio_chunks"],
            "total_audio_duration": pipeline_result.get("total_audio_duration", 0.0),
            "timing": metrics,
            "status": "success"
        }
    
    @staticmethod
    def format_error_response(error: Exception, metrics: Optional[Dict] = None) -> Dict:
        """Format error response"""
        return {
            "status": "error",
            "error": str(error),
            "error_type": type(error).__name__,
            "timing": metrics or {}
        }


# ============================================================
# 5. API LAYER (Controller)
# ============================================================

class VoiceController:
    """Single Responsibility: Handle HTTP requests and coordinate response"""
    
    def __init__(self, orchestrator: StreamingVoicePipelineOrchestrator):
        self.orchestrator = orchestrator
        self.formatter = ResponseFormatter()
    
    async def handle_upload(self, file: UploadFile) -> Dict:
        """Handle voice file upload request with streaming pipeline"""
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid content type: {file.content_type}. Expected audio/*"
            )
        
        # Start total timing
        total_start = time.time()
        
        try:
            # Read file
            read_start = time.time()
            file_bytes = await file.read()
            read_time = round(time.time() - read_start, 2)
            
            if not file_bytes:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            
            # Process through streaming pipeline
            pipeline_result = await self.orchestrator.process_streaming(file_bytes)
            
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
        
        except HTTPException:
            raise
        except Exception as e:
            metrics = self.orchestrator.metrics_collector.get_metrics()
            metrics["total_time"] = round(time.time() - total_start, 2)
            
            error_response = self.formatter.format_error_response(e, metrics)
            return error_response


# ============================================================
# 6. FASTAPI APPLICATION SETUP
# ============================================================

# Global service container
service_container = ServiceContainer()

# Global orchestrator (initialized after startup)
orchestrator: Optional[StreamingVoicePipelineOrchestrator] = None
controller: Optional[VoiceController] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global orchestrator, controller
    
    print("\n" + "="*60)
    print("🚀 Starting Streaming Voice API with SOLID TTS")
    print("="*60)
    
    # Initialize infrastructure
    service_container.initialize_all()
    
    # Create service instances
    transcriber = WhisperTranscriber(
        service_container.whisper_model,
        service_container.whisper_fp16
    )
    
    conversation_manager = ConversationManager()
    rag_service = StreamingRAGService(service_container.weaviate_client)
    file_handler = AudioFileHandler()
    
    # Check if playback should be enabled
    enable_playback = os.getenv("ENABLE_AUDIO_PLAYBACK", "true").lower() == "true"
    
    # Create orchestrator with new TTS service
    orchestrator = StreamingVoicePipelineOrchestrator(
        transcriber=transcriber,
        conversation_manager=conversation_manager,
        rag_service=rag_service,
        tts_service=service_container.tts_service,  # New SOLID TTS service
        file_handler=file_handler,
        enable_playback=enable_playback
    )
    
    # Create controller
    controller = VoiceController(orchestrator)
    
    print("\n✅ All services initialized successfully")
    print("="*60 + "\n")
    
    yield  # Application runs here
    
    # Shutdown
    print("\n" + "="*60)
    print("🛑 Shutting down Streaming Voice API")
    print("="*60)
    
    # Stop audio player
    if orchestrator and orchestrator.audio_player:
        orchestrator.audio_player.stop()
        print("🔇 Audio player stopped")
    
    # Cleanup TTS service
    if service_container.tts_service:
        await service_container.tts_service.cleanup()
        print("🧹 TTS service cleaned up")
    
    # Close Weaviate connection
    if service_container.weaviate_client:
        service_container.weaviate_client.close()
        print("🔌 Weaviate connection closed")
    
    print("="*60 + "\n")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Streaming Voice API with SOLID TTS",
    description="Voice processing pipeline with RAG and streaming TTS using SOLID principles",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================
# 7. API ENDPOINTS
# ============================================================

@app.post("/upload-voice/")
async def upload_voice_route(file: UploadFile = File(...)):
    """
    Voice upload endpoint with streaming pipeline.
    
    Accepts audio file, transcribes it, generates AI response via RAG,
    and converts to speech using streaming TTS with parallel processing.
    """
    if not controller:
        raise HTTPException(
            status_code=503, 
            detail="Service not initialized. Please wait for startup to complete."
        )
    
    response = await controller.handle_upload(file)
    return JSONResponse(content=response)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns status of all critical services.
    """
    return {
        "status": "healthy",
        "services": {
            "whisper": service_container.whisper_model is not None,
            "weaviate": service_container.weaviate_client is not None,
            "tts": service_container.tts_service is not None,
        },
        "features": {
            "streaming_enabled": True,
            "parallel_processing": True,
            "solid_architecture": True,
        },
        "version": "2.0.0"
    }


@app.post("/clear-history/")
async def clear_history():
    """Clear conversation history"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    orchestrator.conversation_manager.clear_history()
    return {"status": "success", "message": "Conversation history cleared"}


@app.get("/conversation-history/")
async def get_conversation_history():
    """Get current conversation history"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    history = orchestrator.conversation_manager.get_history()
    return {
        "status": "success",
        "history": history,
        "count": len(history)
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Streaming Voice API with SOLID TTS",
        "version": "2.0.0",
        "architecture": "SOLID principles",
        "endpoints": {
            "upload": "/upload-voice/",
            "health": "/health",
            "history_clear": "/clear-history/",
            "history_get": "/conversation-history/"
        },
        "features": [
            "Whisper transcription",
            "RAG-based AI responses",
            "Streaming TTS with parallel processing",
            "Automatic resource cleanup",
            "SOLID architecture",
            "Conversation history management"
        ]
    }


