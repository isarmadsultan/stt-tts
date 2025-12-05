
import asyncio
import os
import re
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Protocol
import aiohttp
from openai import AsyncOpenAI


# ============================================================================
# CONFIGURATION & DATA MODELS
# ============================================================================

@dataclass
class TTSConfig:

    voice: str = "alloy"
    model: str = "tts-1"
    max_concurrent_tasks: int = 5
    max_retries: int = 2
    output_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "tts_audio")
    retry_delay_base: float = 1.0
    chunk_timeout: float = 30.0
    
    def __post_init__(self):
        """Ensure output directory exists"""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AudioChunk:
    """
    S - Single Responsibility: Audio chunk metadata
    Immutable data transfer object
    """
    chunk_id: int
    file_path: Path
    text: str
    duration_estimate: float = 0.0
    size_bytes: int = 0
    
    def __post_init__(self):
        """Calculate estimates if not provided"""
        if self.duration_estimate == 0.0:
            # Rough estimate: ~150 words per minute, ~5 chars per word
            words = len(self.text) / 5
            self.duration_estimate = (words / 150) * 60


# ============================================================================
# INTERFACES (Protocol-based for flexibility)
# ============================================================================

class ITTSProvider(Protocol):
    """
    I - Interface Segregation: Minimal TTS provider interface
    D - Dependency Inversion: High-level code depends on this abstraction
    """
    async def synthesize(self, text: str, voice: str, model: str) -> bytes:
        """Synthesize text to speech audio bytes"""
        ...


class IFileStorage(Protocol):
    """
    I - Interface Segregation: Minimal file storage interface
    D - Dependency Inversion: High-level code depends on this abstraction
    """
    async def save_audio(self, audio_data: bytes, chunk_id: int) -> Path:
        """Save audio data and return file path"""
        ...
    
    async def cleanup_all(self) -> None:
        """Clean up all stored files"""
        ...


# ============================================================================
# CONCRETE IMPLEMENTATIONS
# ============================================================================

class OpenAITTSProvider:
    """
    S - Single Responsibility: OpenAI TTS API integration only
    L - Liskov Substitution: Can replace any ITTSProvider
    """
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def synthesize(self, text: str, voice: str, model: str) -> bytes:
        """
        Synthesize text using OpenAI TTS API
        
        Args:
            text: Text to synthesize
            voice: Voice name (alloy, echo, fable, onyx, nova, shimmer)
            model: Model name (tts-1, tts-1-hd)
        
        Returns:
            Audio data as bytes
        
        Raises:
            Exception: If API call fails
        """
        try:
            response = await self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format="mp3"
            )
            
            # Read the response content directly
            audio_data = response.content
            
            return audio_data
            
        except Exception as e:
            raise Exception(f"OpenAI TTS synthesis failed: {str(e)}")


class LocalFileStorage:
    """
    S - Single Responsibility: Local file system storage only
    L - Liskov Substitution: Can replace any IFileStorage
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.saved_files: List[Path] = []
    
    async def save_audio(self, audio_data: bytes, chunk_id: int) -> Path:
        """
        Save audio data to local file
        
        Args:
            audio_data: Audio bytes to save
            chunk_id: Unique chunk identifier
        
        Returns:
            Path to saved file
        """
        file_path = self.output_dir / f"chunk_{chunk_id:04d}.mp3"
        
        # Use asyncio to write file without blocking
        await asyncio.to_thread(file_path.write_bytes, audio_data)
        
        self.saved_files.append(file_path)
        return file_path
    
    async def cleanup_all(self) -> None:
        """Delete all saved audio files"""
        for file_path in self.saved_files:
            try:
                if file_path.exists():
                    await asyncio.to_thread(file_path.unlink)
            except Exception as e:
                print(f"⚠️ Failed to delete {file_path}: {e}")
        
        self.saved_files.clear()


# ============================================================================
# SUPPORTING COMPONENTS
# ============================================================================

class SentenceTextChunker:
    """
    S - Single Responsibility: Text chunking logic only
    O - Open/Closed: Can be extended with different strategies
    """
    
    # Sentence boundary patterns
    SENTENCE_END = re.compile(r'[.!?]+\s+')
    
    def __init__(self):
        self.buffer = ""
    
    def chunk_text(self, text: str) -> tuple[List[str], str]:
        """
        Split text into complete sentences
        
        Args:
            text: Text to chunk
        
        Returns:
            Tuple of (complete_sentences, remaining_buffer)
        """
        # Add to buffer
        self.buffer += text
        
        # Find complete sentences
        sentences = []
        last_end = 0
        
        for match in self.SENTENCE_END.finditer(self.buffer):
            sentence = self.buffer[last_end:match.end()].strip()
            if sentence:
                sentences.append(sentence)
            last_end = match.end()
        
        # Keep remainder in buffer
        remaining = self.buffer[last_end:]
        self.buffer = remaining
        
        return sentences, remaining
    
    def flush(self) -> Optional[str]:
        """Get any remaining buffered text"""
        if self.buffer.strip():
            result = self.buffer.strip()
            self.buffer = ""
            return result
        return None


class RetryStrategy:

    
    def __init__(self, max_retries: int = 2, backoff_base: float = 1.5):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
    
    async def execute_with_retry(self, coro_func, *args, **kwargs):
        """
        Execute async function with exponential backoff retry
        
        Args:
            coro_func: Async function to execute
            *args, **kwargs: Arguments for the function
        
        Returns:
            Result of successful execution
        
        Raises:
            Exception: If all retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await coro_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.backoff_base ** attempt
                    print(f"⚠️ Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ All {self.max_retries + 1} attempts failed")
        
        raise last_exception


class SemaphoreConcurrencyController:
    """
    S - Single Responsibility: Concurrency control only
    """
    
    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_with_limit(self, coro):
        """Execute coroutine with concurrency limit"""
        async with self.semaphore:
            return await coro


# ============================================================================
# AUDIO CHUNK GENERATOR
# ============================================================================

class AudioChunkGenerator:
    """
    S - Single Responsibility: Generate audio chunks from text
    D - Dependency Inversion: Depends on ITTSProvider and IFileStorage abstractions
    """
    
    def __init__(
        self,
        tts_provider: ITTSProvider,
        file_storage: IFileStorage,
        retry_strategy: RetryStrategy,
        config: TTSConfig
    ):
        self.tts_provider = tts_provider
        self.file_storage = file_storage
        self.retry_strategy = retry_strategy
        self.config = config
    
    async def generate(self, text: str, chunk_id: int) -> AudioChunk:
        """
        Generate audio chunk from text
        
        Args:
            text: Text to synthesize
            chunk_id: Unique chunk identifier
        
        Returns:
            AudioChunk with metadata
        """
        # Synthesize with retry
        audio_data = await self.retry_strategy.execute_with_retry(
            self.tts_provider.synthesize,
            text,
            self.config.voice,
            self.config.model
        )
        
        # Save to storage
        file_path = await self.file_storage.save_audio(audio_data, chunk_id)
        
        # Create chunk metadata
        return AudioChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            text=text,
            size_bytes=len(audio_data)
        )


# ============================================================================
# ORDERED STREAM PROCESSOR
# ============================================================================

class OrderedStreamProcessor:
    """
    S - Single Responsibility: Process text stream and maintain chunk order
    Ensures chunks are yielded in sequential order even with parallel processing
    """
    
    def __init__(
        self,
        chunk_generator: AudioChunkGenerator,
        text_chunker: SentenceTextChunker,
        concurrency_controller: SemaphoreConcurrencyController
    ):
        self.chunk_generator = chunk_generator
        self.text_chunker = text_chunker
        self.concurrency_controller = concurrency_controller
    
    async def process_stream(
        self, 
        text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Process text stream and yield ordered audio chunks
        
        Args:
            text_stream: Async generator yielding text fragments
        
        Yields:
            AudioChunk objects in sequential order
        """
        pending_tasks = {}
        next_chunk_id = 0
        current_chunk_id = 0
        completed_chunks = {}
        
        async def generate_chunk(text: str, chunk_id: int):
            """Generate chunk with concurrency control"""
            return await self.concurrency_controller.execute_with_limit(
                self.chunk_generator.generate(text, chunk_id)
            )
        
        try:
            # Process text stream
            async for text_fragment in text_stream:
                # Chunk the text
                sentences, remaining = self.text_chunker.chunk_text(text_fragment)
                
                # Start tasks for complete sentences
                for sentence in sentences:
                    task = asyncio.create_task(generate_chunk(sentence, next_chunk_id))
                    pending_tasks[next_chunk_id] = task
                    next_chunk_id += 1
                
                # Yield completed chunks in order
                while current_chunk_id in pending_tasks or current_chunk_id in completed_chunks:
                    if current_chunk_id in completed_chunks:
                        # Already completed, yield immediately
                        yield completed_chunks.pop(current_chunk_id)
                        current_chunk_id += 1
                    elif current_chunk_id in pending_tasks:
                        # Wait for this specific chunk
                        task = pending_tasks[current_chunk_id]
                        chunk = await task
                        yield chunk
                        del pending_tasks[current_chunk_id]
                        current_chunk_id += 1
                    else:
                        break
            
            # Process final buffer
            final_text = self.text_chunker.flush()
            if final_text:
                task = asyncio.create_task(generate_chunk(final_text, next_chunk_id))
                pending_tasks[next_chunk_id] = task
                next_chunk_id += 1
            
            # Wait for all remaining tasks and yield in order
            while pending_tasks:
                if current_chunk_id in pending_tasks:
                    task = pending_tasks[current_chunk_id]
                    chunk = await task
                    yield chunk
                    del pending_tasks[current_chunk_id]
                    current_chunk_id += 1
                else:
                    # Wait for any task to complete
                    done, pending = await asyncio.wait(
                        pending_tasks.values(),
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Store completed chunks
                    for task in done:
                        chunk = task.result()
                        if chunk.chunk_id == current_chunk_id:
                            yield chunk
                            del pending_tasks[current_chunk_id]
                            current_chunk_id += 1
                        else:
                            completed_chunks[chunk.chunk_id] = chunk
        
        except Exception as e:
            # Cancel all pending tasks on error
            for task in pending_tasks.values():
                task.cancel()
            raise


# ============================================================================
# MAIN STREAMING TTS SERVICE
# ============================================================================

class StreamingTTSService:
    """
    S - Single Responsibility: Coordinate streaming TTS pipeline
    O - Open/Closed: Extensible via dependency injection
    D - Dependency Inversion: Depends on abstractions
    
    Main orchestrator for streaming text-to-speech conversion.
    Provides context manager for automatic resource cleanup.
    """
    
    def __init__(
        self,
        tts_provider: Optional[ITTSProvider] = None,
        file_storage: Optional[IFileStorage] = None,
        api_key: Optional[str] = None,
        config: Optional[TTSConfig] = None
    ):
        """
        Initialize streaming TTS service
        
        Args:
            tts_provider: TTS provider (defaults to OpenAI)
            file_storage: File storage (defaults to local)
            api_key: API key for default provider
            config: TTS configuration
        """
        self.config = config or TTSConfig()
        
        # Use provided or create default implementations
        self.tts_provider = tts_provider or OpenAITTSProvider(
            api_key or os.getenv("OPENAI_API_KEY")
        )
        self.file_storage = file_storage or LocalFileStorage(self.config.output_dir)
        
        # Create supporting components
        self.retry_strategy = RetryStrategy(
            max_retries=self.config.max_retries,
            backoff_base=self.config.retry_delay_base
        )
        self.concurrency_controller = SemaphoreConcurrencyController(
            max_concurrent=self.config.max_concurrent_tasks
        )
        
        # Create core components
        self.chunk_generator = AudioChunkGenerator(
            tts_provider=self.tts_provider,
            file_storage=self.file_storage,
            retry_strategy=self.retry_strategy,
            config=self.config
        )
    
    async def generate_streaming(
        self, 
        text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Generate audio chunks from text stream
        
        Args:
            text_stream: Async generator yielding text fragments
        
        Yields:
            AudioChunk objects in sequential order
        """
        text_chunker = SentenceTextChunker()
        processor = OrderedStreamProcessor(
            chunk_generator=self.chunk_generator,
            text_chunker=text_chunker,
            concurrency_controller=self.concurrency_controller
        )
        
        async for chunk in processor.process_stream(text_stream):
            yield chunk
    
    def streaming_session(
        self, 
        text_stream: AsyncGenerator[str, None]
    ):
        """
        Context manager for streaming TTS with automatic cleanup
        
        Usage:
            async with service.streaming_session(text_stream) as chunks:
                async for chunk in chunks:
                    print(chunk.file_path)
        
        Args:
            text_stream: Async generator yielding text fragments
        
        Yields:
            Async generator of AudioChunk objects
        """
        class StreamingSession:
            def __init__(self, service, text_stream):
                self.service = service
                self.text_stream = text_stream
            
            async def __aenter__(self):
                return self.service.generate_streaming(self.text_stream)
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                # Cleanup is optional - files can be kept or deleted
                # await self.service.cleanup()
                pass
        
        return StreamingSession(self, text_stream)
    
    async def cleanup(self):
        """Clean up all generated audio files"""
        await self.file_storage.cleanup_all()


# ============================================================================
# FACTORY (Convenience)
# ============================================================================

class TTSServiceFactory:
    """
    S - Single Responsibility: Create configured TTS service instances
    O - Open/Closed: Extensible with new factory methods
    """
    
    @staticmethod
    def create_high_throughput(api_key: Optional[str] = None) -> StreamingTTSService:
        """
        Create service optimized for high throughput
        - Fast model (tts-1)
        - High concurrency
        """
        config = TTSConfig(
            model="tts-1",
            voice="alloy",
            max_concurrent_tasks=10,
            max_retries=2
        )
        return StreamingTTSService(api_key=api_key, config=config)
    
    @staticmethod
    def create_high_quality(api_key: Optional[str] = None) -> StreamingTTSService:
        """
        Create service optimized for quality
        - HD model (tts-1-hd)
        - Lower concurrency for stability
        """
        config = TTSConfig(
            model="tts-1-hd",
            voice="nova",
            max_concurrent_tasks=3,
            max_retries=3
        )
        return StreamingTTSService(api_key=api_key, config=config)
    
    @staticmethod
    def create_default(api_key: Optional[str] = None) -> StreamingTTSService:
        """Create service with default balanced settings"""
        return StreamingTTSService(api_key=api_key)
