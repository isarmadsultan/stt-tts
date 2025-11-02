# 🎙️ STT-TTS Prototype

A voice-enabled conversational AI prototype that combines Speech-to-Text (STT), Retrieval-Augmented Generation (RAG), and Text-to-Speech (TTS) technologies. Speak to the system, get intelligent responses from a knowledge base, and hear the answers spoken back to you.

## ✨ Features

- **🎤 Voice Recording**: Record audio directly from your microphone through the web interface
- **🗣️ Speech-to-Text**: Transcribe voice recordings using OpenAI Whisper
- **🧠 Intelligent RAG Agent**: Answer questions using context from a ChromaDB vector database with fuzzy matching capabilities
- **🔊 Text-to-Speech**: Convert AI responses to natural-sounding speech using OpenAI TTS
- **🌐 Web Interface**: User-friendly Streamlit frontend for voice interactions
- **🚀 FastAPI Backend**: High-performance REST API for processing voice inputs

## 🏗️ Architecture

```
┌─────────────┐
│  Streamlit  │  ← Web Interface
│   Frontend  │
└──────┬──────┘
       │ HTTP POST
       ↓
┌─────────────┐
│   FastAPI   │  ← Backend Server
│   Backend   │
└──────┬──────┘
       │
       ├─→ Whisper (STT)
       ├─→ ChromaDB (Vector Search)
       ├─→ GPT-4o-mini (RAG)
       └─→ OpenAI TTS
```

## 📋 Prerequisites

- Python 3.11+
- OpenAI API key
- Microphone access (for recording)
- ChromaDB knowledge base set up (see setup instructions)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/isarmadsultan/stt-tts.git
cd stt-tts-prototype
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r req.txt
```

### 4. Environment Setup

Create a `.env` file in the `backend/` directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
WHISPER_MODEL=medium  # Options: tiny, base, small, medium, large
CHROMA_COLLECTION_NAME=langchain  # Your ChromaDB collection name
```

### 5. Set Up ChromaDB

Ensure you have a ChromaDB collection with embedded documents. The vector database should be located at `backend/chroma_db_openai/`. 

If you need to create/embed documents, use the `backend/embedder_vectordb.py` script with your documents.

## 🎯 Usage

### Starting the Backend Server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The FastAPI server will start on `http://127.0.0.1:8000`

### Starting the Frontend

In a new terminal:

```bash
streamlit run streamlit_app.py
```

The Streamlit app will open in your browser at `http://localhost:8501`

### Using the Application

1. **Click "🎤 Speak Now"** in the Streamlit interface
2. **Speak for up to 10 seconds** - your voice will be recorded
3. **Wait for processing**:
   - Audio is transcribed to text using Whisper
   - Query is processed through RAG with ChromaDB
   - AI generates a contextual response
   - Response is converted to speech
4. **Listen to the response** - The AI's spoken answer will play automatically

## 📁 Project Structure

```
stt-tts-prototype/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI server and endpoints
│   ├── ai_agent.py          # RAG agent with ChromaDB integration
│   ├── tts_service.py       # OpenAI TTS service
│   ├── record_voice.py      # Voice recording functionality
│   ├── voice_input.py       # Voice input configuration
│   ├── embedder_vectordb.py # ChromaDB embedding utilities
│   ├── chroma_db_openai/    # ChromaDB vector database
│   └── recordings/          # Stored voice recordings
├── streamlit_app.py         # Streamlit frontend
├── req.txt                  # Python dependencies
├── Dockerfile               # Docker configuration
└── README.md                # This file
```

## 🔧 Configuration

### Whisper Models

The Whisper model can be configured in `.env`:

- `tiny`: Fastest, least accurate
- `base`: Good balance
- `small`: Better accuracy
- `medium`: **Recommended** - Good balance of speed and accuracy
- `large`: Most accurate, slowest

### TTS Voices

Edit `backend/tts_service.py` to change the TTS voice:
- `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

## 🐳 Docker Support

Build and run with Docker:

```bash
docker build -t stt-tts-prototype .
docker run -p 8501:8501 -p 8000:8000 stt-tts-prototype
```

**Note**: Make sure to configure your `.env` file or use environment variables in the container.

## 🛠️ Technologies Used

- **FastAPI**: Modern Python web framework for the backend API
- **Streamlit**: Rapid web app development for the frontend
- **OpenAI Whisper**: State-of-the-art speech recognition
- **OpenAI GPT-4o-mini**: Efficient language model for RAG
- **OpenAI TTS**: High-quality text-to-speech synthesis
- **ChromaDB**: Vector database for semantic search
- **LangChain**: Framework for RAG and LLM integration
- **sounddevice**: Audio recording
- **wavio**: WAV file handling

## 🔍 How RAG Works

1. **Query Processing**: User's transcribed query is fuzzy-matched against stored documents
2. **Embedding Generation**: Query is converted to a vector using OpenAI embeddings
3. **Vector Search**: ChromaDB retrieves top-k most relevant documents
4. **Context Assembly**: Retrieved documents form the context
5. **Answer Generation**: GPT-4o-mini generates an answer based solely on the context
6. **Fallback Handling**: If no relevant context is found, returns a helpful message

## 📝 API Endpoints

### `POST /upload-voice/`

Upload a WAV audio file for processing.

**Request**: Multipart form data with `file` (WAV audio)

**Response**:
```json
{
  "filename": "streamlit_voice.wav",
  "content_type": "audio/wav",
  "sample_rate": 16000,
  "channels": 1,
  "duration_seconds": 10.0,
  "transcription": "Transcribed text here",
  "ai_response": "AI generated answer",
  "tts_audio_path": "path/to/audio.mp3",
  "processing_time_seconds": 5.23
}
```

## 🐛 Troubleshooting

### Audio Recording Issues

- Ensure microphone permissions are granted
- Check that `sounddevice` can access your audio device
- Try adjusting sample rate in `record_voice.py`

### ChromaDB Errors

- Verify the collection name in `.env` matches your ChromaDB collection
- Ensure `chroma_db_openai/` directory exists and contains data
- Check collection has embedded documents

### TTS Generation Fails

- Verify OpenAI API key is valid
- Check API quota/limits
- Ensure sufficient disk space for audio files

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- OpenAI for Whisper, GPT, and TTS APIs
- Streamlit team for the amazing framework
- ChromaDB for vector database capabilities
- LangChain for RAG utilities

---

**Made with ❤️ for voice-enabled AI interactions**

