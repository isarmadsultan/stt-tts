# Agent Response Time Tracking Feature

## Overview
Added a new timing metric called **"time_to_first_audio"** which measures the time from STT (Speech-to-Text) completion to when the first audio chunk starts playing. This represents the **actual agent response time**.

## Changes Made

### 1. Backend: `backend/main.py`

#### AudioPlayer Class Enhancement
- Added `first_audio_callback` parameter to track when first audio starts playing
- Added `first_audio_played` flag to ensure callback is triggered only once
- Added `reset_first_audio_flag()` method for resetting between requests
- Callback is triggered in `_play_worker()` when the first audio starts playing

#### StreamingVoicePipelineOrchestrator Updates
- Added `time_to_first_audio` timing metric tracking
- Timer starts immediately after transcription completes
- Timer ends when the first audio chunk starts playing (via callback)
- Includes fallback logic if callback doesn't trigger
- Updates audio player callback for each request

### 2. Frontend: `streamlit_app.py`

#### UI Display Updates
- Changed from 3-column to 4-column metrics display
- Added **"🎯 Agent Response"** metric showing `time_to_first_audio`
- Added tooltip explaining what Agent Response Time means
- Updated sidebar to explain the key metrics

## How It Works

### Timing Flow:
```
1. User uploads audio
2. STT processes audio → transcription complete
3. ⏱️ START: time_to_first_audio timer starts
4. RAG retrieves context and generates response
5. TTS creates first audio chunk
6. AudioPlayer starts playing first chunk
7. ⏱️ END: time_to_first_audio timer stops (via callback)
```

## Metrics Displayed

| Metric | Description |
|--------|-------------|
| **Transcription** | Time for STT to convert speech to text |
| **🎯 Agent Response** | Time from STT completion to first audio playback (NEW) |
| **RAG + TTS** | Time for complete RAG and TTS processing |
| **Total** | Complete end-to-end time |

## Key Benefits

1. **True Agent Latency**: Measures actual perceived response time
2. **Performance Optimization**: Identify bottlenecks between STT and first audio
3. **User Experience**: Track the most important latency metric for voice agents
4. **Separate from Total Time**: Distinguishes between "time to speak" vs "time to complete"

## Usage

The `time_to_first_audio` metric will automatically appear in:
- API response JSON under `timing.time_to_first_audio`
- Streamlit UI as "🎯 Agent Response" metric
- Detailed timing breakdown in expandable section

## Example Response

```json
{
  "timing": {
    "transcription": 2.5,
    "time_to_first_audio": 3.2,  // NEW: Actual agent response time
    "parallel_rag_tts": 8.5,
    "total_time": 11.0
  }
}
```

This shows that while the complete process took 11 seconds, the agent started speaking after only 3.2 seconds from transcription completion, providing a much better user experience.
