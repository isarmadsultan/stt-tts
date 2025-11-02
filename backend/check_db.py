import os
from openai import OpenAI
from dotenv import load_dotenv  
load_dotenv()
# Make sure your key is set in environment
# e.g. export OPENAI_API_KEY="sk-..."
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def test_tts():
    text = "Hello Sarmad! This is a test of OpenAI's text to speech voice. Let's verify how clear it sounds."
    output_path = "test_tts_output.mp3"

    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",  # you can try 'verse', 'sage', or 'nova' too
            input=text
        ) as response:
            response.stream_to_file(output_path)

        print(f"✅ TTS audio saved at: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error generating TTS: {e}")
        return None

if __name__ == "__main__":
    path = test_tts()
    if path and os.path.exists(path):
        print(f"Now play it with any audio player or run:\n\n  open {path}\n")
