import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_tts_audio(text: str, output_filename: str = "tts_output.mp3", voice: str = "alloy") -> str:
    """
    Generate a spoken audio file from the given text using OpenAI TTS.
    Returns the file path to the generated audio.
    """
    if not text or text.strip() == "":
        print("⚠️ Empty text provided for TTS.")
        return None

    try:
        output_path = os.path.join(os.getcwd(), output_filename)
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text
        ) as response:
            response.stream_to_file(output_path)

        print(f"✅ TTS audio generated: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ TTS generation failed: {e}")
        return None
