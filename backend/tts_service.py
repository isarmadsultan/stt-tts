import os
from openai import OpenAI


class TTSService:
    """
    Text-to-Speech service using OpenAI GPT-4o-mini-tts.
    Handles audio generation and saving output to a file.
    """

    def __init__(self, api_key: str = None, default_voice: str = "alloy"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not set.")

        self.client = OpenAI(api_key=self.api_key)
        self.default_voice = default_voice

    def generate(self, text: str, output_filename: str = "tts_output.mp3", voice: str = None) -> str:
        """
        Generate a spoken MP3 audio file from text.
        Returns path to generated file.
        """

        if not text or text.strip() == "":
            print("⚠️ Empty text provided for TTS.")
            return None

        voice = voice or self.default_voice
        output_path = os.path.join(os.getcwd(), output_filename)

        try:
            with self.client.audio.speech.with_streaming_response.create(
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
