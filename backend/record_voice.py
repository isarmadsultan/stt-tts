import sounddevice as sd
import wavio
import os

def record_voice(filename, duration, sample_rate):
    """
    Records audio from the default microphone and saves it as a WAV file.
    """
    # Always save in backend/recordings regardless of where called from
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "recordings")
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    print(f"🎙️ Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    print("✅ Recording complete.")

    # Save as a WAV file
    wavio.write(filepath, recording, sample_rate, sampwidth=2)
    print(f"💾 Saved recording to: {filepath}")

    return filepath  # ✅ Return absolute path for Streamlit to use
