from pydantic import BaseModel, Field
from typing import Literal

class VoiceInputConfig(BaseModel):
    """
    Configuration for voice input processing.
    """
    sample_rate: int = Field(
        16000,
        description="Sampling rate in Hz (number of audio samples per second)."
    )
    format: Literal["wav"] = Field(
        "wav",
        description="Audio format. Only 'wav' is supported."
    )

    
