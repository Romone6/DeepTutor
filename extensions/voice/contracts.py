from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class AudioFormat(str, Enum):
    """Supported audio output formats."""

    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"


class TranscribeRequest(BaseModel):
    """Request model for audio transcription."""

    language: Optional[str] = Field(
        None,
        description="Optional language code (e.g., 'en', 'zh', 'ja'). Auto-detected if not provided.",
    )
    enable_speakers: bool = Field(
        default=False,
        description="Return speaker diarization segments if True.",
    )
    model: Optional[str] = Field(
        None,
        description="Optional ASR model override (e.g., 'whisper', 'paraformer').",
    )


class TranscribeResponse(BaseModel):
    """Response model for audio transcription."""

    text: str = Field(..., description="Full transcribed text.")
    language: str = Field(..., description="Detected or specified language code.")
    duration_seconds: float = Field(..., description="Audio duration in seconds.")
    segments: Optional[list[dict]] = Field(
        default=None,
        description="Optional word-level or segment-level timing data.",
    )


class SpeakRequest(BaseModel):
    """Request model for text-to-speech synthesis."""

    text: str = Field(..., min_length=1, description="Text to synthesize.")
    voice_id: Optional[str] = Field(
        default=None,
        description="Voice identifier (e.g., 'en-US-female-1', 'zh-CN-male-2').",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (0.5x to 2.0x).",
    )
    format: AudioFormat = Field(
        default=AudioFormat.MP3,
        description="Output audio format.",
    )
    sample_rate: Optional[int] = Field(
        default=None,
        ge=8000,
        le=48000,
        description="Optional sample rate override (8kHz to 48kHz).",
    )
    emotion: Optional[str] = Field(
        default=None,
        description="Optional emotion/style tag (e.g., 'cheerful', 'calm').",
    )


class SpeakResponse(BaseModel):
    """Response model for text-to-speech synthesis."""

    audio_content: bytes = Field(..., description="Raw audio bytes.")
    format: AudioFormat = Field(..., description="Audio format of the response.")
    duration_seconds: float = Field(..., description="Synthesized audio duration.")
    sample_rate: int = Field(..., description="Sample rate of the audio.")


class HealthStatus(str, Enum):
    """Health check component status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of an individual component."""

    name: str = Field(..., description="Component name.")
    status: HealthStatus = Field(..., description="Component health status.")
    latency_ms: Optional[int] = Field(
        default=None,
        description="Optional latency in milliseconds.",
    )
    message: Optional[str] = Field(
        default=None,
        description="Optional status message or error details.",
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: HealthStatus = Field(..., description="Overall system health status.")
    version: str = Field(..., description="Voice gateway version.")
    components: list[ComponentHealth] = Field(
        ...,
        description="Individual component health details.",
    )
    uptime_seconds: int = Field(..., description="Service uptime in seconds.")


class VoiceGatewayError(BaseModel):
    """Standard error response for voice gateway."""

    error: str = Field(..., description="Error type or code.")
    message: str = Field(..., description="Human-readable error message.")
    details: Optional[dict] = Field(
        default=None,
        description="Additional error details.",
    )
