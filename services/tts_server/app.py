#!/usr/bin/env python
"""
TTS (Text-to-Speech) Service

FastAPI service for speech synthesis using VibeVoice or stub engine.
Provides /speak endpoint for text-to-audio conversion.

Usage:
    python -m services.tts_server.app

Environment Variables:
    TTS_HOST: Bind address (default: 0.0.0.0)
    TTS_PORT: Port number (default: 7002)
    TTS_ENGINE: Engine to use (default: stub, options: stub, vibevoice)
    TTS_VOICE: Default voice ID (default: stub-male-1)
    TTS_FORMAT: Default audio format (default: wav)
    TTS_SAMPLE_RATE: Sample rate in Hz (default: 24000)

VibeVoice-specific:
    VIBEVOICE_MODEL_PATH: Path to VibeVoice model
    VIBEVOICE_DEVICE: Device (cpu, cuda)
    VIBEVOICE_COMPUTE_TYPE: Precision (float16, int8)
"""

import os
import sys
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import engines
from services.tts_server.engine_stub import StubTTSEngine, StubConfig
from services.tts_server.engine_vibevoice import (
    VibeVoiceEngine,
    VibeVoiceConfig,
    VIBEVOICE_AVAILABLE,
)

# Import security middleware (shared via symlink)
from services.tts_server.auth import (
    AuthConfig,
    RateLimitConfig,
    CORSConfig,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    create_auth_dependency,
)


# =============================================================================
# Configuration
# =============================================================================


class AudioFormat(str, Enum):
    """Supported audio output formats."""

    WAV = "wav"
    MP3 = "mp3"


class TTSConfig:
    """TTS service configuration from environment variables."""

    def __init__(self):
        self.host: str = os.getenv("TTS_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("TTS_PORT", "7002"))
        self.engine: str = os.getenv("TTS_ENGINE", "stub")
        self.voice: str = os.getenv("TTS_VOICE", "stub-male-1")
        self.format: str = os.getenv("TTS_FORMAT", "wav")
        self.sample_rate: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))

    def validate(self) -> None:
        """Validate configuration settings."""
        valid_engines = ["stub", "vibevoice"]
        if self.engine not in valid_engines:
            raise ValueError(f"Invalid TTS_ENGINE: {self.engine}. Must be one of {valid_engines}")

        if self.engine == "vibevoice" and not VIBEVOICE_AVAILABLE:
            raise RuntimeError(
                "VibeVoice requested but not installed. Set TTS_ENGINE=stub or install VibeVoice."
            )

        valid_formats = ["wav", "mp3"]
        if self.format not in valid_formats:
            raise ValueError(f"Invalid TTS_FORMAT: {self.format}. Must be one of {valid_formats}")


# =============================================================================
# Engine Factory
# =============================================================================


def get_engine():
    """Get or create the TTS engine based on configuration."""
    global _engine
    return _engine


def create_engine(config: TTSConfig) -> object:
    """Create TTS engine based on configuration."""
    if config.engine == "vibevoice" and VIBEVOICE_AVAILABLE:
        vibevoice_config = VibeVoiceConfig(
            model_path=os.getenv("VIBEVOICE_MODEL_PATH", ""),
            device=os.getenv("VIBEVOICE_DEVICE", "cpu"),
            compute_type=os.getenv("VIBEVOICE_COMPUTE_TYPE", "int8"),
            sample_rate=config.sample_rate,
            voice=config.voice,
        )
        engine = VibeVoiceEngine(vibevoice_config)
        engine.load()
        return engine
    else:
        # Use stub engine
        stub_config = StubConfig(sample_rate=config.sample_rate)
        return StubTTSEngine(stub_config)


# Global engine instance
_engine = None


# =============================================================================
# Pydantic Models
# =============================================================================


class SpeakRequest(BaseModel):
    """Request model for speech synthesis."""

    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice_id: Optional[str] = Field(None, description="Voice identifier")
    format: AudioFormat = Field(default=AudioFormat.WAV, description="Output audio format")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    sample_rate: Optional[int] = Field(None, ge=8000, le=48000, description="Sample rate in Hz")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    engine: str = Field(..., description="Engine type")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    voices_available: int = Field(..., description="Number of available voices")
    timestamp: str = Field(..., description="Server timestamp")


class VoicesResponse(BaseModel):
    """Response model for available voices."""

    voices: list[dict] = Field(..., description="List of available voices")


# =============================================================================
# Lifespan Handler
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    global _engine

    print("[TTS Server] Starting up...")
    try:
        config = TTSConfig()
        config.validate()

        print(f"[TTS Server] Initializing engine: {config.engine}")
        _engine = create_engine(config)

        if hasattr(_engine, "load") and not hasattr(_engine, "is_loaded"):
            _engine.load()

        voices = _engine.get_available_voices()
        print(f"[TTS Server] Engine ready with {len(voices)} voices")
        print(f"[TTS Server] Listening on http://{config.host}:{config.port}")
        print("[TTS Server] Endpoints:")
        print("  POST /speak - Synthesize speech from text")
        print("  GET  /health - Health check")
        print("  GET  /voices - List available voices")

    except Exception as e:
        print(f"[TTS Server] Startup error: {e}")
        raise

    yield

    # Shutdown
    print("[TTS Server] Shutting down...")
    if _engine is not None and hasattr(_engine, "unload"):
        try:
            _engine.unload()
        except Exception:
            pass
    _engine = None


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="TTS Service",
    description="Text-to-Speech service using VibeVoice",
    version="1.0.0",
    lifespan=lifespan,
)

# Load security configuration
cors_config = CORSConfig.from_env()
rate_limit_config = RateLimitConfig.from_env()

# Security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware with configurable origins
if cors_config.enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Rate limiting middleware
if rate_limit_config.enabled:
    rate_limit_middleware = RateLimitMiddleware(rate_limit_config)
    app.add_middleware(lambda app: rate_limit_middleware)

# Create auth dependency for protected endpoints
auth_dependency = create_auth_dependency(AuthConfig.from_env())


# =============================================================================
# Error Handlers
# =============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "synthesis_error" if exc.status_code == 400 else "service_error",
            "message": str(exc.detail),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred during synthesis",
            "detail": str(exc) if os.getenv("DEBUG") else None,
        },
    )


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service status, engine info, and voice count.
    """
    global _engine

    if _engine is None:
        return HealthResponse(
            status="unhealthy",
            engine="none",
            model_loaded=False,
            voices_available=0,
            timestamp=__import__("datetime").datetime.now().isoformat(),
        )

    health = _engine.health_check()
    voices = _engine.get_available_voices()

    return HealthResponse(
        status=health.get("status", "healthy"),
        engine=health.get("engine", "unknown"),
        model_loaded=health.get(
            "model_loaded", _engine.is_loaded if hasattr(_engine, "is_loaded") else True
        ),
        voices_available=len(voices),
        timestamp=__import__("datetime").datetime.now().isoformat(),
    )


@app.get("/voices", response_model=VoicesResponse)
async def list_voices() -> VoicesResponse:
    """
    List available voices.

    Returns list of available voice IDs and their metadata.
    """
    global _engine

    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS engine not loaded",
        )

    voices = _engine.get_available_voices()

    return VoicesResponse(voices=voices)


@app.post("/speak")
async def speak(request: SpeakRequest) -> Response:
    """
    Synthesize speech from text.

    Args:
        request: SpeakRequest with text and synthesis parameters

    Returns:
        Audio file as binary response with appropriate Content-Type

    Raises:
        HTTPException: 400 for invalid input, 503 if engine not available
    """
    global _engine

    # Check if engine is available
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS service is not properly configured. Engine not loaded.",
        )

    # Validate text
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty",
        )

    if len(request.text) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text too long (max 5000 characters)",
        )

    try:
        # Synthesize speech
        audio_bytes, content_type = _engine.synthesize(
            text=request.text,
            voice_id=request.voice_id,
            format=request.format.value,
            sample_rate=request.sample_rate,
            speed=request.speed,
        )

        # Calculate duration for headers
        duration = _engine.get_audio_duration(audio_bytes)

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Length": str(len(audio_bytes)),
                "X-Duration-Seconds": str(duration),
                "X-Sample-Rate": str(request.sample_rate or 24000),
                "X-Format": request.format.value,
            },
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {e}",
        )


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "TTS Service",
        "version": "1.0.0",
        "engine": "vibevoice" if VIBEVOICE_AVAILABLE else "stub",
        "endpoints": {
            "speak": "POST /speak",
            "health": "GET /health",
            "voices": "GET /voices",
        },
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    config = TTSConfig()
    config.validate()

    print(f"[TTS Server] Configuration:")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Engine: {config.engine}")
    print(f"  Format: {config.format}")
    print(f"  Sample Rate: {config.sample_rate}")
    print()

    uvicorn.run(
        "services.tts_server.app:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info",
    )
