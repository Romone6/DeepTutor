#!/usr/bin/env python
"""
ASR (Automatic Speech Recognition) Service

FastAPI service for transcription using faster-whisper.
Provides /transcribe endpoint for audio-to-text conversion.

Usage:
    python -m services.asr_server.app

Environment Variables:
    ASR_HOST: Bind address (default: 0.0.0.0)
    ASR_PORT: Port number (default: 7001)
    ASR_MODEL: Whisper model size (default: small)
    ASR_DEVICE: Computation device (default: cpu)
    ASR_COMPUTE_TYPE: Precision (default: int8)
    ASR_LANGUAGE: Optional language hint (default: None)
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from faster_whisper import Whisper

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

# Import security middleware
from services.asr_server.auth import (
    AuthConfig,
    RateLimitConfig,
    CORSConfig,
    AuthMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    create_auth_dependency,
    create_rate_limit_dependency,
)


# =============================================================================
# Configuration
# =============================================================================


class ASRConfig:
    """ASR service configuration from environment variables."""

    def __init__(self):
        self.host: str = os.getenv("ASR_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("ASR_PORT", "7001"))
        self.model: str = os.getenv("ASR_MODEL", "small")
        self.device: str = os.getenv("ASR_DEVICE", "cpu")
        self.compute_type: str = os.getenv("ASR_COMPUTE_TYPE", "int8")
        self.language: Optional[str] = os.getenv("ASR_LANGUAGE", None)
        self.beam_size: int = int(os.getenv("ASR_BEAM_SIZE", "5"))
        self.num_workers: int = int(os.getenv("ASR_NUM_WORKERS", "4"))

    def validate(self) -> None:
        """Validate configuration settings."""
        valid_devices = ["cpu", "cuda", "auto"]
        if self.device not in valid_devices:
            raise ValueError(
                f"Invalid ASR_DEVICE: {self.device}. Must be one of {valid_devices}"
            )

        valid_compute_types = ["float16", "int8", "int8_float16"]
        if self.compute_type not in valid_compute_types:
            raise ValueError(
                f"Invalid ASR_COMPUTE_TYPE: {self.compute_type}. Must be one of {valid_compute_types}"
            )

        valid_models = [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
            "distil-large-v3",
            "distil-medium-en",
            "distil-small-en",
        ]
        if self.model not in valid_models:
            raise ValueError(
                f"Invalid ASR_MODEL: {self.model}. Must be one of {valid_models}"
            )


# Global model instance
_asr_model: Optional[Whisper] = None


def get_model() -> Optional[Whisper]:
    """Get or initialize the ASR model."""
    global _asr_model
    return _asr_model


def init_model(config: ASRConfig) -> Whisper:
    """Initialize the faster-whisper model."""
    global _asr_model

    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError(
            "faster-whisper is not installed. Install it with: pip install faster-whisper"
        )

    # Validate device for CUDA
    if config.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but no GPU available")
        except ImportError:
            raise RuntimeError(
                "PyTorch with CUDA support is required for GPU inference. "
                "Install with: pip install torch --index-url https://download.pytorch.org/whl/cu118"
            )

    _asr_model = Whisper(
        model_size_or_path=config.model,
        device=config.device,
        compute_type=config.compute_type,
        num_workers=config.num_workers,
    )

    return _asr_model


# =============================================================================
# Pydantic Models
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    model: Optional[str] = Field(None, description="Loaded model name")
    device: str = Field(..., description="Computation device")
    faster_whisper: bool = Field(..., description="Whether faster-whisper is available")
    timestamp: str = Field(..., description="Server timestamp")


class TranscribeResponse(BaseModel):
    """Transcription response model."""

    text: str = Field(..., description="Transcribed text")
    language: str = Field(..., description="Detected or specified language code")
    duration_seconds: float = Field(..., description="Audio duration")
    segments: list[dict] = Field(default_factory=list, description="Segment details")


class TranscribeRequest(BaseModel):
    """Optional transcription parameters."""

    language: Optional[str] = Field(
        None, description="Language hint (e.g., 'en', 'zh')"
    )
    beam_size: Optional[int] = Field(None, ge=1, le=10, description="Beam search size")
    without_timestamps: bool = Field(default=False, description="Exclude timestamps")


# =============================================================================
# Lifespan Handler
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    global _asr_model

    # Startup
    print("[ASR Server] Starting up...")
    try:
        config = ASRConfig()
        config.validate()

        if FASTER_WHISPER_AVAILABLE:
            print(
                f"[ASR Server] Loading model: {config.model} ({config.device}/{config.compute_type})"
            )
            init_model(config)
            print(f"[ASR Server] Model loaded successfully")
        else:
            print(
                "[ASR Server] WARNING: faster-whisper not installed. Running in stub mode."
            )

        print(f"[ASR Server] Listening on http://{config.host}:{config.port}")
        print("[ASR Server] Endpoints:")
        print("  POST /transcribe - Transcribe audio file")
        print("  GET  /health     - Health check")

    except Exception as e:
        print(f"[ASR Server] Startup error: {e}")
        raise

    yield

    # Shutdown
    print("[ASR Server] Shutting down...")
    _asr_model = None


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="ASR Service",
    description="Automatic Speech Recognition service using faster-whisper",
    version="1.0.0",
    lifespan=lifespan,
)

# Load security configuration
auth_config = AuthConfig.from_env()
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
auth_dependency = create_auth_dependency(auth_config)


# =============================================================================
# Error Handlers
# =============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": " transcription_error"
            if exc.status_code == 400
            else "service_error",
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
            "message": "An unexpected error occurred during transcription",
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

    Returns service status, loaded model info, and component health.
    """
    config = ASRConfig()
    model = get_model()

    return HealthResponse(
        status="healthy" if FASTER_WHISPER_AVAILABLE else "degraded",
        model=config.model if model else None,
        device=config.device,
        faster_whisper=FASTER_WHISPER_AVAILABLE,
        timestamp=__import__("datetime").datetime.now().isoformat(),
    )


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = None,
    beam_size: Optional[int] = None,
    without_timestamps: bool = False,
) -> TranscribeResponse:
    """
    Transcribe an audio file to text.

    Args:
        audio: Audio file upload (WAV, MP3, OGG, FLAC, M4A, WebM)
        language: Optional language hint (e.g., 'en', 'zh', 'ja')
        beam_size: Beam search size (1-10, default: 5)
        without_timestamps: Exclude timestamp data from response

    Returns:
        TranscribeResponse with text, language, and optional segments

    Raises:
        HTTPException: 400 for invalid audio, 422 for validation error
    """
    # Check if faster-whisper is available
    if not FASTER_WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ASR service is not properly configured. faster-whisper is not installed.",
        )

    # Validate file was uploaded
    if not audio or not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file provided",
        )

    # Validate content type
    allowed_content_types = {
        "audio/wav",
        "audio/x-wav",
        "audio/mp3",
        "audio/mpeg",
        "audio/ogg",
        "audio/flac",
        "audio/m4a",
        "audio/webm",
        "audio/mp4",
        "audio/aac",
        "audio/opus",
        "audio/quicktime",
        "application/octet-stream",  # Some clients send binary without proper MIME
    }

    content_type = audio.content_type or "application/octet-stream"

    if content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format: {content_type}. "
            f"Supported formats: WAV, MP3, OGG, FLAC, M4A, WebM",
        )

    # Read audio content
    try:
        audio_content = await audio.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read audio file: {e}",
        )

    # Validate file size (max 300MB for transcription)
    max_size_bytes = 300 * 1024 * 1024  # 300 MB
    if len(audio_content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file provided",
        )

    if len(audio_content) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio file too large ({len(audio_content) / (1024 * 1024):.1f}MB). "
            f"Maximum allowed: 300MB",
        )

    # Get model and run transcription
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ASR model not loaded",
        )

    # Run transcription in thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    try:
        # Prepare transcription parameters
        config = ASRConfig()
        transcribe_options = {
            "language": language or config.language,
            "beam_size": beam_size or config.beam_size,
            "without_timestamps": without_timestamps,
        }

        # Run transcription
        segments, info = await loop.run_in_executor(
            None, lambda: model.transcribe(audio_content, **transcribe_options)
        )

        # Collect results
        segments_list = []
        text_parts = []

        async for segment in segments:
            seg_dict = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            }
            if not without_timestamps:
                seg_dict["words"] = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                    for w in (segment.words or [])
                ]
            segments_list.append(seg_dict)
            text_parts.append(seg_dict["text"])

        full_text = " ".join(text_parts).strip()

        return TranscribeResponse(
            text=full_text,
            language=info.language or language or "unknown",
            duration_seconds=info.duration or 0.0,
            segments=segments_list,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {e}",
        )


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "ASR Service",
        "version": "1.0.0",
        "endpoints": {
            "transcribe": "POST /transcribe",
            "health": "GET /health",
        },
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    config = ASRConfig()
    config.validate()

    print(f"[ASR Server] Configuration:")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Model: {config.model}")
    print(f"  Device: {config.device}")
    print(f"  Compute Type: {config.compute_type}")
    print()

    uvicorn.run(
        "services.asr_server.app:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info",
    )
