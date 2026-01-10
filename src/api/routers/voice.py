"""
Voice Chat API Router
=====================

REST endpoint for voice-based chat with DeepTutor.

Flow:
1. Receive audio file upload
2. Transcribe audio using ASR service (extensions/voice/client)
3. Process transcript through ChatAgent
4. Return transcript and tutor response

Environment Variables:
- VOICE_ASR_URL: ASR service URL (default: http://localhost:7001)
- VOICE_TOKEN: Optional Bearer token for ASR service
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.agents.chat import ChatAgent
from src.logging import get_logger
from src.services.config import load_config_with_main
from extensions.voice.client import (
    VoiceGatewayClient,
    VoiceGatewayError,
    VoiceGatewayValidationError,
    VoiceGatewayTimeoutError,
    VoiceGatewayConnectionError,
)
from extensions.voice.cache import get_cache


# Initialize logger
project_root = Path(__file__).parent.parent.parent.parent
config = load_config_with_main("solve_config.yaml", project_root)
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
logger = get_logger("VoiceChatAPI", level="INFO", log_dir=log_dir)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================


class VoiceChatResponse(BaseModel):
    """Response model for voice chat."""

    transcript: str = Field(..., description="Transcribed audio text")
    reply: str = Field(..., description="Tutor's response")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    sources: dict = Field(default_factory=dict, description="Source citations")
    language: str = Field(..., description="Detected language")


class VoiceChatRequest(BaseModel):
    """Request model for voice chat (alternative to multipart)."""

    kb_name: Optional[str] = Field(None, description="Knowledge base name")
    enable_rag: bool = Field(default=False, description="Enable RAG retrieval")
    enable_web_search: bool = Field(default=False, description="Enable web search")
    language: Optional[str] = Field(None, description="Language hint for ASR")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")


# =============================================================================
# Helper Functions
# =============================================================================


def create_voice_client() -> VoiceGatewayClient:
    """Create and configure voice gateway client."""
    try:
        return VoiceGatewayClient()
    except ImportError as e:
        logger.error(f"Failed to create voice client: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice services not available. Install httpx.",
        )


def transcribe_audio(audio_bytes: bytes, filename: str, language: Optional[str] = None) -> str:
    """
    Transcribe audio using voice gateway client.

    Args:
        audio_bytes: Raw audio file bytes
        filename: Original filename (for format detection)
        language: Optional language hint

    Returns:
        Transcribed text string

    Raises:
        HTTPException: On transcription failure
    """
    client = create_voice_client()

    try:
        transcript = client.transcribe_audio(audio_bytes, filename, language=language)
        return transcript
    except VoiceGatewayValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except VoiceGatewayTimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="ASR service timed out. Please try again.",
        )
    except VoiceGatewayConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ASR service is unreachable. Please check configuration.",
        )
    except VoiceGatewayError as e:
        logger.error(f"ASR error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to transcribe audio. Please try again.",
        )


async def process_chat(
    message: str,
    kb_name: Optional[str] = None,
    enable_rag: bool = False,
    enable_web_search: bool = False,
) -> dict:
    """
    Process chat message through ChatAgent.

    Args:
        message: User message
        kb_name: Knowledge base name for RAG
        enable_rag: Enable RAG retrieval
        enable_web_search: Enable web search

    Returns:
        Dict with 'response', 'sources'
    """
    language = config.get("system", {}).get("language", "en")
    agent = ChatAgent(language=language, config=config)

    result = await agent.process(
        message=message,
        history=[],
        kb_name=kb_name,
        enable_rag=enable_rag,
        enable_web_search=enable_web_search,
        stream=False,
    )

    return {
        "response": result.get("response", ""),
        "sources": result.get("sources", {"rag": [], "web": []}),
    }


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/voice/chat",
    response_model=VoiceChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        502: {"model": ErrorResponse, "description": "ASR service error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Voice Chat",
    description="Upload audio, transcribe, and get tutor response in one call",
)
async def voice_chat(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC, M4A, WebM)"),
    kb_name: Optional[str] = Form(None, description="Knowledge base name"),
    enable_rag: bool = Form(False, description="Enable RAG retrieval"),
    enable_web_search: bool = Form(False, description="Enable web search"),
    language: Optional[str] = Form(None, description="Language hint for ASR"),
) -> VoiceChatResponse:
    """
    Process voice chat request.

    Flow:
    1. Receive audio file
    2. Transcribe using ASR service
    3. Process transcript through ChatAgent
    4. Return transcript and tutor response

    Args:
        audio: Audio file upload
        kb_name: Knowledge base for RAG
        enable_rag: Enable RAG retrieval
        enable_web_search: Enable web search
        language: Language hint for ASR

    Returns:
        VoiceChatResponse with transcript, reply, and metadata
    """
    logger.info(f"Voice chat request: {audio.filename}, {audio.content_type}")

    # Validate audio file
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
        "audio/aac",
        "audio/opus",
    }

    content_type = audio.content_type or "application/octet-stream"
    if content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format: {content_type}. "
            f"Supported: WAV, MP3, OGG, FLAC, M4A, WebM",
        )

    # Read audio content
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read audio file: {e}",
        )

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file",
        )

    # Check file size (max 50MB for voice chat)
    max_size = 50 * 1024 * 1024
    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio file too large ({len(audio_bytes) / (1024 * 1024):.1f}MB). "
            f"Maximum is 50MB",
        )

    # Step 1: Transcribe audio
    logger.info(f"Transcribing audio: {len(audio_bytes)} bytes")
    try:
        transcript = transcribe_audio(audio_bytes, audio.filename, language=language)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to transcribe audio",
        )

    if not transcript or not transcript.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not transcribe audio. Please try again with clearer audio.",
        )

    logger.info(f"Transcript: {transcript[:100]}...")

    # Step 2: Process through ChatAgent
    logger.info("Processing chat through ChatAgent")
    try:
        chat_result = await process_chat(
            message=transcript,
            kb_name=kb_name,
            enable_rag=enable_rag,
            enable_web_search=enable_web_search,
        )
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate tutor response",
        )

    # Return combined response
    return VoiceChatResponse(
        transcript=transcript,
        reply=chat_result["response"],
        session_id=None,  # Could add session support
        sources=chat_result["sources"],
        language=language or "unknown",
    )


@router.post(
    "/voice/chat/json",
    response_model=VoiceChatResponse,
    responses={
        400: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
    summary="Voice Chat (JSON body)",
    description="Same as /voice/chat but accepts base64-encoded audio in JSON",
)
async def voice_chat_json(
    audio_data: str = Field(..., description="Base64-encoded audio"),
    filename: str = Field(default="audio.wav", description="Original filename"),
    kb_name: Optional[str] = None,
    enable_rag: bool = False,
    enable_web_search: bool = False,
    language: Optional[str] = None,
) -> VoiceChatResponse:
    """
    Process voice chat with base64-encoded audio in JSON body.

    Useful for clients that prefer JSON over multipart form data.
    """
    import base64

    # Decode base64 audio
    try:
        audio_bytes = base64.b64decode(audio_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 audio data: {e}",
        )

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio data",
        )

    # Step 1: Transcribe
    logger.info(f"Transcribing base64 audio: {len(audio_bytes)} bytes")
    try:
        transcript = transcribe_audio(audio_bytes, filename, language=language)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to transcribe audio",
        )

    if not transcript or not transcript.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not transcribe audio",
        )

    # Step 2: Process through ChatAgent
    try:
        chat_result = await process_chat(
            message=transcript,
            kb_name=kb_name,
            enable_rag=enable_rag,
            enable_web_search=enable_web_search,
        )
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate tutor response",
        )

    return VoiceChatResponse(
        transcript=transcript,
        reply=chat_result["response"],
        session_id=None,
        sources=chat_result["sources"],
        language=language or "unknown",
    )


@router.get("/voice/health", summary="Voice Service Health")
async def voice_health():
    """
    Check health of voice services.

    Returns:
        Health status of ASR service
    """
    try:
        client = create_voice_client()
        asr_health = client.health_check("asr")
        return {
            "status": "healthy",
            "asr": asr_health,
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
            },
        )


# =============================================================================
# Session-Aware Voice Chat (Optional Enhancement)
# =============================================================================


@router.post(
    "/voice/chat/session",
    response_model=VoiceChatResponse,
    summary="Voice Chat with Session",
    description="Voice chat with session support for multi-turn conversations",
)
async def voice_chat_with_session(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None, description="Existing session ID"),
    kb_name: Optional[str] = Form(None),
    enable_rag: bool = Form(False),
    enable_web_search: bool = Form(False),
    language: Optional[str] = Form(None),
) -> VoiceChatResponse:
    """
    Voice chat with session support.

    This is an enhanced version that maintains conversation history
    across multiple voice interactions.
    """
    # Import session manager
    from src.agents.chat import SessionManager

    session_manager = SessionManager()

    # Read and validate audio
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Transcribe
    transcript = transcribe_audio(audio_bytes, audio.filename, language=language)

    if not transcript or not transcript.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe audio")

    # Get or create session
    if session_id:
        session = session_manager.get_session(session_id)
        if not session:
            # Create new session if old one not found
            session = session_manager.create_session(
                title=transcript[:50],
                settings={"kb_name": kb_name, "enable_rag": enable_rag},
            )
            session_id = session["session_id"]
    else:
        session = session_manager.create_session(
            title=transcript[:50],
            settings={"kb_name": kb_name, "enable_rag": enable_rag},
        )
        session_id = session["session_id"]

    # Get history from session
    history = [
        {"role": msg["role"], "content": msg["content"]} for msg in session.get("messages", [])
    ]

    # Add user message
    session_manager.add_message(session_id, "user", transcript)

    # Process through ChatAgent with history
    language_setting = config.get("system", {}).get("language", "en")
    agent = ChatAgent(language=language_setting, config=config)

    result = await agent.process(
        message=transcript,
        history=history,
        kb_name=kb_name,
        enable_rag=enable_rag,
        enable_web_search=enable_web_search,
        stream=False,
    )

    # Add assistant response to session
    session_manager.add_message(
        session_id,
        "assistant",
        result["response"],
        sources=result.get("sources"),
    )

    return VoiceChatResponse(
        transcript=transcript,
        reply=result["response"],
        session_id=session_id,
        sources=result.get("sources", {"rag": [], "web": []}),
        language=language or "unknown",
    )


# =============================================================================
# TTS Endpoints with Caching
# =============================================================================


class SpeakRequest(BaseModel):
    """Request model for TTS synthesis."""

    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice_id: Optional[str] = Field(None, description="Voice identifier")
    format: str = Field(default="wav", description="Audio format (wav, mp3)")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")


class SpeakResponse(BaseModel):
    """Response model for TTS synthesis."""

    text: str = Field(..., description="Original text")
    voice_id: Optional[str] = Field(None, description="Voice used")
    format: str = Field(..., description="Audio format")
    duration_seconds: float = Field(..., description="Audio duration")
    cached: bool = Field(default=False, description="Whether response was served from cache")
    cache_stats: Optional[dict] = Field(None, description="Cache statistics")


@router.post(
    "/voice/speak",
    summary="Text-to-Speech",
    description="Convert text to spoken audio with optional caching",
    responses={
        200: {
            "content": {
                "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                "audio/mpeg": {"schema": {"type": "string", "format": "binary"}},
            },
            "description": "Audio file",
        },
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "TTS service unavailable"},
    },
)
async def speak(
    request: SpeakRequest,
) -> JSONResponse:
    """
    Convert text to speech audio.

    Features:
    - Automatic caching for repeated requests
    - Configurable voice, format, and speed
    - Cache statistics in response headers

    Args:
        request: SpeakRequest with text and synthesis parameters

    Returns:
        Audio file as binary response with appropriate Content-Type
    """
    logger.info(
        f"Speak request: {len(request.text)} chars, voice={request.voice_id}, format={request.format}"
    )

    # Validate input
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty",
        )

    if len(text) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text too long (max 5000 characters)",
        )

    # Check cache first
    cache = get_cache()
    cached_result = cache.get(text, request.voice_id, request.format)
    cache_stats = cache.stats()

    if cached_result is not None:
        audio_bytes, content_type = cached_result
        logger.info(f"Cache hit for text: {text[:50]}...")
        cached = True
    else:
        # Cache miss - call TTS service
        logger.info(f"Cache miss, calling TTS service")
        client = create_voice_client()

        try:
            audio_bytes, content_type = client.synthesize_speech(
                text=text,
                voice_id=request.voice_id,
                format=request.format,
                speed=request.speed,
            )

            # Store in cache
            cache.set(
                text, request.voice_id or "default", request.format, audio_bytes, content_type
            )
            cached = False
            cache_stats = cache.stats()

        except VoiceGatewayValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except VoiceGatewayTimeoutError as e:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="TTS service timed out. Please try again.",
            )
        except VoiceGatewayConnectionError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="TTS service is unreachable. Please check configuration.",
            )
        except VoiceGatewayError as e:
            logger.error(f"TTS error: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to synthesize speech. Please try again.",
            )

    # Calculate duration
    duration = (
        client.get_audio_duration(audio_bytes) if hasattr(client, "get_audio_duration") else 0.0
    )

    # Return audio response
    return JSONResponse(
        content=b"",
        media_type=content_type,
        headers={
            "Content-Length": str(len(audio_bytes)),
            "Content-Disposition": f'inline; filename="speech.{request.format}"',
            "X-Duration-Seconds": str(duration),
            "X-Cached": "true" if cached else "false",
            "X-Cache-Hit-Rate": f"{cache_stats.get('hit_rate_percent', 0)}%",
        },
    )


@router.get("/voice/speak/cache/stats", summary="Cache Statistics")
async def cache_stats():
    """
    Get TTS cache statistics.

    Returns:
        Cache metrics including hit rate, size, and memory usage
    """
    cache = get_cache()
    return cache.stats()


@router.delete("/voice/speak/cache", summary="Clear Cache")
async def clear_cache():
    """
    Clear the TTS audio cache.

    Returns:
        Success message
    """
    clear_cache()
    return {"status": "cleared", "message": "TTS cache cleared"}


@router.post(
    "/voice/speak/stream",
    summary="Text-to-Speech (Streaming)",
    description="Stream audio directly without JSON wrapper",
)
async def speak_stream(
    request: SpeakRequest,
):
    """
    Convert text to speech and stream directly.

    Same as /voice/speak but returns raw audio bytes without JSON wrapper.
    Useful for direct audio playback.
    """
    # Validate and get audio (same logic as /voice/speak)
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    cache = get_cache()
    cached_result = cache.get(text, request.voice_id, request.format)

    if cached_result is not None:
        audio_bytes, content_type = cached_result
    else:
        client = create_voice_client()
        try:
            audio_bytes, content_type = client.synthesize_speech(
                text=text,
                voice_id=request.voice_id,
                format=request.format,
                speed=request.speed,
            )
            cache.set(
                text, request.voice_id or "default", request.format, audio_bytes, content_type
            )
        except VoiceGatewayError as e:
            raise HTTPException(status_code=502, detail=str(e))

    # Return streaming response
    from fastapi.responses import StreamingResponse
    import io

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type=content_type,
        headers={
            "Content-Length": str(len(audio_bytes)),
            "Content-Disposition": f'inline; filename="speech.{request.format}"',
        },
    )


class StreamChunkRequest(BaseModel):
    """Request model for streaming TTS."""

    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    voice_id: Optional[str] = Field(None, description="Voice identifier")
    format: str = Field(default="wav", description="Audio format (wav, mp3)")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    chunk_size: int = Field(default=200, ge=50, le=500, description="Characters per chunk")


def split_text_into_sentences(text: str, max_chars: int = 200) -> list[str]:
    """
    Split text into sentence-like chunks for streaming synthesis.

    Args:
        text: Input text
        max_chars: Maximum characters per chunk

    Returns:
        List of text chunks suitable for streaming TTS
    """
    import re

    if len(text) <= max_chars:
        return [text] if text.strip() else []

    chunks = []
    current_chunk = []

    sentence_endings = re.compile(r"[.!?\n]+")
    words = text.replace("\n", " ").split(" ")

    for word in words:
        if not word.strip():
            continue

        test_chunk = " ".join(current_chunk + [word])

        if len(test_chunk) > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    if not chunks and text.strip():
        chunks = [text.strip()[:max_chars]]

    return chunks


@router.post(
    "/voice/speak/stream/chunked",
    summary="Text-to-Speech (Chunked Streaming)",
    description="Stream audio in chunks for reduced perceived latency",
)
async def speak_stream_chunked(
    request: StreamChunkRequest,
):
    """
    Convert text to speech with chunked streaming.

    For long text, audio begins playing within 1-2 seconds after request starts.
    Each chunk is synthesized and streamed immediately.

    Args:
        request: StreamChunkRequest with text and synthesis parameters

    Returns:
        Streaming response with audio chunks separated by newlines
    """
    import asyncio
    import io
    import re
    import wave

    logger.info(
        f"Chunked stream request: {len(request.text)} chars, chunk_size={request.chunk_size}"
    )

    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty",
        )

    if len(text) > 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text too long (max 10000 characters for streaming)",
        )

    def generate_audio_chunks():
        """Generator that yields audio chunks with WAV headers."""
        client = create_voice_client()

        try:
            chunks = split_text_into_sentences(text, request.chunk_size)

            for i, chunk_text in enumerate(chunks):
                try:
                    if request.format == "wav":
                        audio_bytes, _ = client.synthesize_speech(
                            text=chunk_text,
                            voice_id=request.voice_id,
                            format="wav",
                            speed=request.speed,
                        )
                    else:
                        audio_bytes, _ = client.synthesize_speech(
                            text=chunk_text,
                            voice_id=request.voice_id,
                            format="mp3",
                            speed=request.speed,
                        )

                    chunk_header = f"CHUNK:{i}:{len(audio_bytes)}\n"
                    yield chunk_header.encode("utf-8") + audio_bytes + b"\n"

                except Exception as e:
                    logger.warning(f"Failed to synthesize chunk {i}: {e}")
                    continue

            yield b"END:streaming_complete\n"

        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            error_msg = f"ERROR:{str(e)}\n"
            yield error_msg.encode("utf-8")

    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        generate_audio_chunks(),
        media_type="application/octet-stream",
        headers={
            "Transfer-Encoding": "chunked",
            "X-Streaming-Mode": "chunked-audio",
            "X-Chunk-Count": str(len(split_text_into_sentences(text, request.chunk_size))),
            "Cache-Control": "no-cache",
        },
    )


@router.post(
    "/voice/speak/stream/chunked/simple",
    summary="Text-to-Speech (Simple Chunked)",
    description="Simpler chunked streaming without custom protocol",
)
async def speak_stream_chunked_simple(
    request: StreamChunkRequest,
):
    """
    Simple chunked streaming without custom protocol.

    Returns raw WAV chunks that can be concatenated. First chunk
    is sent quickly to start playback early.

    Returns:
        Streaming response with raw audio chunks
    """
    import asyncio
    import io
    import wave

    logger.info(f"Simple chunked stream: {len(request.text)} chars")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    def generate_simple_chunks():
        """Generator that yields raw WAV chunks."""
        client = create_voice_client()
        chunks = split_text_into_sentences(text, request.chunk_size)

        first_chunk_sent = False
        for chunk_text in chunks:
            try:
                audio_bytes, _ = client.synthesize_speech(
                    text=chunk_text,
                    voice_id=request.voice_id,
                    format="wav",
                    speed=request.speed,
                )

                yield audio_bytes

                if not first_chunk_sent:
                    first_chunk_sent = True
                    logger.info(f"First chunk sent: {len(audio_bytes)} bytes")

            except Exception as e:
                logger.warning(f"Failed to synthesize chunk: {e}")
                continue

    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        generate_simple_chunks(),
        media_type="audio/wav",
        headers={
            "Transfer-Encoding": "chunked",
            "X-Streaming-Mode": "simple-chunks",
            "Cache-Control": "no-cache",
        },
    )


@router.get("/voice/speak/stream/capabilities", summary="Check Streaming Capabilities")
async def check_streaming_capabilities():
    """
    Check what streaming capabilities the server and TTS service support.

    Returns:
        Dict with streaming support information
    """
    client = create_voice_client()

    try:
        tts_health = client.health_check("tts")
    except Exception:
        tts_health = {"status": "unavailable"}

    return {
        "streaming_endpoint": True,
        "chunked_streaming": True,
        "simple_chunked": True,
        "tts_service": tts_health,
        "features": {
            "sentence_splitting": True,
            "audio_chunking": True,
            "wav_format": True,
            "mp3_format": True,
        },
        "recommended_chunk_size": 200,
    }
