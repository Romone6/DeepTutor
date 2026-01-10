"""Voice Gateway Client for DeepTutor.

HTTP client for calling ASR (Automatic Speech Recognition) and TTS (Text-to-Speech)
services. Provides retry logic, timeout handling, and clear error messages.

Usage:
    from extensions.voice.client import VoiceGatewayClient

    client = VoiceGatewayClient()
    text = client.transcribe_audio(audio_bytes, "audio.wav")
    audio = client.synthesize_speech("Hello world", voice_id="stub-male-1")

Configuration via Environment Variables:
    VOICE_ASR_URL: ASR service URL (default: http://localhost:7001)
    VOICE_TTS_URL: TTS service URL (default: http://localhost:7002)
    VOICE_TOKEN: Optional Bearer token for authentication
    VOICE_TIMEOUT: Request timeout in seconds (default: 30)
    VOICE_MAX_RETRIES: Maximum retry attempts (default: 3)
    VOICE_RETRY_DELAY: Delay between retries in seconds (default: 0.5)
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# =============================================================================
# Exceptions
# =============================================================================


class VoiceGatewayError(Exception):
    """Base exception for Voice Gateway errors."""

    pass


class VoiceGatewayTimeoutError(VoiceGatewayError):
    """Raised when a request times out."""

    pass


class VoiceGatewayConnectionError(VoiceGatewayError):
    """Raised when connection to service fails."""

    pass


class VoiceGatewayServiceError(VoiceGatewayError):
    """Raised when service returns an error response."""

    def __init__(self, message: str, status_code: int = None, response_body: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class VoiceGatewayAudioError(VoiceGatewayError):
    """Raised when audio processing fails."""

    pass


class VoiceGatewayValidationError(VoiceGatewayError):
    """Raised when input validation fails."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class VoiceClientConfig:
    """Configuration for Voice Gateway client."""

    asr_url: str = "http://localhost:7001"
    tts_url: str = "http://localhost:7002"
    token: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 0.5

    @classmethod
    def from_env(cls) -> "VoiceClientConfig":
        """Create config from environment variables."""
        return cls(
            asr_url=os.getenv("VOICE_ASR_URL", "http://localhost:7001"),
            tts_url=os.getenv("VOICE_TTS_URL", "http://localhost:7002"),
            token=os.getenv("VOICE_TOKEN", None) or None,
            timeout=float(os.getenv("VOICE_TIMEOUT", "30")),
            max_retries=int(os.getenv("VOICE_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("VOICE_RETRY_DELAY", "0.5")),
        )


# =============================================================================
# Client Class
# =============================================================================


class VoiceGatewayClient:
    """
    HTTP client for DeepTutor Voice Gateway services.

    Provides methods for:
    - transcribe_audio(): Convert audio to text (ASR)
    - synthesize_speech(): Convert text to audio (TTS)

    Features:
    - Automatic retry with exponential backoff
    - Configurable timeouts
    - Optional Bearer token authentication
    - Clear, actionable error messages
    """

    def __init__(self, config: Optional[VoiceClientConfig] = None):
        """
        Initialize Voice Gateway client.

        Args:
            config: Optional configuration. If not provided, loads from environment.
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for VoiceGatewayClient. Install with: pip install httpx"
            )

        self.config = config or VoiceClientConfig.from_env()
        self._client: Optional[httpx.Client] = None

    @property
    def asr_url(self) -> str:
        """Get ASR service URL."""
        return self.config.asr_url.rstrip("/")

    @property
    def tts_url(self) -> str:
        """Get TTS service URL."""
        return self.config.tts_url.rstrip("/")

    def _get_headers(self) -> dict:
        """Get request headers with optional authentication."""
        headers = {}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        return headers

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.config.timeout,
                headers=self._get_headers(),
            )
        return self._client

    def _close(self):
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._close()

    def _execute_with_retry(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        """
        Execute HTTP request with retry logic.

        Args:
            request: The httpx.Request to execute

        Returns:
            httpx.Response on success

        Raises:
            VoiceGatewayTimeoutError: If request times out
            VoiceGatewayConnectionError: If connection fails
            VoiceGatewayServiceError: If service returns error
        """
        client = self._get_client()
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                response = client.send(request)
                return response

            except httpx.TimeoutException as e:
                last_exception = VoiceGatewayTimeoutError(
                    f"Request timed out after {self.config.timeout}s"
                )
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2**attempt)  # Exponential backoff
                    time.sleep(delay)
                    continue
                raise last_exception

            except httpx.ConnectError as e:
                last_exception = VoiceGatewayConnectionError(f"Failed to connect to service: {e}")
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2**attempt)
                    time.sleep(delay)
                    continue
                raise last_exception

            except httpx.HTTPError as e:
                last_exception = VoiceGatewayError(f"HTTP error: {e}")
                raise last_exception

        raise last_exception

    # =============================================================================
    # ASR Methods
    # =============================================================================

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename (for content-type detection)
            language: Optional language hint (e.g., 'en', 'zh')

        Returns:
            Transcribed text string

        Raises:
            VoiceGatewayValidationError: If input validation fails
            VoiceGatewayTimeoutError: If request times out
            VoiceGatewayConnectionError: If connection fails
            VoiceGatewayServiceError: If service returns error
            VoiceGatewayAudioError: If audio processing fails
        """
        # Validate input
        if not audio_bytes:
            raise VoiceGatewayValidationError("Audio bytes cannot be empty")

        if len(audio_bytes) > 300 * 1024 * 1024:
            raise VoiceGatewayValidationError(
                f"Audio file too large ({len(audio_bytes) / (1024 * 1024):.1f}MB). "
                "Maximum size is 300MB"
            )

        # Determine content type from filename
        content_type = self._get_content_type(filename)

        # Build request
        url = f"{self.asr_url}/transcribe"

        files = {"audio": (filename, audio_bytes, content_type)}
        data = {}
        if language:
            data["language"] = language

        request = httpx.Request(
            method="POST",
            url=url,
            files=files,
            data=data,
        )

        # Execute with retry
        response = self._execute_with_retry(request)

        # Handle response
        if response.status_code == 200:
            try:
                result = response.json()
                return result.get("text", "").strip()
            except Exception as e:
                raise VoiceGatewayAudioError(f"Failed to parse transcription response: {e}")

        elif response.status_code == 400:
            error_detail = self._get_error_detail(response)
            raise VoiceGatewayAudioError(f"Audio validation failed: {error_detail}")

        elif response.status_code == 503:
            raise VoiceGatewayServiceError(
                "ASR service is not available",
                status_code=response.status_code,
                response_body=response.text,
            )

        else:
            raise VoiceGatewayServiceError(
                f"ASR service returned status {response.status_code}",
                status_code=response.status_code,
                response_body=response.text,
            )

    async def atranscribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
    ) -> str:
        """
        Async version of transcribe_audio.

        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename
            language: Optional language hint

        Returns:
            Transcribed text string
        """
        if not audio_bytes:
            raise VoiceGatewayValidationError("Audio bytes cannot be empty")

        content_type = self._get_content_type(filename)
        url = f"{self.asr_url}/transcribe"

        files = {"audio": (filename, audio_bytes, content_type)}
        data = {}
        if language:
            data["language"] = language

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(url, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            return result.get("text", "").strip()

    # =============================================================================
    # TTS Methods
    # =============================================================================

    def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        format: str = "wav",
        speed: float = 1.0,
    ) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier (service-specific)
            format: Output format ('wav' or 'mp3')
            speed: Speech speed multiplier (0.5 - 2.0)

        Returns:
            Raw audio bytes

        Raises:
            VoiceGatewayValidationError: If input validation fails
            VoiceGatewayTimeoutError: If request times out
            VoiceGatewayConnectionError: If connection fails
            VoiceGatewayServiceError: If service returns error
        """
        # Validate input
        if not text or not text.strip():
            raise VoiceGatewayValidationError("Text cannot be empty")

        text = text.strip()
        if len(text) > 5000:
            raise VoiceGatewayValidationError(
                f"Text too long ({len(text)} chars). Maximum is 5000 characters"
            )

        if speed < 0.5 or speed > 2.0:
            raise VoiceGatewayValidationError(f"Speed must be between 0.5 and 2.0, got {speed}")

        if format not in ("wav", "mp3"):
            raise VoiceGatewayValidationError(f"Format must be 'wav' or 'mp3', got '{format}'")

        # Build request
        url = f"{self.tts_url}/speak"
        payload = {
            "text": text,
            "format": format,
            "speed": speed,
        }
        if voice_id:
            payload["voice_id"] = voice_id

        request = httpx.Request(
            method="POST",
            url=url,
            json=payload,
        )

        # Execute with retry
        response = self._execute_with_retry(request)

        # Handle response
        if response.status_code == 200:
            return response.content

        elif response.status_code == 400:
            error_detail = self._get_error_detail(response)
            raise VoiceGatewayValidationError(f"Request validation failed: {error_detail}")

        elif response.status_code == 503:
            raise VoiceGatewayServiceError(
                "TTS service is not available",
                status_code=response.status_code,
                response_body=response.text,
            )

        else:
            raise VoiceGatewayServiceError(
                f"TTS service returned status {response.status_code}",
                status_code=response.status_code,
                response_body=response.text,
            )

    async def asynthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        format: str = "wav",
        speed: float = 1.0,
    ) -> bytes:
        """
        Async version of synthesize_speech.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier
            format: Output format ('wav' or 'mp3')
            speed: Speech speed multiplier

        Returns:
            Raw audio bytes
        """
        if not text or not text.strip():
            raise VoiceGatewayValidationError("Text cannot be empty")

        url = f"{self.tts_url}/speak"
        payload = {"text": text.strip(), "format": format, "speed": speed}
        if voice_id:
            payload["voice_id"] = voice_id

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.content

    # =============================================================================
    # Health Check Methods
    # =============================================================================

    def health_check(self, service: str = "asr") -> dict:
        """
        Check health of a voice service.

        Args:
            service: Service to check ('asr' or 'tts')

        Returns:
            Health status dict

        Raises:
            VoiceGatewayConnectionError: If service unreachable
            VoiceGatewayServiceError: If service returns error
        """
        if service == "asr":
            url = f"{self.asr_url}/health"
        elif service == "tts":
            url = f"{self.tts_url}/health"
        else:
            raise VoiceGatewayValidationError(f"Unknown service: {service}. Use 'asr' or 'tts'")

        request = httpx.Request(method="GET", url=url)

        try:
            response = self._execute_with_retry(request)
            if response.status_code == 200:
                return response.json()
            else:
                raise VoiceGatewayServiceError(
                    f"Health check failed with status {response.status_code}",
                    status_code=response.status_code,
                )
        except VoiceGatewayConnectionError:
            raise VoiceGatewayConnectionError(f"Cannot connect to {service} service at {url}")

    async def ahealth_check(self, service: str = "asr") -> dict:
        """Async version of health_check."""
        if service == "asr":
            url = f"{self.asr_url}/health"
        else:
            url = f"{self.tts_url}/health"

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    def get_available_voices(self) -> list[dict]:
        """
        Get list of available TTS voices.

        Returns:
            List of voice dictionaries with 'id', 'name', 'language'

        Raises:
            VoiceGatewayConnectionError: If service unreachable
            VoiceGatewayServiceError: If service returns error
        """
        url = f"{self.tts_url}/voices"

        request = httpx.Request(method="GET", url=url)

        try:
            response = self._execute_with_retry(request)
            if response.status_code == 200:
                result = response.json()
                return result.get("voices", [])
            else:
                raise VoiceGatewayServiceError(
                    f"Failed to get voices: status {response.status_code}",
                    status_code=response.status_code,
                )
        except VoiceGatewayConnectionError:
            raise VoiceGatewayConnectionError(f"Cannot connect to TTS service at {url}")

    # =============================================================================
    # Helper Methods
    # =============================================================================

    def _get_content_type(self, filename: str) -> str:
        """Get content type from filename."""
        ext = Path(filename).suffix.lower()

        content_types = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".webm": "audio/webm",
            ".aac": "audio/aac",
        }

        return content_types.get(ext, "application/octet-stream")

    def _get_error_detail(self, response: httpx.Response) -> str:
        """Extract error detail from response."""
        try:
            error_json = response.json()
            if "detail" in error_json:
                return str(error_json["detail"])
            if "message" in error_json:
                return str(error_json["message"])
            if "error" in error_json:
                return str(error_json["error"])
        except Exception:
            pass

        return response.text[:200] if response.text else f"Status {response.status_code}"


# =============================================================================
# Factory Function
# =============================================================================


def create_voice_client(
    asr_url: str = None,
    tts_url: str = None,
    token: str = None,
    timeout: float = None,
    max_retries: int = None,
) -> VoiceGatewayClient:
    """
    Create a VoiceGatewayClient with the specified configuration.

    All parameters are optional. If not provided, they are loaded from
    environment variables with defaults.

    Args:
        asr_url: ASR service URL
        tts_url: TTS service URL
        token: Optional Bearer token
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts

    Returns:
        Configured VoiceGatewayClient instance
    """
    config = VoiceClientConfig.from_env()

    if asr_url:
        config.asr_url = asr_url
    if tts_url:
        config.tts_url = tts_url
    if token is not None:
        config.token = token
    if timeout is not None:
        config.timeout = timeout
    if max_retries is not None:
        config.max_retries = max_retries

    return VoiceGatewayClient(config)


# =============================================================================
# Convenience Functions
# =============================================================================


def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language: Optional[str] = None,
) -> str:
    """
    Convenience function to transcribe audio.

    Creates a client, transcribes audio, and closes the client.

    Args:
        audio_bytes: Raw audio file bytes
        filename: Original filename
        language: Optional language hint

    Returns:
        Transcribed text string
    """
    with create_voice_client() as client:
        return client.transcribe_audio(audio_bytes, filename, language)


def synthesize_speech(
    text: str,
    voice_id: Optional[str] = None,
    format: str = "wav",
    speed: float = 1.0,
) -> bytes:
    """
    Convenience function to synthesize speech.

    Creates a client, synthesizes speech, and closes the client.

    Args:
        text: Text to synthesize
        voice_id: Voice identifier
        format: Output format ('wav' or 'mp3')
        speed: Speech speed multiplier

    Returns:
        Raw audio bytes
    """
    with create_voice_client() as client:
        return client.synthesize_speech(text, voice_id, format, speed)
