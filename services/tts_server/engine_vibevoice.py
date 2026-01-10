"""VibeVoice TTS Engine Integration.

This module provides integration with VibeVoice for text-to-speech synthesis.
VibeVoice is a neural TTS system that produces high-quality speech.

Requirements:
    - VibeVoice library installed
    - Pre-trained model files
    - GPU recommended for real-time synthesis

Environment Variables:
    VIBEVOICE_MODEL_PATH: Path to VibeVoice model directory
    VIBEVOICE_DEVICE: Device to use (cuda, cpu)
    VIBEVOICE_compute_type: Precision (float16, int8)
"""

import os
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


# Check if VibeVoice is available
try:
    import vibevoice

    VIBEVOICE_AVAILABLE = True
except ImportError:
    VIBEVOICE_AVAILABLE = False


@dataclass
class VibeVoiceConfig:
    """Configuration for VibeVoice TTS engine."""

    model_path: str = ""
    device: str = "cpu"
    compute_type: str = "int8"
    sample_rate: int = 24000
    voice: str = "default"


class VibeVoiceEngine:
    """VibeVoice TTS engine wrapper."""

    def __init__(self, config: Optional[VibeVoiceConfig] = None):
        self.config = config or VibeVoiceConfig()
        self._model = None
        self._loaded = False

        if not VIBEVOICE_AVAILABLE:
            raise RuntimeError("VibeVoice is not installed. Install with: pip install vibevoice")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def load(self) -> None:
        """Load the VibeVoice model."""
        if self._loaded:
            return

        model_path = self.config.model_path
        if not model_path:
            # Try default locations
            default_paths = [
                Path(__file__).parent / "models" / "vibevoice",
                Path.home() / ".cache" / "vibevoice",
                Path("/opt/models/vibevoice"),
            ]
            for path in default_paths:
                if path.exists():
                    model_path = str(path)
                    break

            if not model_path:
                raise ValueError(
                    "VibeVoice model not found. Set VIBEVOICE_MODEL_PATH environment variable "
                    "or ensure model is in default location."
                )

        try:
            # VibeVoice loading pattern (adjust based on actual API)
            self._model = vibevoice.load(
                model_path=model_path,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load VibeVoice model: {e}")

    def unload(self) -> None:
        """Unload the model and release resources."""
        if self._model is not None:
            try:
                vibevoice.unload(self._model)
            except Exception:
                pass
            self._model = None
            self._loaded = False

    def get_available_voices(self) -> list[dict]:
        """Return list of available VibeVoice voices."""
        if not self._loaded:
            self.load()

        try:
            voices = vibevoice.list_voices(self._model)
            return [
                {"id": v["id"], "name": v["name"], "language": v.get("language", "en")}
                for v in voices
            ]
        except Exception:
            # Return default voice if listing fails
            return [{"id": "default", "name": "Default Voice", "language": "en"}]

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        format: str = "wav",
        sample_rate: Optional[int] = None,
        speed: float = 1.0,
    ) -> tuple[bytes, str]:
        """
        Synthesize speech using VibeVoice.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier (uses default if not specified)
            format: Output format (wav, mp3)
            sample_rate: Sample rate in Hz
            speed: Speech speed multiplier (0.5 - 2.0)

        Returns:
            tuple: (audio_bytes, content_type)
        """
        if not self._loaded:
            self.load()

        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")

        if len(text) > 5000:
            raise ValueError("Text too long (max 5000 characters)")

        sr = sample_rate or self.config.sample_rate
        voice = voice_id or self.config.voice

        try:
            # VibeVoice synthesis (adjust API based on actual implementation)
            audio_data = vibevoice.synthesize(
                self._model,
                text=text,
                voice=voice,
                sample_rate=sr,
                speed=speed,
            )

            if format == "wav":
                audio_bytes = self._wrap_in_wav(audio_data, sr)
                content_type = "audio/wav"
            elif format == "mp3":
                audio_bytes = self._encode_mp3(audio_data, sr)
                content_type = "audio/mpeg"
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'wav' or 'mp3'.")

            return audio_bytes, content_type

        except Exception as e:
            raise RuntimeError(f"VibeVoice synthesis failed: {e}")

    def _wrap_in_wav(self, audio_data: bytes, sample_rate: int) -> bytes:
        """Wrap raw audio data in WAV container."""
        import struct
        import wave

        # Create WAV file in memory
        import io

        with io.BytesIO() as wav_file:
            with wave.open(wav_file, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)

            return wav_file.getvalue()

    def _encode_mp3(self, audio_data: bytes, sample_rate: int) -> bytes:
        """Encode audio data to MP3 format."""
        try:
            import lameenc

            encoder = lameenc.Encoder()
            encoder.set_bit_rate(128)
            encoder.set_in_sample_rate(sample_rate)
            encoder.set_channels(1)
            encoder.set_quality(2)
            mp3_data = encoder.encode(audio_data)
            mp3_data += encoder.flush()
            return mp3_data
        except ImportError:
            # Return WAV if lame encoder not available
            return self._wrap_in_wav(audio_data, sample_rate)

    def get_audio_duration(self, audio_bytes: bytes) -> float:
        """Calculate duration of audio in seconds."""
        import struct
        import wave
        import io

        try:
            with io.BytesIO(audio_bytes) as f:
                with wave.open(f) as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    return frames / rate
        except Exception:
            return 0.0

    def health_check(self) -> dict:
        """Return health status."""
        return {
            "status": "healthy" if self._loaded else "degraded",
            "engine": "vibevoice",
            "model_loaded": self._loaded,
            "device": self.config.device,
        }


def create_vibevoice_engine(config: Optional[VibeVoiceConfig] = None) -> VibeVoiceEngine:
    """Factory function to create VibeVoice engine."""
    if not VIBEVOICE_AVAILABLE:
        raise RuntimeError(
            "VibeVoice library is not installed. Install with: pip install vibevoice"
        )
    return VibeVoiceEngine(config)
