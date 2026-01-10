"""Stub TTS Engine.

Returns a generated WAV tone/silence for testing without external dependencies.
"""

import io
import math
import struct
from typing import Optional
from dataclasses import dataclass


@dataclass
class StubConfig:
    """Configuration for stub TTS engine."""

    sample_rate: int = 24000
    duration_ms: int = 1000
    frequency: float = 440.0  # A4 note
    amplitude: float = 0.5


class StubTTSEngine:
    """Stub TTS engine that generates a WAV tone."""

    def __init__(self, config: Optional[StubConfig] = None):
        self.config = config or StubConfig()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        """Check if engine is loaded."""
        return self._loaded

    def get_available_voices(self) -> list[dict]:
        """Return list of available stub voices."""
        return [
            {"id": "stub-male-1", "name": "Stub Male", "language": "en"},
            {"id": "stub-female-1", "name": "Stub Female", "language": "en"},
            {"id": "stub-neutral", "name": "Stub Neutral", "language": "en"},
        ]

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        format: str = "wav",
        sample_rate: Optional[int] = None,
        speed: float = 1.0,
    ) -> tuple[bytes, str]:
        """
        Synthesize speech from text.

        Returns:
            tuple: (audio_bytes, content_type)
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")

        if len(text) > 5000:
            raise ValueError("Text too long (max 5000 characters)")

        sr = sample_rate or self.config.sample_rate

        if format == "wav":
            audio_bytes = self._generate_wav(text, sr, speed)
            content_type = "audio/wav"
        elif format == "mp3":
            audio_bytes = self._generate_mp3_stub(text, sr, speed)
            content_type = "audio/mpeg"
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'wav' or 'mp3'.")

        return audio_bytes, content_type

    def _generate_wav(self, text: str, sample_rate: int, speed: float) -> bytes:
        """Generate a WAV file with a tone."""
        # Calculate duration based on text length (roughly 10 chars per second at 1x speed)
        text_length_factor = max(1.0, len(text) / 10.0)
        duration_seconds = text_length_factor / speed

        # Generate samples
        num_samples = int(sample_rate * duration_seconds)
        amplitude = int(self.config.amplitude * 32767)

        samples = []
        for i in range(num_samples):
            t = i / sample_rate
            # Simple sine wave with varying frequency based on text
            freq_mod = self.config.frequency + (hash(text[:10]) % 100)
            sample = int(amplitude * math.sin(2 * math.pi * freq_mod * t))
            samples.append(sample)

        # Write WAV header
        data_size = len(samples) * 2
        file_size = 36 + data_size

        wav_header = struct.pack(
            "<4sI4s",  # RIFF chunk
            b"RIFF",
            file_size,
            b"WAVE",
        )

        fmt_chunk = struct.pack(
            "<4sIHHIIHH",  # fmt subchunk
            b"fmt ",
            16,  # Subchunk1Size (16 for PCM)
            1,  # AudioFormat (1 = PCM)
            1,  # NumChannels (1 = mono)
            sample_rate,  # SampleRate
            sample_rate * 2,  # ByteRate
            2,  # BlockAlign
            16,  # BitsPerSample
        )

        data_chunk = struct.pack(
            "<4sI",  # data subchunk
            b"data",
            data_size,
        )

        audio_data = struct.pack("<" + "h" * len(samples), *samples)

        return wav_header + fmt_chunk + data_chunk + audio_data

    def _generate_mp3_stub(self, text: str, sample_rate: int, speed: float) -> bytes:
        """Generate a minimal MP3 stub (ID3 header + frame data)."""
        # For stub, we return a minimal MP3 container
        # In real implementation, would use an MP3 encoder

        # ID3v2.4 header
        id3_header = bytes(
            [
                0x49,
                0x44,
                0x33,  # "ID3"
                0x04,
                0x00,  # Version 2.4.0
                0x00,  # Flags
                0x00,
                0x00,
                0x00,
                0x00,  # Size (synchsafe)
            ]
        )

        # Minimal frame data (not a real MP3, but valid container)
        frame_data = bytes([0xFF] * 100)

        return id3_header + frame_data

    def get_audio_duration(self, audio_bytes: bytes) -> float:
        """Calculate approximate duration of audio in seconds."""
        # For WAV, parse header to get duration
        if len(audio_bytes) < 44:
            return 0.0

        try:
            # Check RIFF header
            if audio_bytes[:4] != b"RIFF":
                return 0.0
            if audio_bytes[8:12] != b"WAVE":
                return 0.0

            # Parse format
            # Look for fmt chunk
            offset = 12
            while offset + 8 < len(audio_bytes):
                chunk_id = audio_bytes[offset : offset + 4]
                chunk_size = struct.unpack("<I", audio_bytes[offset + 4 : offset + 8])[0]

                if chunk_id == b"fmt ":
                    num_channels = struct.unpack("<H", audio_bytes[offset + 12 : offset + 14])[0]
                    sample_rate = struct.unpack("<I", audio_bytes[offset + 16 : offset + 20])[0]
                    bits_per_sample = struct.unpack("<H", audio_bytes[offset + 22 : offset + 24])[0]

                    # Find data chunk
                    data_offset = offset + 8
                    while data_offset + 8 < len(audio_bytes):
                        data_id = audio_bytes[data_offset : data_offset + 4]
                        data_size = struct.unpack(
                            "<I", audio_bytes[data_offset + 4 : data_offset + 8]
                        )[0]

                        if data_id == b"data":
                            byte_rate = sample_rate * num_channels * bits_per_sample // 8
                            duration = data_size / byte_rate
                            return duration

                        data_offset += 8 + data_size
                        if data_offset % 2 != 0:
                            data_offset += 1  # Word alignment

                offset += 8 + chunk_size
                if offset % 2 != 0:
                    offset += 1  # Word alignment

        except Exception:
            pass

        return 0.0

    def health_check(self) -> dict:
        """Return health status."""
        return {
            "status": "healthy",
            "engine": "stub",
            "voices": len(self.get_available_voices()),
        }
