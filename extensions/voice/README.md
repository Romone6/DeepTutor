# Voice Gateway API

This document describes the HTTP API contract for the DeepTutor Voice Gateway, enabling consistent ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) interactions across the platform.

## Design Principles

- **Stateless**: Each request contains all information needed for processing.
- **Content-Type Aware**: Audio I/O uses appropriate MIME types.
- **Extensible**: Optional fields allow gradual feature adoption.
- **Provider Agnostic**: Contracts abstract underlying ASR/TTS engines.

## Base URL

```
https://voice.deeptutor.hku.hk/v1
```

Headers required for all requests:

```
Authorization: Bearer <api_key>
Content-Type: <as specified per endpoint>
```

---

## Endpoints

### 1. Transcribe Audio

Convert spoken audio to text.

**Endpoint:** `POST /transcribe`

**Content-Type:** `multipart/form-data` or `audio/*`

**Request Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio` | binary | Yes | - | Audio file (WAV, MP3, OGG, FLAC) |
| `language` | string | No | auto | Language code (e.g., `en`, `zh`, `ja`) |
| `enable_speakers` | boolean | No | false | Return speaker diarization |
| `model` | string | No | default | ASR model override |

**Example Request:**

```bash
curl -X POST "https://voice.deeptutor.hku.hk/v1/transcribe" \
  -H "Authorization: Bearer $API_KEY" \
  -F "audio=@recording.wav" \
  -F "language=en" \
  -F "enable_speakers=true"
```

**Response (200 OK):**

```json
{
  "text": "What is the capital of France?",
  "language": "en",
  "duration_seconds": 3.2,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 1.5,
      "text": "What",
      "speaker": "SPEAKER_01"
    },
    {
      "id": 1,
      "start": 1.5,
      "end": 3.2,
      "text": "is the capital of France?",
      "speaker": "SPEAKER_01"
    }
  ]
}
```

**Error Response (4xx/5xx):**

```json
{
  "error": "AUDIO_TOO_LONG",
  "message": "Audio duration exceeds maximum allowed (300 seconds)",
  "details": {"max_seconds": 300, "received_seconds": 345}
}
```

---

### 2. Synthesize Speech

Convert text to spoken audio.

**Endpoint:** `POST /speak`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "text": "The capital of France is Paris.",
  "voice_id": "en-US-female-1",
  "speed": 1.0,
  "format": "mp3",
  "sample_rate": 24000,
  "emotion": "neutral"
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | - | Text to synthesize |
| `voice_id` | string | No | default | Voice identifier |
| `speed` | number | No | 1.0 | Speed multiplier (0.5 - 2.0) |
| `format` | string | No | mp3 | Output format (mp3, wav, ogg, flac) |
| `sample_rate` | integer | No | 24000 | Sample rate (8000 - 48000) |
| `emotion` | string | No | - | Emotion/style tag |

**Example Request:**

```bash
curl -X POST "https://voice.deeptutor.hku.hk/v1/speak" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The answer is forty-two.",
    "voice_id": "en-US-male-1",
    "speed": 0.9,
    "format": "wav"
  }' \
  --output answer.wav
```

**Response (200 OK):**

```
HTTP/1.1 200 OK
Content-Type: audio/wav
X-Duration-Seconds: 2.5
X-Sample-Rate: 24000

<binary audio content>
```

**Error Response (4xx/5xx):**

```json
{
  "error": "VOICE_NOT_FOUND",
  "message": "Voice 'en-US-female-xyz' is not available",
  "details": {
    "requested": "en-US-female-xyz",
    "available": ["en-US-female-1", "en-US-female-2", "zh-CN-male-1"]
  }
}
```

---

### 3. Health Check

Monitor gateway and component health.

**Endpoint:** `GET /health`

**Example Request:**

```bash
curl "https://voice.deeptutor.hku.hk/v1/health" \
  -H "Authorization: Bearer $API_KEY"
```

**Response (200 OK):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "components": [
    {
      "name": "asr",
      "status": "healthy",
      "latency_ms": 150
    },
    {
      "name": "tts",
      "status": "healthy",
      "latency_ms": 320
    },
    {
      "name": "cache",
      "status": "degraded",
      "latency_ms": 2000,
      "message": "Elevated latency detected"
    }
  ]
}
```

**Status Values:**

| Status | Meaning |
|--------|---------|
| `healthy` | Component fully operational |
| `degraded` | Component working but with elevated latency/errors |
| `unhealthy` | Component non-functional |

---

## Provider Integration

The gateway abstracts these providers:

| Provider | ASR | TTS | Notes |
|----------|-----|-----|-------|
| Whisper (local) | ✓ | - | Open-source, CPU/GPU |
| Paraformer | ✓ | - | Chinese-optimized |
| Azure TTS | - | ✓ | Cloud, high quality |
| Edge TTS | - | ✓ | Free, web API |
| ElevenLabs | - | ✓ | Premium voices |

### Local Deployment

```bash
# Run voice gateway locally
export VOICE_GATEWAY_PORT=8001
export ASR_ENGINE=whisper
export TTS_ENGINE=edge
python -m extensions.voice.main
```

### Docker Deployment

```yaml
services:
  voice-gateway:
    image: deeptutor/voice-gateway:latest
    ports:
      - "8001:8001"
    environment:
      - ASR_ENGINE=whisper
      - TTS_ENGINE=azure
    volumes:
      - ./voices:/app/voices
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUDIO_TOO_LONG` | 400 | Audio exceeds max duration |
| `INVALID_AUDIO_FORMAT` | 400 | Unsupported audio format |
| `VOICE_NOT_FOUND` | 404 | Requested voice_id invalid |
| `TEXT_TOO_LONG` | 400 | Text exceeds max characters |
| `RATE_LIMITED` | 429 | Too many requests |
| `ASR_ERROR` | 500 | ASR processing failed |
| `TTS_ERROR` | 500 | TTS processing failed |
| `AUTH_FAILED` | 401 | Invalid or missing API key |
