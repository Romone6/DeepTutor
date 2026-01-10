# TTS Service

FastAPI-based Text-to-Speech (TTS) service with VibeVoice integration and stub engine for testing.

## Features

- **Dual engine support**: VibeVoice (production) and stub (testing)
- **Configurable output**: WAV and MP3 formats
- **Adjustable parameters**: Voice, speed, sample rate
- **Health monitoring**: Engine state and voice listing
- **Zero dependencies for stub mode**: Works without VibeVoice installed

## Installation

### Basic Installation (Stub Mode Only)

```bash
cd services/tts_server
pip install -r requirements.txt
```

### With VibeVoice Support

```bash
pip install -r requirements.txt
pip install vibevoice  # VibeVoice library
```

### Model Setup (VibeVoice)

Download VibeVoice model files and set environment variable:

```bash
export VIBEVOICE_MODEL_PATH=/path/to/vibevoice/model
```

Or place model files in default locations:
- `services/tts_server/models/vibevoice/`
- `~/.cache/vibevoice/`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_HOST` | `0.0.0.0` | Bind address |
| `TTS_PORT` | `7002` | Port number |
| `TTS_ENGINE` | `stub` | Engine: `stub` or `vibevoice` |
| `TTS_VOICE` | `stub-male-1` | Default voice ID |
| `TTS_FORMAT` | `wav` | Output format: `wav` or `mp3` |
| `TTS_SAMPLE_RATE` | `24000` | Sample rate in Hz |

### VibeVoice-Specific Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIBEVOICE_MODEL_PATH` | auto-detect | Path to model directory |
| `VIBEVOICE_DEVICE` | `cpu` | Device: `cpu` or `cuda` |
| `VIBEVOICE_COMPUTE_TYPE` | `int8` | Precision: `float16` or `int8` |

## Usage

### Running the Server

```bash
python -m services.tts_server.app
```

With VibeVoice:
```bash
TTS_ENGINE=vibevoice VIBEVOICE_MODEL_PATH=/models python -m services.tts_server.app
```

### API Endpoints

#### Health Check

```bash
curl http://localhost:7002/health
```

Response:
```json
{
  "status": "healthy",
  "engine": "stub",
  "model_loaded": true,
  "voices_available": 3,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### List Voices

```bash
curl http://localhost:7002/voices
```

Response:
```json
{
  "voices": [
    {"id": "stub-male-1", "name": "Stub Male", "language": "en"},
    {"id": "stub-female-1", "name": "Stub Female", "language": "en"}
  ]
}
```

#### Synthesize Speech

```bash
curl -X POST http://localhost:7002/speak \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "voice_id": "stub-male-1",
    "format": "wav",
    "speed": 1.0
  }' \
  --output hello.wav
```

Response headers:
```
Content-Type: audio/wav
Content-Length: 1234
X-Duration-Seconds: 1.5
X-Sample-Rate: 24000
X-Format: wav
```

#### With Python (httpx)

```python
import httpx
import asyncio

async def synthesize(text: str, voice_id: str = None):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:7002/speak",
            json={
                "text": text,
                "voice_id": voice_id,
                "format": "wav",
                "speed": 1.0,
            },
        )
        if response.status_code == 200:
            return response.content  # Raw audio bytes
        else:
            raise Exception(f"Synthesis failed: {response.json()}")

audio = asyncio.run(synthesize("Hello, world!"))
with open("output.wav", "wb") as f:
    f.write(audio)
```

## Engine Details

### Stub Engine

The stub engine generates a sine wave tone for testing. It requires no external dependencies and is always available.

**Features:**
- Zero dependencies
- Generates WAV files with sine wave tone
- Duration proportional to text length
- Useful for CI/CD and development

**Example:**
```bash
TTS_ENGINE=stub python -m services.tts_server.app
```

### VibeVoice Engine

VibeVoice is a neural TTS system that produces high-quality speech.

**Requirements:**
- VibeVoice library installed
- Pre-trained model files
- GPU recommended for real-time synthesis

**Performance:**
- Model sizes: 100MB - 1GB depending on quality
- GPU memory: 512MB - 4GB
- Real-time factor: 0.1x - 10x (CPU/GPU dependent)

## Integration with DeepTutor

### HTTP Client

```python
class TTSClient:
    def __init__(self, base_url: str = "http://localhost:7002"):
        self.base_url = base_url

    async def speak(
        self,
        text: str,
        voice_id: str = None,
        format: str = "wav",
    ) -> bytes:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/speak",
                json={
                    "text": text,
                    "voice_id": voice_id,
                    "format": format,
                },
            )
            response.raise_for_status()
            return response.content

    async def health(self) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            return response.json()

    async def voices(self) -> list:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/voices")
            return response.json()["voices"]
```

### Direct Import

```python
from services.tts_server.engine_stub import StubTTSEngine
from services.tts_server.engine_vibevoice import VibeVoiceEngine

# Use stub for testing
engine = StubTTSEngine()
audio, content_type = engine.synthesize("Hello", voice_id="stub-male-1")
```

## Testing

Run tests with pytest:

```bash
pytest services/tts_server/tests/ -v
```

Run stub-only tests (no VibeVoice required):

```bash
pytest services/tts_server/tests/ -v -k "stub"
```

### Test Coverage

- Health endpoint responses
- Voice listing
- Audio synthesis (stub mode)
- Error handling (empty text, invalid format)
- WAV header validation
- Response headers validation

## Docker

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install VibeVoice (optional)
# RUN pip install vibevoice

COPY . .

CMD ["python", "-m", "services.tts_server.app"]
```

### Docker Compose

```yaml
services:
  tts:
    build: .
    ports:
      - "7002:7002"
    environment:
      - TTS_ENGINE=stub
      - TTS_HOST=0.0.0.0
```

## Error Handling

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `EMPTY_TEXT` | 400 | No text provided |
| `TEXT_TOO_LONG` | 400 | Text exceeds 5000 characters |
| `UNSUPPORTED_FORMAT` | 400 | Invalid format specified |
| `ENGINE_NOT_LOADED` | 503 | TTS engine not initialized |
| `SYNTHESIS_FAILED` | 500 | Internal synthesis error |

## Security

### Authentication

The service supports optional Bearer token authentication.

```bash
# Set token (must be set on both client and server)
export VOICE_TOKEN="your-secure-token-here"

# Client request with token
curl -X POST http://localhost:7002/speak \
  -H "Authorization: Bearer your-secure-token-here" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

If `VOICE_TOKEN` is not set, authentication is bypassed (for local development).

### CORS Configuration

Restrict allowed origins for browser-based clients:

```bash
# Allow specific origins (comma-separated)
export VOICE_ALLOWED_ORIGINS="http://localhost:3000,https://deeptutor.hku.hk"

# Disable CORS entirely
export VOICE_CORS_ENABLED=false
```

### Rate Limiting

Prevent abuse with per-IP rate limiting:

```bash
# Requests per minute (default: 60)
export VOICE_RATE_LIMIT_RPM=30

# Disable rate limiting
export VOICE_RATE_LIMIT_ENABLED=false
```

### Security Headers

All responses include security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'self'`

### LAN-Only Deployment

For local network access only:

```bash
# Bind to all interfaces (accessible on LAN)
export TTS_HOST=0.0.0.0

# Or restrict to specific interface
export TTS_HOST=192.168.1.100
```

**Windows Firewall:**
```powershell
# Allow inbound connections
New-NetFirewallRule -DisplayName "Allow TTS Server" `
  -Direction Inbound -Protocol TCP -LocalPort 7002 `
  -Action Allow -Profile Any
```

### Environment Variables Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICE_TOKEN` | - | Bearer token for authentication |
| `VOICE_ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `VOICE_CORS_ENABLED` | `true` | Enable/disable CORS |
| `VOICE_RATE_LIMIT_RPM` | `60` | Requests per minute |
| `VOICE_RATE_LIMIT_ENABLED` | `true` | Enable/disable rate limiting |

## License

Part of the DeepTutor project. See parent repository for license information.
