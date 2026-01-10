# ASR Service

FastAPI-based Automatic Speech Recognition (ASR) service using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Features

- **High-performance transcription** using OpenAI's Whisper with CTranslate2 optimization
- **Configurable model sizes**: tiny, base, small, medium, large-v3, distil variants
- **GPU acceleration**: CUDA support for faster inference
- **Multiple precision modes**: float16, int8, int8_float16
- **Language detection**: Automatic or explicit language specification
- **Timestamp support**: Word-level and segment-level timing data

## Installation

### From Source

```bash
cd services/asr_server
pip install -r requirements.txt
```

### With GPU Support

For CUDA-enabled transcription:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_HOST` | `0.0.0.0` | Bind address |
| `ASR_PORT` | `7001` | Port number |
| `ASR_MODEL` | `small` | Model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `ASR_DEVICE` | `cpu` | Compute device (`cpu`, `cuda`, `auto`) |
| `ASR_COMPUTE_TYPE` | `int8` | Precision (`float16`, `int8`, `int8_float16`) |
| `ASR_LANGUAGE` | `None` | Language hint (e.g., `en`, `zh`) |
| `ASR_BEAM_SIZE` | `5` | Beam search size (1-10) |
| `ASR_NUM_WORKERS` | `4` | Number of transcription workers |

### Model Selection Guide

| Model | VRAM (GPU) | Speed | Accuracy |
|-------|------------|-------|----------|
| `tiny` | ~1GB | Fastest | Good |
| `base` | ~1GB | Fast | Better |
| `small` | ~2GB | Fast | Good |
| `medium` | ~5GB | Medium | Better |
| `large-v3` | ~10GB | Slow | Best |
| `distil-medium-en` | ~3GB | Medium | Good (English) |
| `distil-small-en` | ~1GB | Fast | Good (English) |

## Usage

### Running the Server

```bash
python -m services.asr_server.app
```

With custom configuration:

```bash
ASR_MODEL=medium ASR_DEVICE=cuda ASR_COMPUTE_TYPE=float16 python -m services.asr_server.app
```

### Using with Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "services.asr_server.app"]
```

```bash
docker build -t deeptutor-asr -f services/asr_server/Dockerfile .
docker run -p 7001:7001 deeptutor-asr
```

### API Endpoints

#### Health Check

```bash
curl http://localhost:7001/health
```

Response:
```json
{
  "status": "healthy",
  "model": "small",
  "device": "cpu",
  "faster_whisper": true,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Transcribe Audio

```bash
curl -X POST http://localhost:7001/transcribe \
  -F "audio=@recording.wav" \
  -F "language=en"
```

Response:
```json
{
  "text": "Hello, this is a test transcription.",
  "language": "en",
  "duration_seconds": 3.5,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 1.5,
      "text": "Hello, this is",
      "words": [
        {"word": "Hello", "start": 0.0, "end": 0.3, "probability": 0.95},
        {"word": "this", "start": 0.3, "end": 0.6, "probability": 0.92}
      ]
    }
  ]
}
```

#### With Python (httpx)

```python
import httpx

async def transcribe_audio(audio_path: str, language: str = "en"):
    async with httpx.AsyncClient() as client:
        with open(audio_path, "rb") as audio:
            response = await client.post(
                "http://localhost:7001/transcribe",
                files={"audio": audio},
                data={"language": language},
            )
        return response.json()
```

## Error Handling

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `INVALID_AUDIO_FORMAT` | 400 | Unsupported audio format |
| `AUDIO_TOO_LARGE` | 400 | File exceeds 300MB limit |
| `EMPTY_AUDIO` | 400 | Empty audio file |
| `MODEL_NOT_LOADED` | 503 | ASR model not initialized |
| `TRANSCRIPTION_FAILED` | 500 | Internal transcription error |

## Integration with DeepTutor

### Direct Import

```python
from services.asr_server.app import app
```

### HTTP Client Integration

```python
import httpx

class ASRClient:
    def __init__(self, base_url: str = "http://localhost:7001"):
        self.base_url = base_url

    async def transcribe(self, audio_bytes: bytes, language: str = None) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/transcribe",
                files={"audio": ("audio.wav", audio_bytes, "audio/wav")},
                data={"language": language} if language else {},
            )
            response.raise_for_status()
            return response.json()

    async def health(self) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            return response.json()
```

## Testing

Run tests with pytest:

```bash
pytest services/asr_server/tests/ -v
```

Run with coverage:

```bash
pytest services/asr_server/tests/ --cov=services.asr_server --cov-report=html
```

### Mock Testing

For CI environments without GPU:

```bash
pytest services/asr_server/tests/test_asr.py -v -k "not gpu"
```

## Performance Optimization

### GPU Memory

If using CUDA, set environment variables for optimal memory:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_block_size:16777216
export CUDA_VISIBLE_DEVICES=0
```

### Batching

For high-throughput scenarios, consider adding a batching endpoint:

```python
@app.post("/transcribe/batch")
async def transcribe_batch(audios: list[UploadFile] = File(...)):
    results = []
    for audio in audios:
        result = await transcribe(audio)
        results.append(result)
    return results
```

## Security

### Authentication

The service supports optional Bearer token authentication.

```bash
# Set token (must be set on both client and server)
export VOICE_TOKEN="your-secure-token-here"

# Client request with token
curl -X POST http://localhost:7001/transcribe \
  -H "Authorization: Bearer your-secure-token-here" \
  -F "audio=@recording.wav"
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
export ASR_HOST=0.0.0.0

# Or restrict to specific interface
export ASR_HOST=192.168.1.100
```

**Windows Firewall:**
```powershell
# Allow inbound connections
New-NetFirewallRule -DisplayName "Allow ASR Server" `
  -Direction Inbound -Protocol TCP -LocalPort 7001 `
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
