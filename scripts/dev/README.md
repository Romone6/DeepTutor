# DeepTutor Development Stack - Windows Setup Guide

This guide covers running the complete DeepTutor development stack on Windows, including voice services (ASR/TTS) and the main backend.

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Backend runtime |
| Node.js | 18+ | Frontend runtime |
| PowerShell | 5.1+ | Script execution (Windows 10+) |

### Installation Steps

#### 1. Install Python

Download from [python.org](https://www.python.org/downloads/windows/):
- Check "Add Python to PATH" during installation
- Recommended: Python 3.11 or 3.12

Verify:
```powershell
python --version
pip --version
```

#### 2. Install Node.js

Download from [nodejs.org](https://nodejs.org/):
- LTS version recommended (18.x or 20.x)

Verify:
```powershell
node --version
npm --version
```

#### 3. Clone Repository

```powershell
git clone https://github.com/HKUDS/DeepTutor.git
cd DeepTutor
```

#### 4. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

#### 5. Install Frontend Dependencies

```powershell
cd web
npm install
cd ..
```

#### 6. Configure Environment

Copy the example environment file:
```powershell
copy .env.example .env
```

Edit `.env` and configure:
```env
LLM_BINDING=openai
LLM_MODEL=gpt-4
LLM_HOST=https://api.openai.com/v1
LLM_API_KEY=your-api-key-here
```

---

## Running the Development Stack

### Quick Start

Open PowerShell and run:
```powershell
cd scripts\dev
.\run_all.ps1
```

This starts:
- **ASR Service** on port 7001 (speech-to-text)
- **TTS Service** on port 7002 (text-to-speech)
- **DeepTutor Backend** on port 8000 (API server)
- **Frontend** on port 3000 (web UI)

### Options

```powershell
.\run_all.ps1 -AsrPort 7001 -TtsPort 7002 -BackendPort 8000
.\run_all.ps1 -SkipFrontend              # Skip frontend (for API-only testing)
.\run_all.ps1 -StubVoice                 # Use stub voice services (no external dependencies)
```

### Access URLs

| Service | Local URL | LAN URL |
|---------|-----------|---------|
| Frontend | http://localhost:3000 | http://192.168.x.x:3000 |
| Backend API | http://localhost:8000 | http://192.168.x.x:8000 |
| API Docs | http://localhost:8000/docs | http://192.168.x.x:8000/docs |
| ASR Service | http://localhost:7001 | http://192.168.x.x:7001 |
| TTS Service | http://localhost:7002 | http://192.168.x.x:7002 |

---

## Environment Variables

### Core Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BINDING` | LLM provider (`openai`, `azure`, `ollama`) | `openai` |
| `LLM_MODEL` | Model name | `gpt-4` |
| `LLM_HOST` | API endpoint URL | - |
| `LLM_API_KEY` | API key | - |

### Voice Service Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ASR_ENGINE` | ASR engine (`whisper`, `azure`, `deepgram`) | `stub` |
| `TTS_ENGINE` | TTS engine (`edge`, `azure`, `elevenlabs`) | `stub` |
| `ASR_PORT` | ASR service port | `7001` |
| `TTS_PORT` | TTS service port | `7002` |
| `VOICE_LANGUAGE` | Default language for voice | `en` |

### Example .env for Voice Services

```env
# Core LLM
LLM_BINDING=openai
LLM_MODEL=gpt-4
LLM_HOST=https://api.openai.com/v1
LLM_API_KEY=sk-...

# Voice Services (optional)
ASR_ENGINE=whisper
TTS_ENGINE=azure
ASR_PORT=7001
TTS_PORT=7002
```

---

## Troubleshooting

### Port Already in Use

```
Port 8000 is in use by process 'python' (PID: 12345)
```

**Solution:**
```powershell
# Find the process
netstat -ano | findstr :8000

# Stop the process (replace PID)
taskkill /PID 12345 /F
```

Or use a different port:
```powershell
.\run_all.ps1 -BackendPort 8001
```

### Python Not Found

Ensure Python is in your PATH:
```powershell
# Check Python location
where python

# If not found, reinstall Python with "Add to PATH" checked
```

### npm install Fails

```powershell
# Clear npm cache
npm cache clean --force

# Delete node_modules and retry
Remove-Item -Recurse -Force web\node_modules
cd web && npm install && cd ..
```

### Permission Errors

Run PowerShell as Administrator, or:
```powershell
# Set execution policy (one-time)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Voice Services Not Starting

If using real ASR/TTS engines instead of stubs:

1. **Whisper (ASR)**: Install [Whisper](https://github.com/openai/whisper) or use [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
2. **Azure TTS**: Configure `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION`
3. **ElevenLabs**: Set `ELEVENLABS_API_KEY`

For development without external services, use `-StubVoice` flag.

---

## Local LLM with Ollama

DeepTutor can use [Ollama](https://ollama.com/) as a local LLM provider, eliminating API costs and enabling offline operation.

### Why Ollama?

- **Privacy**: All processing happens locally
- **No API costs**: Free to use any model
- **Offline capable**: Works without internet
- **Fast setup**: Simple installation on Windows

### Recommended Models

| Model | Size | VRAM Required | Best For |
|-------|------|---------------|----------|
| `llama3.1:8b-instruct` | ~8GB | 6GB+ | General purpose, balanced speed/quality |
| `qwen2.5:7b-instruct` | ~7GB | 5GB+ | Good coding abilities, fast inference |
| `llama3.2:3b-instruct` | ~3GB | 4GB+ | Lightweight, suitable for older hardware |

### Installation

#### 1. Download Ollama

Download from [ollama.com](https://ollama.com/) for Windows.

Run the installer and follow the prompts.

#### 2. Start Ollama

Ollama runs in the background as a system service.

```powershell
# Verify Ollama is running
curl http://localhost:11434/api/version
```

You should see a JSON response with version info.

#### 3. Pull a Model

```powershell
# Pull the recommended model (one at a time)
ollama pull llama3.1:8b-instruct

# Or try Qwen2.5 (good for coding)
ollama pull qwen2.5:7b-instruct

# Or use the lightweight 3B model
ollama pull llama3.2:3b-instruct

# List installed models
ollama list
```

#### 4. Configure DeepTutor

Edit `.env`:
```env
# Switch to Ollama binding
LLM_BINDING=ollama

# Use the model you pulled
LLM_MODEL=llama3.1:8b-instruct

# Ollama's local endpoint
LLM_HOST=http://localhost:11434/v1

# No API key needed for local Ollama
LLM_API_KEY=

# Optional: Embedding model for RAG (also via Ollama)
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_HOST=http://localhost:11434/v1
EMBEDDING_API_KEY=
```

### Troubleshooting Ollama

#### Model Not Found

```
Error: model 'llama3.1:8b-instruct' not found
```

**Solution:** Pull the model first:
```powershell
ollama pull llama3.1:8b-instruct
```

#### Connection Refused

```
Connection error: http://localhost:11434/v1
```

**Solutions:**
1. Verify Ollama is running:
   ```powershell
   curl http://localhost:11434/api/version
   ```
2. Restart Ollama service
3. Check firewall settings

#### Out of Memory

```
Error: model requires more VRAM than available
```

**Solutions:**
1. Use a smaller model (e.g., `llama3.2:3b-instruct`)
2. Close other GPU-intensive applications
3. Enable CPU offloading in Ollama settings

#### Slow Performance

**Tips:**
- Use `qwen2.5:7b-instruct` for faster inference
- Reduce `max_tokens` in prompts
- Close background applications

### Health Check

Verify Ollama is working:
```powershell
curl http://localhost:11434/api/tags
```

DeepTutor also provides a health endpoint:
```powershell
curl http://localhost:8000/api/llm/health
```

Expected response:
```json
{
  "status": "healthy",
  "binding": "ollama",
  "model": "llama3.1:8b-instruct"
}
```

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Development Machine         │
                    │                                     │
┌──────────────┐    │  ┌─────────┐   ┌─────────┐        │
│   Browser    │───▶│  │  Next.js│   │  Uvicorn│        │
│  (Port 3000) │    │  │ Frontend│   │ Backend │        │
└──────────────┘    │  └─────────┘   │(Port8000)│        │
                    │                └────┬────┘        │
                    │                     │              │
                    │    ┌────────────────┼──────────┐  │
                    │    │                │          │  │
                    │    ▼                ▼          │  │
                    │  ┌─────────┐   ┌─────────┐    │  │
                    │  │   ASR   │   │   TTS   │    │  │
                    │  │(Port7001)│   │(Port7002)│   │  │
                    │  └─────────┘   └─────────┘    │  │
                    │                     ▲          │  │
                    └─────────────────────┼──────────┘  │
                                      HTTP/REST         │
                                                      │
                        External Services (optional)   │
                        • OpenAI (LLM)                 │
                        • Azure Speech                 │
                        • Whisper                      │
```

---

## Production vs Development

| Aspect | Development | Production |
|--------|-------------|------------|
| ASR Engine | Stub/Whisper local | Azure/Deepgram |
| TTS Engine | Stub/Edge TTS | Azure/ElevenLabs |
| Backend | Uvicorn (reload=True) | Gunicorn + Nginx |
| Frontend | `npm run dev` | `npm run build` |
| Port binding | 0.0.0.0 | 127.0.0.1 |

---

## Next Steps

After successfully running the stack:

1. Open http://localhost:3000 in your browser
2. Test the API at http://localhost:8000/docs
3. Try voice features (requires real ASR/TTS engines configured)
4. Check logs in `data/logs/` for debugging
