# vLLM Runtime for DeepTutor

vLLM provides high-throughput local LLM inference with GPU acceleration. This is an alternative to Ollama for users with capable GPUs (RTX 4070 Super or better).

## Why vLLM?

| Feature | Ollama | vLLM |
|---------|--------|------|
| GPU Acceleration | Partial | Full (PagedAttention) |
| Throughput | Moderate | High |
| Setup Complexity | Simple | Moderate |
| Best For | Ease of use | Performance |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 12GB | RTX 4070 Super 12GB+ |
| VRAM | 8GB | 12GB+ |
| RAM | 16GB | 32GB |
| Storage | 20GB | 50GB+ |

## Installation

### Option 1: pip install (Windows/Linux)

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate

# Install vLLM
pip install vllm
```

### Option 2: Docker (Recommended for Windows)

```powershell
# Install Docker Desktop from docker.com

# Run vLLM container
docker run --runtime nvidia --gpus all \
    -p 8000:8000 \
    --env-file .env \
    -v C:\models:/models \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --host 0.0.0.0
```

### Option 3: From Source (Linux/WSL2)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip git

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install vLLM from source
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

## Starting vLLM Server

### Quick Start (Default Settings)

```powershell
# Activate environment
.\venv\Scripts\activate

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

### With Optimized Settings for RTX 4070 Super

```powershell
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --dtype half \
    --enforce-eager
```

### Recommended Models

| Model | VRAM | Speed | Notes |
|-------|------|-------|-------|
| `meta-llama/Llama-3.1-8B-Instruct` | ~8GB | Fast | Best balance |
| `Qwen/Qwen2.5-7B-Instruct` | ~7GB | Very Fast | Great for coding |
| `meta-llama/Llama-3.2-3B-Instruct` | ~3GB | Fastest | Low VRAM |
| `meta-llama/Llama-3.1-70B-Instruct` | ~70GB | Moderate | High quality (needs A100) |

## Configuring DeepTutor

Edit `.env`:
```env
# Select vLLM runtime
LLM_BINDING=vllm

# vLLM endpoint
VLLM_BASE_URL=http://localhost:8000/v1

# Model (must match what's loaded in vLLM)
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct

# No API key needed for local vLLM
LLM_API_KEY=
```

## Verify Installation

### Test vLLM Server

```powershell
# Check if server is running
curl http://localhost:8000/v1/models
```

Expected response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      "object": "model",
      "created": 1234567890,
      "owned_by": "vllm"
    }
  ]
}
```

### Test Connection

```powershell
curl http://localhost:8000/api/llm/health
```

Expected response:
```json
{
  "status": "healthy",
  "binding": "vllm",
  "model": "meta-llama/Llama-3.1-8B-Instruct"
}
```

## Health Check Endpoint

DeepTutor provides a unified health check at:

```
GET /api/llm/health
```

Response includes:
- `status`: "healthy", "degraded", or "unhealthy"
- `binding`: Current LLM binding (e.g., "vllm", "ollama")
- `model`: Model name
- `message`: Status description
- `error`: Error details if unhealthy

## Troubleshooting

### GPU Not Detected

```
RuntimeError: No GPU found. vLLM requires CUDA-enabled GPUs.
```

**Solutions:**
1. Verify NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
3. Check Docker GPU setup: `docker run --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi`

### Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce `--gpu-memory-utilization`: `--gpu-memory-utilization 0.7`
2. Reduce `--max-model-len`: `--max-model-len 4096`
3. Use a smaller model
4. Close GPU-intensive applications

### Connection Refused

```
Connection error: http://localhost:8000/v1
```

**Solutions:**
1. Verify vLLM is running: `curl http://localhost:8000/v1/models`
2. Check port is correct: Default is `8000`
3. Firewall may be blocking: Add exception for port 8000

### Slow Inference

**Solutions:**
1. Use `--enforce-eager` for reduced memory, faster startup
2. Use `--dtype half` for FP16 inference
3. Consider a smaller model if GPU is older

## Performance Tuning

### RTX 4070 Super Optimized Settings

```powershell
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --dtype half \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

### Throughput Test

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="sk-no-key")

# Simple latency test
import time
start = time.time()
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
print(f"Latency: {time.time() - start:.2f}s")
print(f"Response: {response.choices[0].message.content}")
```

## Switching Between Runtimes

| Current | Switch To | Action |
|---------|-----------|--------|
| Ollama | vLLM | Set `LLM_BINDING=vllm`, `VLLM_BASE_URL`, `VLLM_MODEL` |
| vLLM | Ollama | Set `LLM_BINDING=ollama`, `OLLAMA_MODEL`, restart |
| OpenAI | vLLM | Set `LLM_BINDING=vllm`, configure vLLM, restart |

No code changes required - DeepTutor uses the shared provider interface.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           DeepTutor Backend          │
                    │                                     │
                    │  ┌─────────────┐   ┌─────────────┐  │
                    │  │ LLM Client  │──▶│ vLLM Client │  │
                    │  │  (unified)  │   │ (optional)  │  │
                    │  └─────────────┘   └─────────────┘  │
                    │         │                │          │
                    │         ▼                ▼          │
                    │  ┌─────────────────────────────────┤
                    │  │      OpenAI-Compatible API      │
                    │  └─────────────────────────────────┘
                    │                    │
                    │                    ▼
                    │         ┌────────────────────┐
                    │         │  vLLM Server       │
                    │         │  (Port 8000)       │
                    │         │                    │
                    │         │  • PagedAttention  │
                    │         │  • Tensor Parallel │
                    │         │  • Continuous Batching │
                    │         └────────────────────┘
                    │                    │
                    │                    ▼
                    │         ┌────────────────────┐
                    │         │    GPU (RTX 4070   │
                    │         │    Super+)         │
                    │         └────────────────────┘
                    └─────────────────────────────────────┘
```

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Model IDs](https://huggingface.co/models)
- [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)
