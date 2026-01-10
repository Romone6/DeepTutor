#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Unified development runner for DeepTutor stack on Windows.
.DESCRIPTION
    Starts ASR service (port 7001), TTS service (port 7002), and DeepTutor backend (port 8000).
    Uses background jobs for concurrent execution with proper port conflict detection.
.PARAMETER AsrPort
    Port for ASR service (default: 7001)
.PARAMETER TtsPort
    Port for TTS service (default: 7002)
.PARAMETER BackendPort
    Port for DeepTutor backend (default: 8000)
.PARAMETER FrontendPort
    Port for frontend dev server (default: 3000)
.PARAMETER SkipFrontend
    Skip starting the frontend dev server
.PARAMETER StubVoice
    Use stub implementations for ASR/TTS (no external dependencies)
.EXAMPLE
    .\run_all.ps1
    # Start full stack with default ports
.EXAMPLE
    .\run_all.ps1 -StubVoice -SkipFrontend
    # Start backend only with stub voice services
#>
param(
    [int]$AsrPort = 7001,
    [int]$TtsPort = 7002,
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000,
    [switch]$SkipFrontend,
    [switch]$StubVoice
)

$ErrorActionPreference = "Stop"
$script:jobs = @()
$script:projectRoot = (Get-Item $PSScriptRoot).Parent.Parent

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK]   " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERR]  " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] " -ForegroundColor Blue -NoNewline
    Write-Host $Message
}

function Get-LanIp {
    $interfaces = Get-NetIPAddress -AddressFamily IPv4 -PrefixOrigin Dhcp | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" }
    if ($interfaces) {
        return ($interfaces | Select-Object -First 1).IPAddress
    }
    return "127.0.0.1"
}

function Test-PortAvailable {
    param([int]$Port)
    try {
        $socket = New-Object System.Net.Sockets.TcpClient
        $socket.Connect("127.0.0.1", $Port)
        $socket.Close()
        return $false
    } catch {
        return $true
    }
}

function Test-PortInUse {
    param([int]$Port)
    $process = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
    return $null -ne $process
}

function Stop-AllJobs {
    Write-Info "Stopping all background jobs..."
    $script:jobs | ForEach-Object {
        try {
            Stop-Job -Job $_ -ErrorAction SilentlyContinue
            Remove-Job -Job $_ -ErrorAction SilentlyContinue
        } catch {
            Write-Warning "Failed to stop job: $_"
        }
    }
    $script:jobs = @()
}

function Start-AsrService {
    param([int]$Port)
    Write-Info "Starting ASR service on port $Port..."

    if (-not (Test-PortAvailable $Port)) {
        throw "Port $Port is already in use. Stop the existing service or use a different port."
    }

    $asrScript = @'
$ErrorActionPreference = "Stop"
$Port = $args[0]

$FastApiCode = @"
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import asyncio

app = FastAPI(title="ASR Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    engine: str
    latency_ms: int

class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration_seconds: float
    segments: Optional[list] = None

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "healthy", "engine": "whisper-stub", "latency_ms": 5}

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    content = await audio.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    
    await asyncio.sleep(0.1)
    
    return {
        "text": "This is a stub transcription. Replace with real ASR engine.",
        "language": "en",
        "duration_seconds": len(content) / 32000.0,
        "segments": []
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=$Port)
"@

    $tempFile = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '_asr.py'
    $FastApiCode | Set-Content -Path $tempFile -Encoding UTF8

    $job = Start-Job -ScriptBlock {
        param($Port, $ScriptPath)
        python $ScriptPath $Port
    } -ArgumentList $Port, $tempFile

    $script:jobs += $job
    $tempFile
}

function Start-TtsService {
    param([int]$Port)
    Write-Info "Starting TTS service on port $Port..."

    if (-not (Test-PortAvailable $Port)) {
        throw "Port $Port is already in use. Stop the existing service or use a different port."
    }

    $ttsScript = @'
$ErrorActionPreference = "Stop"
$Port = $args[0]

$FastApiCode = @"
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from enum import Enum
import uvicorn
import asyncio

class AudioFormat(str, Enum):
    MP3 = "mp3"
    WAV = "wav"

class SpeakRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    speed: float = 1.0
    format: AudioFormat = AudioFormat.MP3

class SpeakResponse(BaseModel):
    audio_content: bytes
    format: str
    duration_seconds: float

class HealthResponse(BaseModel):
    status: str
    engine: str
    latency_ms: int

app = FastAPI(title="TTS Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "healthy", "engine": "edge-tts-stub", "latency_ms": 10}

@app.post("/speak", response_model=SpeakResponse)
async def speak(request: SpeakRequest):
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(request.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
    
    await asyncio.sleep(0.1)
    
    stub_audio = b'\x52\x49\x46\x46\x24\x00\x00\x00\x57\x41\x56\x45\x66\x6d\x74\x20\x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00\x64\x61\x74\x61\x00\x00\x00\x00'
    
    return {
        "audio_content": stub_audio,
        "format": request.format.value,
        "duration_seconds": len(request.text) * 0.1
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=$Port)
"@

    $tempFile = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '_tts.py'
    $FastApiCode | Set-Content -Path $tempFile -Encoding UTF8

    $job = Start-Job -ScriptBlock {
        param($Port, $ScriptPath)
        python $ScriptPath $Port
    } -ArgumentList $Port, $tempFile

    $script:jobs += $job
    $tempFile
}

function Start-BackendService {
    param([int]$Port)
    Write-Info "Starting DeepTutor backend on port $Port..."

    if (-not (Test-PortAvailable $Port)) {
        throw "Port $Port is already in use. Stop the existing service or use a different port."
    }

    $job = Start-Job -ScriptBlock {
        param($Port, $ProjectRoot)
        $env:PYTHONPATH = $ProjectRoot
        python -m uvicorn "src.api.main:app" --host "0.0.0.0" --port $Port --reload --log-level "info"
    } -ArgumentList $Port, $script:projectRoot

    $script:jobs += $job
}

function Start-FrontendService {
    param([int]$Port)
    Write-Info "Starting frontend dev server on port $Port..."

    if (-not (Test-PortAvailable $Port)) {
        throw "Port $Port is already in use. Stop the existing service or use a different port."
    }

    $webDir = Join-Path $script:projectRoot "web"
    $job = Start-Job -ScriptBlock {
        param($Port, $WebDir)
        Set-Location $WebDir
        npm run dev
    } -ArgumentList $Port, $webDir

    $script:jobs += $job
}

function Wait-ForService {
    param(
        [string]$Name,
        [string]$Url,
        [int]$TimeoutSeconds = 30
    )
    Write-Info "Waiting for $Name to be ready at $Url..."
    $startTime = Get-Date
    $maxTime = $startTime.AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $maxTime) {
        try {
            $response = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 2
            if ($response.status -eq "healthy" -or $response.status -eq "ok") {
                Write-Success "$Name is healthy"
                return $true
            }
        } catch {
        }
        Start-Sleep -Seconds 1
    }

    Write-Warning "$Name health check timed out (may still be starting)"
    return $false
}

function Write-ServiceUrls {
    $lanIp = Get-LanIp

    Write-Header "DeepTutor Development Stack"

    Write-Host "Services:" -ForegroundColor White
    Write-Host ""
    Write-Host "  DeepTutor Backend:" -ForegroundColor Yellow
    Write-Host "    Local:    http://localhost:$BackendPort" -ForegroundColor Gray
    Write-Host "    LAN:      http://$lanIp`:$BackendPort" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  ASR Service:" -ForegroundColor Yellow
    Write-Host "    Local:    http://localhost:$AsrPort" -ForegroundColor Gray
    Write-Host "    LAN:      http://$lanIp`:$AsrPort" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  TTS Service:" -ForegroundColor Yellow
    Write-Host "    Local:    http://localhost:$TtsPort" -ForegroundColor Gray
    Write-Host "    LAN:      http://$lanIp`:$TtsPort" -ForegroundColor Gray
    Write-Host ""

    if (-not $SkipFrontend) {
        Write-Host "  Frontend (Next.js):" -ForegroundColor Yellow
        Write-Host "    Local:    http://localhost:$FrontendPort" -ForegroundColor Gray
        Write-Host "    LAN:      http://$lanIp`:$FrontendPort" -ForegroundColor Gray
        Write-Host ""
    }

    Write-Host "API Documentation:" -ForegroundColor White
    Write-Host "  http://localhost:$BackendPort/docs" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Voice Gateway:" -ForegroundColor White
    Write-Host "  POST http://localhost:$AsrPort/transcribe" -ForegroundColor Gray
    Write-Host "  POST http://localhost:$TtsPort/speak" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Press Ctrl+C to stop all services." -ForegroundColor Magenta
    Write-Host ""
}

try {
    Write-Header "DeepTutor Development Stack"

    Write-Info "Project root: $script:projectRoot"
    Write-Info "ASR port: $AsrPort | TTS port: $TtsPort | Backend port: $BackendPort"
    if ($StubVoice) {
        Write-Warning "Using stub implementations for ASR/TTS (no external dependencies)"
    }
    Write-Host ""

    Write-Info "Checking port availability..."
    $ports = @($AsrPort, $TtsPort, $BackendPort)
    if (-not $SkipFrontend) {
        $ports += $FrontendPort
    }

    foreach ($port in $ports) {
        if (-not (Test-PortAvailable $port)) {
            $proc = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
            $procName = if ($proc.OwningProcess) { (Get-Process -Id $proc.OwningProcess -ErrorAction SilentlyContinue).Name } else { "unknown" }
            throw "Port $port is in use by process '$procName' (PID: $($proc.OwningProcess)). Please stop the existing service or choose a different port."
        }
    }
    Write-Success "All ports are available"

    Write-Host ""
    Write-Info "Starting services..."

    Start-AsrService -Port $AsrPort
    Start-TtsService -Port $TtsPort
    Start-BackendService -Port $BackendPort

    if (-not $SkipFrontend) {
        Start-FrontendService -Port $FrontendPort
    }

    Write-Host ""
    Write-Info "Waiting for services to initialize..."

    Start-Sleep -Seconds 3

    Wait-ForService -Name "ASR" -Url "http://localhost:$AsrPort/health" -TimeoutSeconds 15
    Wait-ForService -Name "TTS" -Url "http://localhost:$TtsPort/health" -TimeoutSeconds 15
    Wait-ForService -Name "Backend" -Url "http://localhost:$BackendPort/api/v1/system/health" -TimeoutSeconds 30

    Write-ServiceUrls

    Write-Info "All services started. Press Ctrl+C to stop."

    while ($true) {
        Start-Sleep -Seconds 5

        $runningJobs = @()
        foreach ($job in $script:jobs) {
            $jobState = Get-Job -Id $job.Id -ErrorAction SilentlyContinue
            if ($null -eq $jobState -or $jobState.State -notmatch "Running") {
                $runningJobs += $job
            }
        }

        if ($runningJobs.Count -gt 0) {
            Write-Error "One or more services have stopped unexpectedly!"
            Get-Job | Where-Object State -ne "Running" | ForEach-Object {
                $output = Receive-Job -Job $_ -ErrorAction SilentlyContinue
                if ($output) {
                    Write-Host "Job output: $output" -ForegroundColor Red
                }
            }
            break
        }
    }

} catch {
    Write-Error "Fatal error: $_"
    Write-Error $_.ScriptStackTrace
    Stop-AllJobs
    exit 1
} finally {
    Stop-AllJobs
    Write-Host ""
    Write-Info "All services stopped."
}
