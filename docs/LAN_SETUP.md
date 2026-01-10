# MacBook Client Mode - LAN Setup Guide

This guide explains how to run the DeepTutor UI on your MacBook while connecting to a Windows PC backend over your local network.

## Overview

```
MacBook (Client)  <--LAN-->  Windows PC (Backend)
   :3000                    :8000 (DeepTutor API)
                              :7001 (ASR Service)
                              :7002 (TTS Service)
```

## Prerequisites

### On Windows PC
1. DeepTutor installed and running
2. Firewall configured to allow inbound connections
3. Known IP address or hostname

### On MacBook
1. DeepTutor UI installed
2. Modern web browser (Chrome, Firefox, Safari)

## Step 1: Configure Windows Firewall

On your Windows PC, you need to allow inbound connections to DeepTutor services:

### Option A: Using PowerShell (Admin)
```powershell
# Allow port 8000 (FastAPI backend)
New-NetFirewallRule -DisplayName "DeepTutor API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow

# Allow port 7001 (ASR service)
New-NetFirewallRule -DisplayName "DeepTutor ASR" -Direction Inbound -Protocol TCP -LocalPort 7001 -Action Allow

# Allow port 7002 (TTS service)
New-NetFirewallRule -DisplayName "DeepTutor TTS" -Direction Inbound -Protocol TCP -LocalPort 7002 -Action Allow
```

### Option B: Using GUI (Windows Defender Firewall)
1. Open **Windows Defender Firewall with Advanced Security** (`wf.msc`)
2. Click **Inbound Rules** → **New Rule...**
3. **Rule Type**: Port
4. **Protocol**: TCP
5. **Port**: 8000 (and 7001, 7002 if using voice features)
6. **Action**: Allow the connection
7. **Profile**: Select "Private" only (recommended for security)
8. **Name**: "DeepTutor API"

### Security Best Practices
- **Only allow on private networks** - Never open these ports on public networks
- **Use VPN if on different network** - If connecting from outside your home network
- **Disable when not in use** - Remove the firewall rules after testing

## Step 2: Find Your Windows PC's IP Address

### Method 1: Command Prompt
```cmd
ipconfig
```
Look for "IPv4 Address" under your active network adapter (typically `192.168.x.x` or `10.x.x.x`)

### Method 2: Settings
1. Open **Settings** → **Network & Internet**
2. Click **View your network properties**
3. Look for "IPv4 address"

### Method 3: PowerShell
```powershell
(Get-NetIPAddress | Where-Object {$_.AddressFamily -eq 'IPv4' -and $_.PrefixOrigin -eq 'Dhcp'}).IPAddress
```

## Step 3: Start DeepTutor on Windows

On your Windows PC, start DeepTutor:
```bash
# Start the backend (keep this terminal open)
python src/api/run_server.py

# In a separate terminal, start the web UI (optional, or use Mac)
python scripts/start_web.py
```

## Step 4: Configure MacBook Client

### Option A: Using the Settings UI (Recommended)
1. Open DeepTutor in your browser on MacBook
2. Navigate to **Settings** → **Remote Backend**
3. Enter the Windows PC URL, e.g.:
   - `http://192.168.1.100:8000` (using IP address)
   - `http://YOUR-PC-NAME.local:8000` (using hostname)
4. Click **Test Connection** to verify
5. Click **Connect to Remote Backend**

### Option B: Manual Configuration
Create/edit `.env.local` in the web directory:
```bash
NEXT_PUBLIC_API_BASE=http://192.168.1.100:8000
```

Then restart the web server.

## Step 5: Verify Connection

The connection test in Settings will check:
- **Backend API** (`/api/v1/health`) - Main API health
- **LLM** (`/api/v1/settings/env/test/`) - Language model connectivity
- **RAG** (`/api/v1/knowledge/list`) - Knowledge base access
- **TTS** (`/api/voice/health`) - Text-to-speech service
- **ASR** (`/api/voice/health`) - Speech recognition service

## Troubleshooting

### "Connection Refused"
- Verify Windows firewall rules are correctly configured
- Check that DeepTutor is running on Windows
- Ensure both devices are on the same network

### "CORS Error"
The backend includes CORS headers for cross-origin requests. If you see CORS errors:
1. Verify the backend is configured to allow your MacBook's IP
2. Check that the URL format matches exactly

### Slow Responses
LAN connections may have latency. First audio chunk should arrive within 1-2 seconds.

### Cannot Resolve Hostname
If using `YOUR-PC-NAME.local` doesn't work:
1. Use the IP address instead
2. Or add the Windows PC to your Mac's `/etc/hosts` file

### Finding Your PC's Hostname
On Windows:
```cmd
hostname
```

On Mac, edit `/etc/hosts`:
```bash
sudo nano /etc/hosts
```
Add line:
```
192.168.1.100  your-pc-name
```

## Network Requirements

| Requirement | Details |
|------------|---------|
| Port (API) | 8000 (TCP) |
| Port (ASR) | 7001 (TCP, optional) |
| Port (TTS) | 7002 (TCP, optional) |
| Protocol | HTTP (not HTTPS for local) |
| Network Type | Private LAN only |

## Security Considerations

1. **Private Networks Only** - Only use this setup on trusted home/work networks
2. **Firewall Rules** - Only open ports on Private profile
3. **Disable When Not in Use** - Remove or disable firewall rules after testing
4. **VPN for Remote Access** - If you need to connect from outside your network, use a VPN instead of opening ports publicly
5. **No Authentication** - The API currently has no authentication; this is acceptable on isolated LANs but not for internet-facing deployments

## Quick Reference

| Step | Action |
|------|--------|
| 1 | Get Windows IP: `ipconfig` |
| 2 | Open firewall ports 8000, 7001, 7002 |
| 3 | Start DeepTutor on Windows |
| 4 | On Mac: Settings → Remote Backend |
| 5 | Enter `http://<WINDOWS_IP>:8000` |
| 6 | Test connection |
| 7 | Use normally! |
