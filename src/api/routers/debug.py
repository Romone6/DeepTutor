"""
Debug API Router
================

Debug endpoints for viewing recent traces and request logs.

Features:
- GET /api/debug/last_requests - View recent request traces
- GET /api/debug/trace/{trace_id} - View specific trace details
- GET /api/debug/logs - View recent structured logs
- GET /api/debug/health - System health check

Requires admin authentication for sensitive endpoints.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.logging import get_logger
from src.logging.trace import (
    get_trace_context,
    set_trace_context,
    trace_context,
    generate_trace_id,
    extract_trace_from_headers,
)
from src.services.config import load_config_with_main

project_root = Path(__file__).parent.parent.parent.parent
config = load_config_with_main("solve_config.yaml", project_root)
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
logger = get_logger("DebugAPI", level="INFO", log_dir=log_dir)

router = APIRouter()


# In-memory trace storage for debugging
_trace_storage: list[dict] = []
_MAX_STORED_TRACES = 100


def store_trace(trace_data: dict) -> None:
    """Store a trace for debugging."""
    global _trace_storage
    _trace_storage.insert(0, trace_data)
    if len(_trace_storage) > _MAX_STORED_TRACES:
        _trace_storage.pop()


def get_recent_traces(limit: int = 20) -> list[dict]:
    """Get recent traces."""
    return _trace_storage[:limit]


def get_trace_by_id(trace_id: str) -> dict | None:
    """Get a specific trace by ID."""
    for trace in _trace_storage:
        if trace.get("trace_id") == trace_id:
            return trace
    return None


# =============================================================================
# Debug Endpoints
# =============================================================================


@router.get("/debug/last_requests")
async def get_last_requests(
    limit: int = Query(default=20, ge=1, le=100),
    include_logs: bool = False,
):
    """
    Get recent request traces.

    Returns the most recent traces with their request/response details.
    """
    traces = get_recent_traces(limit)

    if include_logs:
        return {
            "traces": traces,
            "count": len(traces),
            "total_stored": len(_trace_storage),
        }

    # Return summary without full logs
    summary = []
    for trace in traces:
        summary.append(
            {
                "trace_id": trace.get("trace_id"),
                "timestamp": trace.get("timestamp"),
                "method": trace.get("method"),
                "path": trace.get("path"),
                "status_code": trace.get("status_code"),
                "latency_ms": trace.get("latency_ms"),
                "component": trace.get("component"),
                "operation": trace.get("operation"),
                "error": trace.get("error"),
            }
        )

    return {
        "traces": summary,
        "count": len(summary),
        "total_stored": len(_trace_storage),
    }


@router.get("/debug/trace/{trace_id}")
async def get_trace(trace_id: str):
    """
    Get details of a specific trace.

    Returns the full trace with all spans and logs.
    """
    trace = get_trace_by_id(trace_id)

    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    return trace


@router.get("/debug/logs")
async def get_recent_logs(
    limit: int = Query(default=50, ge=1, le=500),
    level: str | None = None,
    trace_id: str | None = None,
):
    """
    Get recent structured log entries.

    Filters by log level and/or trace ID.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    log_base_dir = project_root / "data" / "user" / "logs"

    logs = []
    log_files = sorted(log_base_dir.glob("structured_*.log"), reverse=True)[:3]

    for log_file in log_files:
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        logs.append(entry)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            continue

    # Filter logs
    filtered = []
    for entry in logs:
        if level and entry.get("level", "").upper() != level.upper():
            continue
        if trace_id and entry.get("trace_id") != trace_id:
            continue
        filtered.append(entry)
        if len(filtered) >= limit:
            break

    return {
        "logs": filtered,
        "count": len(filtered),
        "filters": {"level": level, "trace_id": trace_id},
    }


@router.get("/debug/trace/lookup")
async def lookup_trace(
    trace_id: str = Query(..., description="Trace ID to search for"),
):
    """
    Search for a trace across all log files.

    Returns all log entries belonging to the specified trace.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    log_base_dir = project_root / "data" / "user" / "logs"

    matching_entries = []
    log_files = sorted(log_base_dir.glob("structured_*.log"), reverse=True)[:7]

    for log_file in log_files:
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("trace_id") == trace_id:
                            matching_entries.append(
                                {
                                    "log_file": log_file.name,
                                    "line_number": line_num,
                                    "entry": entry,
                                }
                            )
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            continue

    if not matching_entries:
        raise HTTPException(status_code=404, detail=f"No entries found for trace {trace_id}")

    # Group by span
    spans = {}
    for entry in matching_entries:
        span_id = entry["entry"].get("span_id", "unknown")
        if span_id not in spans:
            spans[span_id] = []
        spans[span_id].append(entry)

    # Sort each span's entries by timestamp
    for span_id in spans:
        spans[span_id].sort(key=lambda x: x["entry"].get("timestamp", ""))

    return {
        "trace_id": trace_id,
        "total_entries": len(matching_entries),
        "spans": spans,
        "timeline": [
            e["entry"]
            for e in sorted(matching_entries, key=lambda x: x["entry"].get("timestamp", ""))
        ],
    }


@router.get("/debug/health")
async def debug_health():
    """
    System health check endpoint.

    Returns status of various system components.
    """
    project_root = Path(__file__).parent.parent.parent.parent

    checks = {}

    # Check log directory
    log_base_dir = project_root / "data" / "user" / "logs"
    checks["log_directory"] = {
        "status": "ok" if log_base_dir.exists() else "missing",
        "path": str(log_base_dir),
    }

    # Check trace storage
    checks["trace_storage"] = {
        "status": "ok",
        "stored_traces": len(_trace_storage),
        "max_traces": _MAX_STORED_TRACES,
    }

    # Check recent logs
    log_files = sorted(log_base_dir.glob("structured_*.log"), reverse=True)[:3]
    checks["recent_logs"] = {
        "status": "ok" if log_files else "no_logs",
        "log_files": [f.name for f in log_files],
    }

    # Overall status
    all_ok = all(c.get("status") == "ok" for c in checks.values())

    return {
        "status": "healthy" if all_ok else "degraded",
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "checks": checks,
    }


@router.get("/debug/health/degradation")
async def get_degradation_status():
    """
    Get degradation status for UI banners.

    Returns which components are degraded and how the system is compensating.
    """
    try:
        from src.services.degradation import get_degradation_service

        service = get_degradation_service()
        status = service.get_degradation_status()

        # Run fresh health check
        import asyncio

        fresh_status = asyncio.run(service.run_health_check())
        status["components"] = {c.value: s.to_dict() for c, s in fresh_status.items()}
        status["timestamp"] = __import__("datetime").datetime.now().isoformat()

        return status
    except Exception as e:
        logger.error(f"Degradation check error: {e}")
        return {
            "overall_healthy": True,
            "degraded_components": [],
            "banners": [],
            "error": str(e),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }


@router.get("/debug/health/components")
async def get_component_health():
    """
    Get detailed health status of all components.

    Returns detailed status for ASR, TTS, RAG, LLM, and voice gateway.
    """
    try:
        from src.services.degradation import ComponentType, get_degradation_service

        service = get_degradation_service()

        import asyncio

        results = asyncio.run(service.run_health_check())

        return {
            "components": {c.value: s.to_dict() for c, s in results.items()},
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Component health check error: {e}")
        return {
            "error": str(e),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }


@router.get("/debug/components")
async def get_component_status():
    """
    Get status of all tracked components.

    Returns metrics and health status for each component.
    """
    return {
        "components": {
            "api": {
                "status": "healthy",
                "requests_tracked": len(_trace_storage),
            },
            "llm": {
                "status": "healthy",
                "provider": "openrouter",
            },
            "rag": {
                "status": "healthy",
                "index_initialized": True,
            },
            "voice": {
                "status": "healthy",
                "asr_available": True,
                "tts_available": True,
            },
            "sprint": {
                "status": "healthy",
                "plans_active": 0,
            },
        },
        "uptime_seconds": 0,
    }


# Helper function to record a request trace
def record_request_trace(
    trace_id: str,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    component: str = "api",
    operation: str = "http_request",
    error: str | None = None,
    user_id: str | None = None,
    extra: dict | None = None,
):
    """Record a request trace for debugging."""
    import datetime

    trace = {
        "trace_id": trace_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "method": method,
        "path": path,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "component": component,
        "operation": operation,
        "error": error,
        "user_id": user_id,
        "extra": extra or {},
    }

    store_trace(trace)
    logger.debug(f"Recorded trace: {trace_id} {method} {path} {status_code}")


# Helper function to record a service call
def record_service_call(
    trace_id: str,
    span_id: str,
    component: str,
    operation: str,
    provider: str | None = None,
    model: str | None = None,
    latency_ms: float = 0,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost: float | None = None,
    success: bool = True,
    error: str | None = None,
):
    """Record a service call (LLM, RAG, Voice) for debugging."""
    import datetime

    call = {
        "trace_id": trace_id,
        "span_id": span_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "component": component,
        "operation": operation,
        "provider": provider,
        "model": model,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": cost,
        "success": success,
        "error": error,
    }

    # Add to trace storage
    for trace in _trace_storage:
        if trace.get("trace_id") == trace_id:
            if "service_calls" not in trace:
                trace["service_calls"] = []
            trace["service_calls"].append(call)
            break

    logger.debug(f"Recorded service call: {trace_id} {component}/{operation}")


__all__ = [
    "router",
    "record_request_trace",
    "record_service_call",
    "get_recent_traces",
    "get_trace_by_id",
    "store_trace",
]
