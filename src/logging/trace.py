"""
Trace Context Utilities
=======================

Utilities for generating and propagating trace IDs across the application.

Features:
- Trace ID generation (UUID-based)
- Context management (thread-local storage)
- Header propagation for HTTP requests
- Secure secret filtering in logs

Usage:
    from src.logging.trace import get_trace_id, set_trace_id, create_trace_context

    # At request ingress
    trace_id = create_trace_id()
    set_trace_id(trace_id)

    # In logs
    log_with_trace(logger, "Processing request", trace_id=trace_id)

    # Propagate to downstream services
    headers = get_trace_headers(trace_id)
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional


# Thread-local storage for trace context
_trace_context = threading.local()


# Header names for trace propagation
TRACE_HEADER = "X-Trace-ID"
PARENT_TRACE_HEADER = "X-Parent-Trace-ID"
SPAN_HEADER = "X-Span-ID"


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid.uuid4())


def generate_span_id() -> str:
    """Generate a unique span ID."""
    return secrets.token_hex(8)


@dataclass
class TraceContext:
    """
    Trace context containing all tracing information for a request.
    """

    trace_id: str
    span_id: str = field(default_factory=generate_span_id)
    parent_span_id: Optional[str] = None
    request_path: Optional[str] = None
    request_method: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    component: str = "unknown"
    operation: str = "unknown"
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        elapsed_ms = (time.time() - self.start_time) * 1000
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "request_path": self.request_path,
            "request_method": self.request_method,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "elapsed_ms": round(elapsed_ms, 2),
            "component": self.component,
            "operation": self.operation,
            "tags": self.tags,
        }

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP headers for propagation."""
        headers = {TRACE_HEADER: self.trace_id, SPAN_HEADER: self.span_id}
        if self.parent_span_id:
            headers[PARENT_TRACE_HEADER] = self.parent_span_id
        return headers


def get_trace_context() -> Optional[TraceContext]:
    """Get the current trace context from thread-local storage."""
    return getattr(_trace_context, "context", None)


def set_trace_context(context: TraceContext) -> None:
    """Set the trace context in thread-local storage."""
    _trace_context.context = context


def clear_trace_context() -> None:
    """Clear the trace context from thread-local storage."""
    if hasattr(_trace_context, "context"):
        del _trace_context.context


@contextmanager
def trace_context(
    trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    component: str = "unknown",
    operation: str = "unknown",
):
    """
    Context manager for trace context.

    Usage:
        with trace_context(component="API", operation="chat") as ctx:
            log_info("Processing chat request", trace_id=ctx.trace_id)
            # ... do work ...
    """
    context = TraceContext(
        trace_id=trace_id or generate_trace_id(),
        parent_span_id=parent_span_id,
        component=component,
        operation=operation,
    )
    set_trace_context(context)
    try:
        yield context
    finally:
        clear_trace_context()


def get_trace_id() -> Optional[str]:
    """Get the current trace ID."""
    context = get_trace_context()
    return context.trace_id if context else None


def set_trace_id(trace_id: str) -> None:
    """Set the current trace ID."""
    context = get_trace_context()
    if context:
        context.trace_id = trace_id
    else:
        set_trace_context(TraceContext(trace_id=trace_id))


def get_trace_headers(trace_id: Optional[str] = None) -> dict[str, str]:
    """Get headers for propagating trace to downstream services."""
    context = get_trace_context()
    if context:
        return context.to_headers()
    if trace_id:
        return {TRACE_HEADER: trace_id, SPAN_HEADER: generate_span_id()}
    return {}


def extract_trace_from_headers(
    headers: dict[str, str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract trace information from HTTP headers.

    Returns:
        (trace_id, parent_span_id, span_id)
    """
    trace_id = headers.get(TRACE_HEADER) or headers.get("x-trace-id")
    parent_span_id = headers.get(PARENT_TRACE_HEADER) or headers.get("x-parent-trace-id")
    span_id = headers.get(SPAN_HEADER) or headers.get("x-span-id") or generate_span_id()
    return trace_id, parent_span_id, span_id


# Secret patterns for filtering sensitive data
_SECRET_PATTERNS = [
    re.compile(r"(api[_-]?key|apikey)[\s=:\"']+([^\s\"']+)", re.IGNORECASE),
    re.compile(r"(token|auth)[\s=:\"']+([^\s\"']+)", re.IGNORECASE),
    re.compile(r"password[\s=:\"']+([^\s\"']+)", re.IGNORECASE),
    re.compile(r"secret[\s=:\"']+([^\s\"']+)", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),  # OpenAI keys
    re.compile(r"[A-Za-z0-9]{32,}", re.IGNORECASE),  # Generic long secrets
]


def filter_secrets(text: str) -> str:
    """
    Filter secrets from text to prevent logging sensitive data.

    Usage:
        safe_message = filter_secrets(f"API key: sk-abc123...")
        # Returns: "API key: [REDACTED]"
    """
    filtered = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            filtered = pattern.sub(r"\1: [REDACTED]", filtered)
        else:
            filtered = pattern.sub("[REDACTED]", filtered)
    return filtered


def filter_dict_secrets(data: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively filter secrets from a dictionary.

    Usage:
        safe_data = filter_dict_secrets({
            "api_key": "sk-abc123...",
            "user": "john"
        })
        # Returns: {"api_key": "[REDACTED]", "user": "john"}
    """
    sensitive_keys = {"api_key", "apikey", "token", "password", "secret", "auth", "authorization"}

    result = {}
    for key, value in data.items():
        if key.lower() in sensitive_keys:
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = filter_dict_secrets(value)
        elif isinstance(value, list):
            result[key] = filter_list_secrets(value)
        else:
            result[key] = value
    return result


def filter_list_secrets(data: list[Any]) -> list[Any]:
    """Recursively filter secrets from a list."""
    result = []
    for item in data:
        if isinstance(item, dict):
            result.append(filter_dict_secrets(item))
        elif isinstance(item, list):
            result.append(filter_list_secrets(item))
        else:
            result.append(item)
    return result


# Request/response logging helpers


@dataclass
class RequestLogEntry:
    """Structured log entry for HTTP requests."""

    trace_id: Optional[str]
    span_id: Optional[str]
    timestamp: str
    level: str
    component: str
    operation: str
    method: str
    path: str
    status_code: Optional[int]
    latency_ms: float
    request_size: Optional[int]
    response_size: Optional[int]
    user_id: Optional[str]
    error: Optional[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "level": self.level,
            "component": self.component,
            "operation": self.operation,
            "http": {
                "method": self.method,
                "path": self.path,
                "status_code": self.status_code,
                "latency_ms": self.latency_ms,
                "request_size": self.request_size,
                "response_size": self.response_size,
            },
            "user_id": self.user_id,
            "error": self.error,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


def create_request_log_entry(
    trace_id: Optional[str],
    span_id: Optional[str],
    method: str,
    path: str,
    status_code: Optional[int],
    latency_ms: float,
    request_size: Optional[int] = None,
    response_size: Optional[int] = None,
    user_id: Optional[str] = None,
    error: Optional[str] = None,
    component: str = "api",
    operation: str = "http_request",
) -> RequestLogEntry:
    """Create a structured request log entry."""
    from datetime import datetime

    level = "error" if status_code and status_code >= 400 else "info"

    return RequestLogEntry(
        trace_id=trace_id,
        span_id=span_id,
        timestamp=datetime.now().isoformat(),
        level=level,
        component=component,
        operation=operation,
        method=method,
        path=path,
        status_code=status_code,
        latency_ms=latency_ms,
        request_size=request_size,
        response_size=response_size,
        user_id=user_id,
        error=error,
    )


@dataclass
class ServiceCallEntry:
    """Structured log entry for service calls (LLM, RAG, etc.)."""

    trace_id: Optional[str]
    span_id: Optional[str]
    timestamp: str
    level: str
    component: str
    operation: str
    provider: Optional[str]
    model: Optional[str]
    latency_ms: float
    tokens_in: Optional[int]
    tokens_out: Optional[int]
    cost: Optional[float]
    success: bool
    error: Optional[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "level": self.level,
            "component": self.component,
            "operation": self.operation,
            "service": {
                "provider": self.provider,
                "model": self.model,
                "latency_ms": self.latency_ms,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "cost": self.cost,
            },
            "success": self.success,
            "error": self.error,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


def create_service_call_entry(
    trace_id: Optional[str],
    span_id: Optional[str],
    component: str,
    operation: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    latency_ms: float = 0,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    cost: Optional[float] = None,
    success: bool = True,
    error: Optional[str] = None,
) -> ServiceCallEntry:
    """Create a structured service call log entry."""
    from datetime import datetime

    level = "error" if not success else "debug"

    return ServiceCallEntry(
        trace_id=trace_id,
        span_id=span_id,
        timestamp=datetime.now().isoformat(),
        level=level,
        component=component,
        operation=operation,
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=cost,
        success=success,
        error=error,
    )


__all__ = [
    "generate_trace_id",
    "generate_span_id",
    "TraceContext",
    "get_trace_context",
    "set_trace_context",
    "clear_trace_context",
    "trace_context",
    "get_trace_id",
    "set_trace_id",
    "get_trace_headers",
    "extract_trace_from_headers",
    "filter_secrets",
    "filter_dict_secrets",
    "filter_list_secrets",
    "RequestLogEntry",
    "create_request_log_entry",
    "ServiceCallEntry",
    "create_service_call_entry",
    "TRACE_HEADER",
    "PARENT_TRACE_HEADER",
    "SPAN_HEADER",
]
