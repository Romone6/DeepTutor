"""
Middleware for trace context propagation.
Injects trace ID on request ingress and measures latency.
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.logging.trace import (
    generate_trace_id,
    set_trace_context,
    extract_trace_from_headers,
    TRACE_HEADER,
    PARENT_TRACE_HEADER,
    SPAN_HEADER,
)


class TraceContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    1. Extracts or generates trace ID from request headers
    2. Sets the trace context for the duration of the request
    3. Measures request latency
    4. Adds trace headers to response
    """

    def __init__(self, app, exclude_paths: list[str] | None = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/docs", "/redoc", "/openapi.json", "/health"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip trace injection for excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        # Extract trace context from headers or generate new
        trace_id = request.headers.get(TRACE_HEADER)
        parent_span_id = request.headers.get(PARENT_TRACE_HEADER)
        span_id = request.headers.get(SPAN_HEADER)

        if not trace_id:
            trace_id = generate_trace_id()

        # Generate a span ID for this request
        if not span_id:
            from src.logging.trace import generate_span_id

            span_id = generate_span_id()

        # Extract operation from request
        operation = f"{request.method.lower()}_{request.url.path.strip('/').replace('/', '_').replace('-', '_')}"

        # Set trace context for this request
        import contextvars
        from src.logging.trace import _trace_context

        token = _trace_context.set(
            {
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "component": "api",
                "operation": operation,
                "start_time": time.time(),
            }
        )

        try:
            # Process request and measure latency
            start_time = time.time()
            response = await call_next(request)
            latency_ms = (time.time() - start_time) * 1000

            # Add trace headers to response
            response.headers[TRACE_HEADER] = trace_id
            response.headers[SPAN_HEADER] = span_id
            response.headers["X-Request-Latency-Ms"] = f"{latency_ms:.2f}"

            return response
        finally:
            # Clear trace context
            _trace_context.reset(token)


def add_trace_middleware(app, exclude_paths: list[str] | None = None):
    """Add trace middleware to FastAPI app."""
    app.add_middleware(TraceContextMiddleware, exclude_paths=exclude_paths)
