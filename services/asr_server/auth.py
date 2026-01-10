"""Authentication and Security Middleware for Voice Services.

Provides:
- Bearer token authentication
- CORS origin validation
- Rate limiting (in-memory, per-IP)

Usage:
    from services.asr_server.auth import AuthMiddleware, RateLimitMiddleware

    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
"""

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AuthConfig:
    """Authentication configuration."""

    token: Optional[str] = None
    token_env_var: str = "VOICE_TOKEN"
    header_name: str = "Authorization"
    scheme: str = "Bearer"

    @classmethod
    def from_env(cls) -> "AuthConfig":
        """Load config from environment variables."""
        return cls(token=os.getenv(cls.token_env_var, None))


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    window_seconds: int = 60
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """Load config from environment variables."""
        rpm = os.getenv("VOICE_RATE_LIMIT_RPM", None)
        return cls(
            requests_per_minute=int(rpm) if rpm else 60,
            enabled=os.getenv("VOICE_RATE_LIMIT_ENABLED", "true").lower() == "true",
        )


@dataclass
class CORSConfig:
    """CORS configuration."""

    allowed_origins: list = field(default_factory=list)
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "CORSConfig":
        """Load config from environment variables."""
        origins = os.getenv("VOICE_ALLOWED_ORIGINS", "")
        origin_list = [o.strip() for o in origins.split(",") if o.strip()]
        return cls(
            allowed_origins=origin_list if origin_list else ["*"],
            enabled=os.getenv("VOICE_CORS_ENABLED", "true").lower() == "true",
        )


# =============================================================================
# Token Auth Middleware
# =============================================================================


class AuthMiddleware:
    """
    Bearer token authentication middleware.

    If VOICE_TOKEN is set, all requests must include a valid Bearer token.
    If not set, authentication is bypassed (for local development).
    """

    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig.from_env()

    async def __call__(self, scope, receive, send):
        """Process request through auth middleware."""
        if self.config.token is None:
            # No token configured, skip authentication
            await self.app(scope, receive, send)
            return

        # Create a custom scope with request info
        async def asgi(receive, send):
            async def receive_wrapper():
                message = await receive()
                return message

            # Process the request
            await self._process_request(scope, receive_wrapper, send)

        await self.app(scope, receive, asgi)

    async def _process_request(self, scope, receive, send):
        """Process authenticated request."""
        # Extract Authorization header from scope
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()

        if not auth_header:
            response = JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "unauthorized",
                    "message": "Missing Authorization header",
                },
            )
            await response(receive, send)
            return

        # Validate token
        expected = f"{self.config.scheme} {self.config.token}"
        if auth_header != expected:
            response = JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "unauthorized",
                    "message": "Invalid or expired token",
                },
            )
            await response(receive, send)
            return

        # Token is valid, continue to app
        await self.app(scope, receive, send)

    def set_app(self, app):
        """Set the ASGI application to call after auth."""
        self.app = app


def create_auth_dependency(config: Optional[AuthConfig] = None):
    """
    Create FastAPI dependency for token authentication.

    Usage:
        from fastapi import Depends
        from services.asr_server.auth import create_auth_dependency

        auth_dep = create_auth_dependency()

        @app.get("/protected", dependencies=[Depends(auth_dep)])
        async def protected_endpoint():
            return {"message": "Access granted"}
    """
    config = config or AuthConfig.from_env()

    async def verify_token(request: Request):
        """Verify bearer token from Authorization header."""
        if config.token is None:
            # No token required
            return

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                detail="Missing Authorization header",
            )

        expected = f"{config.scheme} {config.token}"
        if auth_header != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                detail="Invalid or expired token",
            )

    return verify_token


# =============================================================================
# Rate Limiting Middleware
# =============================================================================


class RateLimitMiddleware:
    """
    In-memory rate limiting middleware.

    Tracks requests per IP address and enforces rate limits.
    Uses a sliding window algorithm for accurate limiting.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig.from_env()
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def __call__(self, scope, receive, send):
        """Process request through rate limiting."""
        if not self.config.enabled:
            await self.app(scope, receive, send)
            return

        # Extract client IP
        headers = dict(scope.get("headers", []))
        client_ip = self._get_client_ip(scope, headers)

        # Check rate limit
        now = time.time()
        window_start = now - self.config.window_seconds

        # Clean old requests
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > window_start
        ]

        # Check if over limit
        if len(self._requests[client_ip]) >= self.config.requests_per_minute:
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limited",
                    "message": f"Rate limit exceeded. Max {self.config.requests_per_minute} requests per minute.",
                    "retry_after": int(self.config.window_seconds),
                },
                headers={"Retry-After": str(self.config.window_seconds)},
            )
            await response(receive, send)
            return

        # Record request
        self._requests[client_ip].append(now)

        # Continue
        await self.app(scope, receive, send)

    def _get_client_ip(self, scope: dict, headers: dict) -> str:
        """Extract client IP from scope."""
        # Check X-Forwarded-For header (for proxied requests)
        forwarded = headers.get(b"x-forwarded-for", b"").decode()
        if forwarded:
            # Take first IP in chain
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = headers.get(b"x-real-ip", b"").decode()
        if real_ip:
            return real_ip

        # Fall back to client address from scope
        client = scope.get("client")
        if client:
            return client[0]

        return "unknown"

    def set_app(self, app):
        """Set the ASGI application to call after rate limiting."""
        self.app = app


def create_rate_limit_dependency(config: Optional[RateLimitConfig] = None):
    """
    Create FastAPI dependency for rate limiting.

    Note: This is less effective than middleware but can be used
    for specific endpoints.
    """
    from fastapi import Request, HTTPException, status

    config = config or RateLimitConfig.from_env()

    async def rate_limit(request: Request):
        """Check rate limit for request."""
        if not config.enabled:
            return

        client_ip = request.client.host if request.client else "unknown"

        now = time.time()
        window_start = now - config.window_seconds

        # This is a simplified version; middleware is recommended
        # Store in app state for sharing across requests
        if not hasattr(request.state, "_rate_limiter"):
            request.state._rate_limiter = defaultdict(list)

        rate_limiter = request.state._rate_limiter
        rate_limiter[client_ip] = [
            t for t in rate_limiter[client_ip] if t > window_start
        ]

        if len(rate_limiter[client_ip]) >= config.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {config.requests_per_minute} requests per minute.",
                headers={"Retry-After": str(config.window_seconds)},
            )

        rate_limiter[client_ip].append(now)

    return rate_limit


# =============================================================================
# CORS Middleware
# =============================================================================


def create_cors_middleware(config: Optional[CORSConfig] = None):
    """
    Create custom CORS middleware with origin validation.

    Usage:
        from services.asr_server.auth import create_cors_middleware

        app.add_middleware(create_cors_middleware())
    """
    config = config or CORSConfig.from_env()

    class ValidatedCORSMiddleware:
        def __init__(self, app):
            self.app = app
            self.allowed_origins = set(config.allowed_origins)
            self.enabled = config.enabled

        async def __call__(self, scope, receive, send):
            if not self.enabled:
                await self.app(scope, receive, send)
                return

            headers = dict(scope.get("headers", []))
            origin = headers.get(b"origin", b"").decode()

            # Handle preflight requests
            if scope.get("method") == "OPTIONS":
                await self._handle_preflight(scope, origin, send)
                return

            # Validate origin for regular requests
            if origin and "*" not in self.allowed_origins:
                if origin not in self.allowed_origins:
                    response = JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={
                            "error": "forbidden",
                            "message": f"Origin '{origin}' not allowed",
                        },
                    )
                    await response(receive, send)
                    return

            # Continue with request
            await self.app(scope, receive, send)

        async def _handle_preflight(self, scope, origin, send):
            """Handle CORS preflight (OPTIONS) request."""
            if "*" in self.allowed_origins:
                allow_origin = "*"
            elif origin and origin in self.allowed_origins:
                allow_origin = origin
            else:
                response = JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"error": "forbidden", "message": "Origin not allowed"},
                )
                await response(receive, send)
                return

            # Build preflight response
            response = JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": "OK"},
            )

            # Add CORS headers
            if hasattr(response, "headers"):
                existing = dict(response.headers)
                existing.update(
                    {
                        "Access-Control-Allow-Origin": allow_origin,
                        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                        "Access-Control-Allow-Headers": "Authorization, Content-Type",
                        "Access-Control-Max-Age": "86400",
                    }
                )

            await response(receive, send)

    return ValidatedCORSMiddleware


# =============================================================================
# Security Headers Middleware
# =============================================================================


class SecurityHeadersMiddleware:
    """Add security headers to all responses."""

    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'",
    }

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        """Add security headers to response."""
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                headers = dict(message.get("headers", []))
                for name, value in self.SECURITY_HEADERS.items():
                    headers[name.encode()] = value.encode()
                message = {**message, "headers": list(headers.items())}
            await send(message)

        await self.app(scope, receive, send_wrapper)


# =============================================================================
# Factory Functions
# =============================================================================


def apply_security_middleware(
    app, auth_config=None, cors_config=None, rate_limit_config=None
):
    """
    Apply all security middleware to a FastAPI app.

    Args:
        app: FastAPI application
        auth_config: AuthConfig instance (optional)
        cors_config: CORSConfig instance (optional)
        rate_limit_config: RateLimitConfig instance (optional)
    """
    from fastapi.middleware.cors import CORSMiddleware

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # CORS
    if cors_config is None:
        cors_config = CORSConfig.from_env()

    if cors_config.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Rate limiting
    if rate_limit_config is None:
        rate_limit_config = RateLimitConfig.from_env()

    if rate_limit_config.enabled:
        # Add rate limit middleware
        middleware = RateLimitMiddleware(rate_limit_config)
        app.add_middleware(lambda app: middleware)

    return app
