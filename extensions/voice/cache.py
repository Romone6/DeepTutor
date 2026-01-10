"""Voice Response Cache.

Simple LRU cache for TTS audio responses to reduce latency for repeated requests.

Cache Key: (text, voice_id, format)
Cache Value: (audio_bytes, content_type, timestamp)

Usage:
    from extensions.voice.cache import TTSCache, get_cache

    cache = get_cache()
    audio = cache.get("hello", "stub-male-1", "wav")
    if audio:
        return audio
    else:
        audio = synthesize_speech("hello", voice_id="stub-male-1")
        cache.set("hello", "stub-male-1", "wav", audio)
        return audio
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheEntry:
    """Cache entry with audio data."""

    audio_bytes: bytes
    content_type: str
    timestamp: float
    hits: int = 0


class TTSCache:
    """
    Thread-safe LRU cache for TTS audio responses.

    Features:
    - Configurable max size (default: 100 entries)
    - LRU eviction when full
    - Per-entry hit counting for monitoring
    - TTL-based expiration (default: 1 hour)
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 3600,
    ):
        """
        Initialize TTS cache.

        Args:
            max_size: Maximum number of entries in cache
            ttl_seconds: Time-to-live for entries (1 hour default)
        """
        self._cache: dict[str, CacheEntry] = {}
        self._order: list[str] = []  # For LRU ordering
        self._lock_cache = {}  # Per-key locks for thread safety
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._total_hits = 0
        self._total_misses = 0

    def _get_key(self, text: str, voice_id: str, format: str) -> str:
        """Generate cache key from parameters."""
        # Normalize inputs
        text_norm = text.strip().lower()
        voice_norm = (voice_id or "default").strip().lower()
        format_norm = format.lower() if format else "wav"

        # Create deterministic key
        key_content = f"{text_norm}|{voice_norm}|{format_norm}"
        return hashlib.sha256(key_content.encode()).hexdigest()[:32]

    def _acquire_lock(self, key: str) -> None:
        """Acquire lock for a specific key."""
        if key not in self._lock_cache:
            self._lock_cache[key] = __import__("threading").Lock()
        self._lock_cache[key].acquire()

    def _release_lock(self, key: str) -> None:
        """Release lock for a specific key."""
        if key in self._lock_cache:
            self._lock_cache[key].release()

    def get(self, text: str, voice_id: str, format: str) -> Optional[tuple[bytes, str]]:
        """
        Get audio from cache.

        Args:
            text: Input text
            voice_id: Voice identifier
            format: Audio format (wav, mp3)

        Returns:
            Tuple of (audio_bytes, content_type) or None if not cached
        """
        key = self._get_key(text, voice_id, format)

        self._acquire_lock(key)
        try:
            entry = self._cache.get(key)
            if entry is None:
                self._total_misses += 1
                return None

            # Check TTL
            now = time.time()
            if now - entry.timestamp > self.ttl_seconds:
                # Expired, remove
                del self._cache[key]
                self._order.remove(key)
                self._total_misses += 1
                return None

            # Update LRU order and hit count
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            entry.hits += 1
            self._total_hits += 1

            return (entry.audio_bytes, entry.content_type)

        finally:
            self._release_lock(key)

    def set(
        self,
        text: str,
        voice_id: str,
        format: str,
        audio_bytes: bytes,
        content_type: str,
    ) -> None:
        """
        Store audio in cache.

        Args:
            text: Input text
            voice_id: Voice identifier
            format: Audio format
            audio_bytes: Audio data
            content_type: MIME type (audio/wav, audio/mpeg)
        """
        key = self._get_key(text, voice_id, format)

        self._acquire_lock(key)
        try:
            # Evict LRU entry if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                if self._order:
                    oldest_key = self._order.pop(0)
                    del self._cache[oldest_key]

            # Create entry
            entry = CacheEntry(
                audio_bytes=audio_bytes,
                content_type=content_type,
                timestamp=time.time(),
            )

            # Update or add
            if key in self._cache:
                self._order.remove(key)

            self._cache[key] = entry
            self._order.append(key)

        finally:
            self._release_lock(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._order.clear()
        self._total_hits = 0
        self._total_misses = 0

    def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache metrics
        """
        total_requests = self._total_hits + self._total_misses
        hit_rate = (self._total_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_bytes": sum(len(e.audio_bytes) for e in self._cache.values()),
        }

    def remove_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            key for key, entry in self._cache.items() if now - entry.timestamp > self.ttl_seconds
        ]

        for key in expired_keys:
            del self._cache[key]
            if key in self._order:
                self._order.remove(key)

        return len(expired_keys)


# Global cache instance
_cache_instance: Optional[TTSCache] = None


def get_cache(max_size: int = 100, ttl_seconds: int = 3600) -> TTSCache:
    """
    Get or create the global TTS cache instance.

    Args:
        max_size: Maximum cache entries
        ttl_seconds: Cache TTL in seconds

    Returns:
        TTSCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = TTSCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return _cache_instance


def clear_cache() -> None:
    """Clear the global cache."""
    global _cache_instance
    if _cache_instance is not None:
        _cache_instance.clear()
    _cache_instance = None


def cache_stats() -> dict:
    """Get global cache statistics."""
    cache = get_cache()
    return cache.stats()
