// API configuration and utility functions

// Get API base URL from environment variable
// This is automatically set by start_web.py based on config/main.yaml
// The .env.local file is auto-generated on startup with the correct backend port
// Can be overridden by localStorage for remote backend (LAN mode)
export const getBackendBaseUrl = (): string => {
  if (typeof window !== "undefined") {
    // Check localStorage for remote backend override
    const storedUrl = localStorage.getItem("deeptutor-backend-url");
    if (storedUrl && storedUrl.trim()) {
      return storedUrl;
    }
  }
  return process.env.NEXT_PUBLIC_API_BASE || "";
};

export const API_BASE_URL = getBackendBaseUrl();

/**
 * Set the backend URL (for remote backend / LAN mode)
 * @param url - The backend URL to use
 */
export function setBackendBaseUrl(url: string): void {
  if (typeof window !== "undefined") {
    if (url && url.trim()) {
      localStorage.setItem("deeptutor-backend-url", url.trim());
    } else {
      localStorage.removeItem("deeptutor-backend-url");
    }
    // Force reload to apply the new URL
    window.location.reload();
  }
}

/**
 * Check if a custom backend URL is configured
 */
export function isUsingRemoteBackend(): boolean {
  if (typeof window !== "undefined") {
    return !!localStorage.getItem("deeptutor-backend-url");
  }
  return false;
}

/**
 * Get the stored backend URL without triggering reload
 */
export function getStoredBackendUrl(): string | null {
  if (typeof window !== "undefined") {
    return localStorage.getItem("deeptutor-backend-url");
  }
  return null;
}

/**
 * Construct a full API URL from a path
 * @param path - API path (e.g., '/api/v1/knowledge/list')
 * @returns Full URL (e.g., 'http://localhost:8000/api/v1/knowledge/list')
 */
export function apiUrl(path: string): string {
  const base = getBackendBaseUrl();
  if (!base) {
    throw new Error(
      "Backend URL not configured. Please set NEXT_PUBLIC_API_BASE in .env.local or configure a remote backend in Settings.",
    );
  }

  // Remove leading slash if present to avoid double slashes
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  // Remove trailing slash from base URL if present
  const normalizedBase = base.endsWith("/") ? base.slice(0, -1) : base;

  return `${normalizedBase}${normalizedPath}`;
}

/**
 * Construct a WebSocket URL from a path
 * @param path - WebSocket path (e.g., '/api/v1/solve')
 * @returns WebSocket URL (e.g., 'ws://localhost:{backend_port}/api/v1/solve')
 * Note: backend_port is configured in config/main.yaml
 */
export function wsUrl(path: string): string {
  const base = getBackendBaseUrl().replace(/^http:/, "ws:").replace(/^https:/, "wss:");

  if (!base) {
    throw new Error(
      "Backend URL not configured. Please set NEXT_PUBLIC_API_BASE in .env.local or configure a remote backend in Settings.",
    );
  }

  // Remove leading slash if present to avoid double slashes
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  // Remove trailing slash from base URL if present
  const normalizedBase = base.endsWith("/") ? base.slice(0, -1) : base;

  return `${normalizedBase}${normalizedPath}`;
}
