"use client";

export type StreamingCapability = "streaming" | "fallback" | "unknown";

export interface BrowserCapabilities {
  streamingAudio: StreamingCapability;
  webAudioApi: boolean;
  mediaSourceExtensions: boolean;
  fetchStreaming: boolean;
}

export function detectStreamingCapabilities(): BrowserCapabilities {
  const capabilities: BrowserCapabilities = {
    streamingAudio: "unknown",
    webAudioApi: false,
    mediaSourceExtensions: false,
    fetchStreaming: false,
  };

  if (typeof window === "undefined") {
    return capabilities;
  }

  capabilities.webAudioApi = !!(
    window.AudioContext || (window as any).webkitAudioContext
  );

  capabilities.mediaSourceExtensions = !!(
    window.MediaSource || (window as any).MediaSource
  );

  capabilities.fetchStreaming = !!(
    typeof fetch === "function" &&
    typeof Request === "function" &&
    Request.prototype.hasOwnProperty("cache")
  );

  if (capabilities.webAudioApi && capabilities.fetchStreaming) {
    capabilities.streamingAudio = "streaming";
  } else if (capabilities.webAudioApi || capabilities.mediaSourceExtensions) {
    capabilities.streamingAudio = "fallback";
  } else {
    capabilities.streamingAudio = "unknown";
  }

  return capabilities;
}

export function canStreamAudio(): boolean {
  const caps = detectStreamingCapabilities();
  return caps.streamingAudio === "streaming";
}

export function needsFallback(): boolean {
  const caps = detectStreamingCapabilities();
  return caps.streamingAudio === "fallback" || caps.streamingAudio === "unknown";
}

export async function testStreamingSupport(): Promise<StreamingCapability> {
  if (typeof window === "undefined") {
    return "unknown";
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1000);

    const response = await fetch("/api/voice/speak/stream/chunked/simple", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "test", chunk_size: 50 }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (response.ok && response.body) {
      return "streaming";
    }
    return "fallback";
  } catch {
    return "unknown";
  }
}
