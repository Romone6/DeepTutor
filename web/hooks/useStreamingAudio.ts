"use client";

import { useState, useRef, useCallback, useEffect } from "react";

interface UseStreamingAudioOptions {
  onPlaybackStart?: () => void;
  onPlaybackEnd?: () => void;
  onError?: (error: Error) => void;
  onChunk?: (chunkIndex: number, totalChunks: number) => void;
  autoPlay?: boolean;
}

interface UseStreamingAudioReturn {
  play: (text: string, voiceId?: string) => Promise<void>;
  stop: () => void;
  pause: () => void;
  resume: () => void;
  isPlaying: boolean;
  isLoading: boolean;
  progress: number;
  error: Error | null;
}

export function useStreamingAudio(
  options: UseStreamingAudioOptions = {}
): UseStreamingAudioReturn {
  const {
    onPlaybackStart,
    onPlaybackEnd,
    onError,
    onChunk,
    autoPlay = false,
  } = options;

  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<Error | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const audioChunksRef = useRef<Uint8Array[]>([]);
  const streamControllerRef = useRef<AbortController | null>(null);
  const playbackStartedRef = useRef(false);

  const initializeAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    return audioContextRef.current;
  }, []);

  const decodeAudioChunks = useCallback(
    async (chunks: Uint8Array[]): Promise<AudioBuffer | null> => {
      const ctx = initializeAudioContext();

      try {
        const combinedLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const combinedArray = new Uint8Array(combinedLength);

        let offset = 0;
        for (const chunk of chunks) {
          combinedArray.set(chunk, offset);
          offset += chunk.length;
        }

        const arrayBuffer = combinedArray.buffer;
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        return audioBuffer;
      } catch (err) {
        console.error("Failed to decode audio:", err);
        return null;
      }
    },
    [initializeAudioContext]
  );

  const playDecodedBuffer = useCallback(
    async (audioBuffer: AudioBuffer) => {
      const ctx = initializeAudioContext();

      if (ctx.state === "suspended") {
        await ctx.resume();
      }

      if (sourceNodeRef.current) {
        sourceNodeRef.current.stop();
        sourceNodeRef.current = null;
      }

      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.start(0);
      sourceNodeRef.current = source;

      source.onended = () => {
        setIsPlaying(false);
        setProgress(100);
        onPlaybackEnd?.();
      };

      setIsPlaying(true);
      playbackStartedRef.current = true;
      onPlaybackStart?.();
    },
    [initializeAudioContext, onPlaybackEnd, onPlaybackStart]
  );

  const streamAudio = useCallback(
    async (text: string, voiceId?: string) => {
      setIsLoading(true);
      setError(null);
      audioChunksRef.current = [];
      playbackStartedRef.current = false;

      try {
        const endpoint = `${process.env.NEXT_PUBLIC_API_URL || ""}/api/voice/speak/stream/chunked/simple`;
        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text,
            voice_id: voiceId || "default",
            format: "wav",
            speed: 1.0,
            chunk_size: 200,
          }),
        });

        if (!response.ok) {
          throw new Error(`Streaming failed: ${response.statusText}`);
        }

        if (!response.body) {
          throw new Error("Response body is null");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let totalChunks = 0;
        let receivedChunks = 0;

        const readChunk = async (): Promise<void> => {
          const { done, value } = await reader.read();

          if (done) {
            if (audioChunksRef.current.length > 0) {
              const audioBuffer = await decodeAudioChunks(audioChunksRef.current);
              if (audioBuffer) {
                if (autoPlay || playbackStartedRef.current) {
                  await playDecodedBuffer(audioBuffer);
                }
              }
            }
            setIsLoading(false);
            return;
          }

          audioChunksRef.current.push(value);
          receivedChunks += 1;

          if (receivedChunks === 1 && !playbackStartedRef.current) {
            setProgress(10);
          } else {
            setProgress(Math.min(90, (receivedChunks / Math.max(totalChunks, 1)) * 100));
          }

          onChunk?.(receivedChunks, totalChunks);

          await readChunk();
        };

        await readChunk();
      } catch (err) {
        const audioError = err instanceof Error ? err : new Error(String(err));
        setError(audioError);
        setIsLoading(false);
        onError?.(audioError);

        await playNonStreamingFallback(text, voiceId);
      }
    },
    [autoPlay, decodeAudioChunks, onChunk, onError, playDecodedBuffer]
  );

  const playNonStreamingFallback = useCallback(
    async (text: string, voiceId?: string) => {
      try {
        setIsLoading(true);
        setError(null);

        const endpoint = `${process.env.NEXT_PUBLIC_API_URL || ""}/api/voice/speak`;
        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text,
            voice_id: voiceId || "default",
            format: "wav",
            speed: 1.0,
          }),
        });

        if (!response.ok) {
          throw new Error(`Fallback failed: ${response.statusText}`);
        }

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);

        audio.onplay = () => {
          setIsPlaying(true);
          onPlaybackStart?.();
        };

        audio.onended = () => {
          setIsPlaying(false);
          setProgress(100);
          onPlaybackEnd?.();
          URL.revokeObjectURL(audioUrl);
        };

        audio.onerror = () => {
          const audioError = new Error("Audio playback failed");
          setError(audioError);
          setIsPlaying(false);
          onError?.(audioError);
        };

        await audio.play();
      } catch (err) {
        const fallbackError = err instanceof Error ? err : new Error(String(err));
        setError(fallbackError);
        onError?.(fallbackError);
      } finally {
        setIsLoading(false);
      }
    },
    [onPlaybackEnd, onPlaybackStart, onError]
  );

  const stop = useCallback(() => {
    if (sourceNodeRef.current) {
      sourceNodeRef.current.stop();
      sourceNodeRef.current = null;
    }

    if (streamControllerRef.current) {
      streamControllerRef.current.abort();
      streamControllerRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setIsPlaying(false);
    setProgress(0);
    playbackStartedRef.current = false;
  }, []);

  const pause = useCallback(() => {
    if (sourceNodeRef.current && audioContextRef.current) {
      sourceNodeRef.current.stop();
      setIsPlaying(false);
    }
  }, []);

  const resume = useCallback(async () => {
    if (audioChunksRef.current.length > 0 && playbackStartedRef.current) {
      const audioBuffer = await decodeAudioChunks(audioChunksRef.current);
      if (audioBuffer) {
        await playDecodedBuffer(audioBuffer);
      }
    }
  }, [decodeAudioChunks, playDecodedBuffer]);

  const play = useCallback(
    async (text: string, voiceId?: string) => {
      if (!text.trim()) return;

      stop();
      audioChunksRef.current = [];

      try {
        await streamAudio(text, voiceId);
      } catch (err) {
        const playError = err instanceof Error ? err : new Error(String(err));
        setError(playError);
        onError?.(playError);
      }
    },
    [streamAudio, stop, onError]
  );

  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return {
    play,
    stop,
    pause,
    resume,
    isPlaying,
    isLoading,
    progress,
    error,
  };
}

export default useStreamingAudio;
