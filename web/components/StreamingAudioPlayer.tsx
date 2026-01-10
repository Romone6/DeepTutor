"use client";

import { useStreamingAudio } from "@/hooks/useStreamingAudio";
import { Volume2, VolumeX, Play, Square, Loader2 } from "lucide-react";
import { useState, useCallback } from "react";

interface StreamingAudioPlayerProps {
  text: string;
  voiceId?: string;
  autoPlay?: boolean;
  className?: string;
}

export function StreamingAudioPlayer({
  text,
  voiceId,
  autoPlay = false,
  className = "",
}: StreamingAudioPlayerProps) {
  const {
    play,
    stop,
    pause,
    isPlaying,
    isLoading,
    progress,
    error,
  } = useStreamingAudio({
    autoPlay,
    onPlaybackStart: () => console.log("Playback started"),
    onPlaybackEnd: () => console.log("Playback ended"),
    onError: (err) => console.error("Audio error:", err),
  });

  const [isMuted, setIsMuted] = useState(false);

  const handlePlay = useCallback(async () => {
    if (isPlaying) {
      pause();
    } else {
      await play(text, voiceId);
    }
  }, [text, voiceId, isPlaying, pause, play]);

  const handleStop = useCallback(() => {
    stop();
  }, [stop]);

  const toggleMute = useCallback(() => {
    setIsMuted((prev) => !prev);
  }, []);

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <button
        onClick={handlePlay}
        disabled={isLoading && !isPlaying}
        className="p-2 rounded-full bg-amber-500 hover:bg-amber-600 disabled:bg-slate-300 dark:disabled:bg-slate-600 text-white transition-colors"
        title={isPlaying ? "Pause" : "Play audio"}
      >
        {isLoading && !isPlaying ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : isPlaying ? (
          <Square className="w-4 h-4" />
        ) : (
          <Play className="w-4 h-4" />
        )}
      </button>

      <button
        onClick={toggleMute}
        className="p-2 rounded-full bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300 transition-colors"
        title={isMuted ? "Unmute" : "Mute"}
      >
        {isMuted ? (
          <VolumeX className="w-4 h-4" />
        ) : (
          <Volume2 className="w-4 h-4" />
        )}
      </button>

      {isLoading && (
        <div className="flex-1 h-1 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-amber-500 transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {error && (
        <span className="text-xs text-red-500">
          Audio error: {error.message}
        </span>
      )}
    </div>
  );
}

export default StreamingAudioPlayer;
