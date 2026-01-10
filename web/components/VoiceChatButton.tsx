"use client";

import { useState, useEffect } from "react";
import {
  Mic,
  MicOff,
  Square,
  Play,
  Pause,
  Volume2,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { useVoiceChat } from "@/hooks/useVoiceChat";
import { getTranslation } from "@/lib/i18n";
import { useGlobal } from "@/context/GlobalContext";

interface VoiceChatButtonProps {
  onTranscript?: (transcript: string) => void;
  onReply?: (reply: string, audioUrl: string | null) => void;
  disabled?: boolean;
  className?: string;
}

export function VoiceChatButton({
  onTranscript,
  onReply,
  disabled = false,
  className = "",
}: VoiceChatButtonProps) {
  const { uiSettings } = useGlobal();
  const t = (key: string) => getTranslation(uiSettings.language, key);

  const {
    isRecording,
    isProcessing,
    isPlaying,
    transcript,
    reply,
    audioUrl,
    error,
    startRecording,
    stopRecording,
    playAudio,
    stopAudio,
    clear,
    hasRecording,
  } = useVoiceChat({
    onTranscribe: onTranscript,
    onReply: onReply,
    onError: (err) => console.error("Voice chat error:", err),
    enableTTS: true,
  });

  const [showError, setShowError] = useState(false);

  useEffect(() => {
    if (error) {
      setShowError(true);
      const timer = setTimeout(() => setShowError(false), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  const handleMicClick = async () => {
    if (isRecording) {
      await stopRecording();
    } else {
      clear();
      await startRecording();
    }
  };

  // Check for microphone support
  const [hasMicSupport, setHasMicSupport] = useState(true);
  useEffect(() => {
    const checkMic = async () => {
      try {
        await navigator.mediaDevices.getUserMedia({ audio: true });
        setHasMicSupport(true);
      } catch {
        setHasMicSupport(false);
      }
    };
    checkMic();
  }, []);

  if (!hasMicSupport) {
    return null; // Don't show button if no mic
  }

  return (
    <div className={`relative ${className}`}>
      {/* Main Button */}
      <button
        onClick={handleMicClick}
        disabled={disabled || isProcessing}
        className={`
          relative w-12 h-12 rounded-full flex items-center justify-center
          transition-all duration-200
          ${
            isRecording
              ? "bg-red-500 hover:bg-red-600 animate-pulse"
              : "bg-amber-500 hover:bg-amber-600"
          }
          disabled:opacity-50 disabled:cursor-not-allowed
          shadow-lg hover:shadow-xl
        `}
        title={isRecording ? t("Stop Recording") : t("Start Recording")}
      >
        {isProcessing ? (
          <Loader2 className="w-5 h-5 text-white animate-spin" />
        ) : isRecording ? (
          <>
            <div className="absolute inset-0 rounded-full bg-red-500 animate-ping opacity-50" />
            <Mic className="w-5 h-5 text-white relative z-10" />
          </>
        ) : (
          <Mic className="w-5 h-5 text-white" />
        )}
      </button>

      {/* Error Toast */}
      {showError && error && (
        <div className="absolute bottom-14 left-1/2 -translate-x-1/2 w-64 p-3 bg-red-500 text-white text-sm rounded-lg shadow-lg flex items-start gap-2 animate-in slide-in-from-bottom-2">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* Recording Indicator */}
      {isRecording && (
        <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-xs text-red-500 font-medium whitespace-nowrap">
          {t("Recording...")}
        </div>
      )}

      {/* Processing Indicator */}
      {isProcessing && (
        <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-xs text-amber-500 font-medium whitespace-nowrap flex items-center gap-1">
          <Loader2 className="w-3 h-3 animate-spin" />
          {t("Processing...")}
        </div>
      )}

      {/* Result Controls */}
      {(audioUrl || isPlaying) && (
        <div className="absolute bottom-14 right-0 flex items-center gap-1 bg-white dark:bg-slate-800 rounded-lg shadow-lg p-1 animate-in slide-in-from-bottom-2">
          <button
            onClick={isPlaying ? stopAudio : playAudio}
            className="p-2 rounded hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
            title={isPlaying ? t("Stop") : t("Play")}
          >
            {isPlaying ? (
              <Pause className="w-4 h-4 text-slate-700 dark:text-slate-300" />
            ) : (
              <Play className="w-4 h-4 text-slate-700 dark:text-slate-300" />
            )}
          </button>
          <button
            onClick={clear}
            className="p-2 rounded hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
            title={t("Clear")}
          >
            <Square className="w-4 h-4 text-slate-700 dark:text-slate-300" />
          </button>
        </div>
      )}
    </div>
  );
}

export default VoiceChatButton;
