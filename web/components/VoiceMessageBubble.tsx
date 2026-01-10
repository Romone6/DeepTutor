"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import {
  Play,
  Pause,
  Volume2,
  VolumeX,
  RotateCcw,
  Mic,
  Bot,
} from "lucide-react";
import { processLatexContent } from "@/lib/latex";
import { getTranslation } from "@/lib/i18n";
import { useGlobal } from "@/context/GlobalContext";

interface VoiceMessageBubbleProps {
  role: "user" | "assistant";
  transcript?: string;
  content: string;
  audioUrl?: string | null;
  timestamp?: number;
  onReplay?: () => void;
  onReRecord?: () => void;
  showControls?: boolean;
}

export function VoiceMessageBubble({
  role,
  transcript,
  content,
  audioUrl,
  timestamp,
  onReplay,
  onReRecord,
  showControls = true,
}: VoiceMessageBubbleProps) {
  const { uiSettings } = useGlobal();
  const t = (key: string) => getTranslation(uiSettings.language, key);

  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackProgress, setPlaybackProgress] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const hasVoiceContent = role === "assistant" && audioUrl;
  const hasTranscript = role === "user" && transcript;

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        URL.revokeObjectURL(audioRef.current.src);
      }
    };
  }, []);

  const togglePlay = () => {
    if (!audioRef.current || !audioUrl) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      const progress = (audioRef.current.currentTime / audioRef.current.duration) * 100;
      setPlaybackProgress(progress);
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setPlaybackProgress(0);
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || !audioUrl) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    audioRef.current.currentTime = percent * audioRef.current.duration;
  };

  const replayAudio = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play();
    }
    onReplay?.();
  };

  const toggleMute = () => {
    if (audioRef.current) {
      audioRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  return (
    <div className={`flex gap-3 ${role === "user" ? "justify-end" : "justify-start"}`}>
      {/* Avatar */}
      {role === "assistant" && (
        <div className="w-8 h-8 rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center shrink-0">
          <Bot className="w-4 h-4 text-amber-600 dark:text-amber-400" />
        </div>
      )}

      {/* Message Bubble */}
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 ${
          role === "user"
            ? "bg-blue-500 text-white rounded-br-none"
            : "bg-slate-100 dark:bg-slate-700 text-slate-800 dark:text-slate-200 rounded-bl-none"
        }`}
      >
        {/* Voice Transcript (User side) */}
        {hasTranscript && transcript && (
          <div className="mb-2 pb-2 border-b border-blue-400/30 dark:border-blue-400/20">
            <div className="flex items-center gap-1.5 text-xs text-blue-200 dark:text-blue-300 mb-1">
              <Mic className="w-3 h-3" />
              <span>{t("Transcript")}</span>
            </div>
            <p className="text-sm italic opacity-90">{transcript}</p>
            {showControls && onReRecord && (
              <button
                onClick={onReRecord}
                className="mt-2 text-xs flex items-center gap-1 text-blue-200 hover:text-white transition-colors"
              >
                <RotateCcw className="w-3 h-3" />
                {t("Re-record")}
              </button>
            )}
          </div>
        )}

        {/* Main Content */}
        {role === "user" ? (
          <p className="whitespace-pre-wrap">{content}</p>
        ) : (
          <div className="prose prose-sm dark:prose-invert max-w-none prose-p:my-2 prose-headings:my-2">
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
            >
              {processLatexContent(content)}
            </ReactMarkdown>
          </div>
        )}

        {/* Audio Player (Assistant side) */}
        {hasVoiceContent && audioUrl && (
          <div className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-600">
            <audio
              ref={audioRef}
              src={audioUrl}
              onTimeUpdate={handleTimeUpdate}
              onEnded={handleEnded}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              preload="metadata"
            />

            {/* Progress Bar */}
            <div
              className="h-1.5 bg-slate-200 dark:bg-slate-600 rounded-full cursor-pointer overflow-hidden"
              onClick={handleSeek}
            >
              <div
                className="h-full bg-amber-500 rounded-full transition-all duration-100"
                style={{ width: `${playbackProgress}%` }}
              />
            </div>

            {/* Controls */}
            <div className="flex items-center justify-between mt-2">
              <div className="flex items-center gap-2">
                <button
                  onClick={togglePlay}
                  className="w-8 h-8 rounded-full bg-amber-500 hover:bg-amber-600 flex items-center justify-center transition-colors"
                >
                  {isPlaying ? (
                    <Pause className="w-4 h-4 text-white" />
                  ) : (
                    <Play className="w-4 h-4 text-white" />
                  )}
                </button>
                <button
                  onClick={toggleMute}
                  className="w-8 h-8 rounded-full bg-slate-200 dark:bg-slate-600 hover:bg-slate-300 dark:hover:bg-slate-500 flex items-center justify-center transition-colors"
                >
                  {isMuted ? (
                    <VolumeX className="w-4 h-4 text-slate-700 dark:text-slate-300" />
                  ) : (
                    <Volume2 className="w-4 h-4 text-slate-700 dark:text-slate-300" />
                  )}
                </button>
              </div>

              {showControls && onReplay && (
                <button
                  onClick={replayAudio}
                  className="text-xs flex items-center gap-1 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                >
                  <RotateCcw className="w-3 h-3" />
                  {t("Replay")}
                </button>
              )}
            </div>
          </div>
        )}

        {/* Timestamp */}
        {timestamp && (
          <p
            className={`text-xs mt-2 ${
              role === "user" ? "text-blue-200" : "text-slate-400 dark:text-slate-500"
            }`}
          >
            {new Date(timestamp * 1000).toLocaleTimeString(
              uiSettings.language === "zh" ? "zh-CN" : "en-US",
              { hour: "2-digit", minute: "2-digit" },
            )}
          </p>
        )}

        {/* Voice Indicator Badge (if has audio) */}
        {hasVoiceContent && (
          <div
            className={`absolute -bottom-1 -right-1 w-5 h-5 rounded-full bg-amber-500 flex items-center justify-center ${
              isPlaying ? "animate-pulse" : ""
            }`}
          >
            <Volume2 className="w-3 h-3 text-white" />
          </div>
        )}
      </div>

      {/* User Avatar */}
      {role === "user" && (
        <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center shrink-0">
          <Mic className="w-4 h-4 text-white" />
        </div>
      )}
    </div>
  );
}

export default VoiceMessageBubble;
