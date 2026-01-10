"use client";

import { useState, useRef } from "react";
import { Send, Mic, X } from "lucide-react";
import { VoiceChatButton } from "./VoiceChatButton";
import { getTranslation } from "@/lib/i18n";
import { useGlobal } from "@/context/GlobalContext";

interface VoiceChatInputProps {
  onSendMessage: (message: string, audioUrl?: string) => void;
  onVoiceTranscript?: (transcript: string) => void;
  onVoiceReply?: (reply: string, audioUrl: string | null) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function VoiceChatInput({
  onSendMessage,
  onVoiceTranscript,
  onVoiceReply,
  disabled = false,
  placeholder,
}: VoiceChatInputProps) {
  const { uiSettings } = useGlobal();
  const t = (key: string) => getTranslation(uiSettings.language, key);

  const [message, setMessage] = useState("");
  const [showVoiceResult, setShowVoiceResult] = useState(false);
  const [voiceTranscript, setVoiceTranscript] = useState<string | null>(null);
  const [voiceReply, setVoiceReply] = useState<string | null>(null);
  const [voiceAudioUrl, setVoiceAudioUrl] = useState<string | null>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if (!message.trim() || disabled) return;

    // If there's a voice result, include the audio URL
    const audioToSend = showVoiceResult ? voiceAudioUrl : undefined;
    onSendMessage(message, audioToSend);

    // Clear state
    setMessage("");
    setShowVoiceResult(false);
    setVoiceTranscript(null);
    setVoiceReply(null);
    setVoiceAudioUrl(null);

    // Focus textarea
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleTranscript = (transcript: string) => {
    setVoiceTranscript(transcript);
    setShowVoiceResult(true);
    onVoiceTranscript?.(transcript);
  };

  const handleReply = (reply: string, audioUrl: string | null) => {
    setVoiceReply(reply);
    setVoiceAudioUrl(audioUrl);
    onVoiceReply?.(reply, audioUrl);
  };

  const handleReRecord = () => {
    setShowVoiceResult(false);
    setVoiceTranscript(null);
    setVoiceReply(null);
    setVoiceAudioUrl(null);
  };

  const clearVoiceResult = () => {
    setShowVoiceResult(false);
    setVoiceTranscript(null);
    setVoiceReply(null);
    setVoiceAudioUrl(null);
  };

  const useVoiceReply = () => {
    if (voiceReply) {
      onSendMessage(voiceReply, voiceAudioUrl || undefined);
      clearVoiceResult();
    }
  };

  return (
    <div className="border-t border-slate-200 dark:border-slate-700 p-4 bg-white dark:bg-slate-800">
      {/* Voice Result Preview */}
      {showVoiceResult && voiceTranscript && (
        <div className="mb-3 p-3 bg-slate-100 dark:bg-slate-700 rounded-xl">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Mic className="w-4 h-4 text-blue-500" />
              <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
                {t("Voice Input")}
              </span>
            </div>
            <button
              onClick={handleReRecord}
              className="text-xs text-blue-500 hover:text-blue-600"
            >
              {t("Re-record")}
            </button>
          </div>

          {/* Transcript */}
          <p className="text-sm text-slate-700 dark:text-slate-300 mb-2">
            {voiceTranscript}
          </p>

          {/* Reply Preview */}
          {voiceReply && (
            <>
              <div className="border-t border-slate-200 dark:border-slate-600 my-2" />
              <div className="flex items-center gap-2 text-xs text-green-600 dark:text-green-400 mb-1">
                <span className="font-medium">{t("AI Reply")}</span>
                {voiceAudioUrl && (
                  <span className="px-1.5 py-0.5 bg-green-100 dark:bg-green-900/30 rounded text-xs">
                    🎵 {t("Audio available")}
                  </span>
                )}
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-400 line-clamp-2">
                {voiceReply}
              </p>

              <div className="flex items-center gap-2 mt-2">
                <button
                  onClick={useVoiceReply}
                  className="px-3 py-1.5 bg-amber-500 hover:bg-amber-600 text-white text-xs rounded-lg transition-colors flex items-center gap-1"
                >
                  <Send className="w-3 h-3" />
                  {t("Send Reply")}
                </button>
                <button
                  onClick={clearVoiceResult}
                  className="px-3 py-1.5 bg-slate-200 dark:bg-slate-600 hover:bg-slate-300 dark:hover:bg-slate-500 text-slate-700 dark:text-slate-300 text-xs rounded-lg transition-colors"
                >
                  <X className="w-3 h-3" />
                  {t("Cancel")}
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {/* Input Area */}
      <div className="flex items-end gap-3">
        {/* Voice Button */}
        <VoiceChatButton
          onTranscript={handleTranscript}
          onReply={handleReply}
          disabled={disabled}
        />

        {/* Text Input */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder={placeholder || t("Type a message...")}
            rows={1}
            className="w-full px-4 py-3 pr-12 bg-slate-100 dark:bg-slate-700 border-0 rounded-xl resize-none focus:ring-2 focus:ring-amber-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500"
            style={{
              minHeight: "48px",
              maxHeight: "120px",
            }}
          />

          {/* Send Button */}
          <button
            onClick={handleSend}
            disabled={!message.trim() || disabled}
            className="absolute right-2 bottom-2 p-2 bg-amber-500 hover:bg-amber-600 disabled:bg-slate-300 dark:disabled:bg-slate-600 disabled:cursor-not-allowed rounded-lg transition-colors"
          >
            <Send className="w-4 h-4 text-white" />
          </button>
        </div>
      </div>

      {/* Tip */}
      <p className="text-xs text-slate-400 dark:text-slate-500 mt-2 text-center">
        {t("Click the microphone to speak, or type a message")}
      </p>
    </div>
  );
}

export default VoiceChatInput;
