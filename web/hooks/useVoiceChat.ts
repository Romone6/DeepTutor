"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { apiUrl } from "@/lib/api";

interface UseVoiceChatOptions {
  onTranscribe?: (transcript: string) => void;
  onReply?: (reply: string, audioUrl: string | null) => void;
  onError?: (error: string) => void;
  enableTTS?: boolean;
}

interface UseVoiceChatReturn {
  isRecording: boolean;
  isProcessing: boolean;
  isPlaying: boolean;
  transcript: string | null;
  reply: string | null;
  audioUrl: string | null;
  error: string | null;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  playAudio: () => void;
  stopAudio: () => void;
  clear: () => void;
  hasRecording: boolean;
}

export function useVoiceChat(options: UseVoiceChatOptions = {}): UseVoiceChatReturn {
  const { onTranscribe, onReply, onError, enableTTS = true } = options;

  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [reply, setReply] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (audioPlayerRef.current) {
        audioPlayerRef.current.pause();
        URL.revokeObjectURL(audioPlayerRef.current.src);
      }
    };
  }, []);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      audioChunksRef.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to start recording";
      setError(message);
      onError?.(message);
    }
  }, [onError]);

  const stopRecording = useCallback(() => {
    return new Promise<void>((resolve) => {
      if (!mediaRecorderRef.current || !isRecording) {
        resolve();
        return;
      }

      const mediaRecorder = mediaRecorderRef.current;

      mediaRecorder.onstop = async () => {
        // Stop all tracks to release microphone
        mediaRecorder.stream.getTracks().forEach((track) => track.stop());

        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });

        setIsRecording(false);
        setIsProcessing(true);

        try {
          // Send to voice chat endpoint
          const formData = new FormData();
          formData.append("audio", audioBlob, "recording.webm");

          const response = await fetch(apiUrl("/api/v1/voice/chat"), {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
          }

          const data = await response.json();

          setTranscript(data.transcript);
          onTranscribe?.(data.transcript);

          setReply(data.reply);

          // Optionally synthesize speech for the reply
          if (enableTTS && data.reply) {
            try {
              const speakResponse = await fetch(apiUrl("/api/v1/voice/speak"), {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({
                  text: data.reply,
                  voice_id: "stub-male-1",
                  format: "wav",
                }),
              });

              if (speakResponse.ok) {
                const audioBlob = await speakResponse.blob();
                const url = URL.createObjectURL(audioBlob);
                setAudioUrl(url);
                onReply?.(data.reply, url);
              } else {
                // TTS failed, but we still have the text reply
                console.warn("TTS failed, showing text only");
                onReply?.(data.reply, null);
              }
            } catch (ttsErr) {
              console.warn("TTS error:", ttsErr);
              onReply?.(data.reply, null);
            }
          } else {
            onReply?.(data.reply, null);
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : "Failed to process voice";
          setError(message);
          onError?.(message);
        } finally {
          setIsProcessing(false);
          resolve();
        }
      };

      mediaRecorder.stop();
    });
  }, [isRecording, onTranscribe, onReply, onError, enableTTS]);

  const playAudio = useCallback(() => {
    if (!audioUrl) return;

    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
    }

    audioPlayerRef.current = new Audio(audioUrl);
    audioPlayerRef.current.onended = () => setIsPlaying(false);
    audioPlayerRef.current.onplay = () => setIsPlaying(true);
    audioPlayerRef.current.play();
  }, [audioUrl]);

  const stopAudio = useCallback(() => {
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
      audioPlayerRef.current.currentTime = 0;
    }
    setIsPlaying(false);
  }, []);

  const clear = useCallback(() => {
    setTranscript(null);
    setReply(null);
    setAudioUrl(null);
    setError(null);
    audioChunksRef.current = [];
    stopAudio();
  }, [stopAudio]);

  return {
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
    hasRecording: !!audioUrl || !!transcript,
  };
}

export default useVoiceChat;
