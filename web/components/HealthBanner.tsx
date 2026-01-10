"use client";

import { useState, useEffect } from "react";
import {
  AlertTriangle,
  AlertOctagon,
  Info,
  X,
  ChevronDown,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  BookOpen,
  Brain,
} from "lucide-react";
import { apiUrl } from "@/lib/api";

interface DegradationBanner {
  component: string;
  message: string;
  severity: "warning" | "error" | "info";
  action: string;
}

interface DegradationStatus {
  overall_healthy: boolean;
  degraded_components: string[];
  banners: DegradationBanner[];
  components: Record<string, {
    component: string;
    healthy: boolean;
    message: string;
    error?: string;
    degradation_mode?: string;
  }>;
  timestamp: string;
}

export default function HealthBanner() {
  const [status, setStatus] = useState<DegradationStatus | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(apiUrl("/api/v1/debug/health/degradation"));
        if (response.ok) {
          const data = await response.json();
          setStatus(data);
        }
      } catch (error) {
        // Silently fail - component will show healthy by default
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 60000); // Check every minute
    return () => clearInterval(interval);
  }, []);

  // Don't show if healthy or dismissed
  if (!status || status.overall_healthy || dismissed) {
    return null;
  }

  const getSeverityConfig = (severity: string) => {
    switch (severity) {
      case "error":
        return {
          bg: "bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-800",
          text: "text-red-800 dark:text-red-200",
          icon: AlertOctagon,
        };
      case "warning":
        return {
          bg: "bg-amber-50 dark:bg-amber-900/30 border-amber-200 dark:border-amber-800",
          text: "text-amber-800 dark:text-amber-200",
          icon: AlertTriangle,
        };
      default:
        return {
          bg: "bg-blue-50 dark:bg-blue-900/30 border-blue-200 dark:border-blue-800",
          text: "text-blue-800 dark:text-blue-200",
          icon: Info,
        };
    }
  };

  const getComponentIcon = (component: string) => {
    switch (component) {
      case "asr":
        return <MicOff className="w-4 h-4" />;
      case "tts":
        return <VolumeX className="w-4 h-4" />;
      case "rag":
        return <BookOpen className="w-4 h-4" />;
      case "llm":
        return <Brain className="w-4 h-4" />;
      default:
        return <AlertTriangle className="w-4 h-4" />;
    }
  };

  const getActionText = (action: string) => {
    switch (action) {
      case "text_only":
        return "Text mode active";
      case "general_explanation":
        return "General knowledge only";
      case "local_only":
        return "Using local fallback";
      default:
        return action;
    }
  };

  const handleDismiss = () => {
    setDismissed(true);
    // Store dismissal in localStorage for 5 minutes
    localStorage.setItem("health_banner_dismissed", Date.now().toString());
  };

  const primaryBanner = status.banners[0];
  const severityConfig = primaryBanner
    ? getSeverityConfig(primaryBanner.severity)
    : getSeverityConfig("warning");
  const Icon = severityConfig.icon;

  return (
    <div
      className={`relative mb-3 rounded-lg border ${severityConfig.bg} ${severityConfig.text} transition-all`}
    >
      {/* Collapsed State */}
      <div
        className="px-4 py-3 flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          <Icon className="w-5 h-5 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="font-medium text-sm truncate">
              {primaryBanner?.message || "Some services are unavailable"}
            </p>
            <p className="text-xs opacity-80">
              {status.degraded_components.length} degraded component(s)
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs opacity-75">
            {expanded ? "Hide details" : "Show details"}
          </span>
          <ChevronDown
            className={`w-4 h-4 transition-transform ${expanded ? "rotate-180" : ""}`}
          />
        </div>
      </div>

      {/* Expanded State */}
      {expanded && (
        <div className="px-4 pb-4 pt-2 border-t border-current/20">
          {/* All Degraded Components */}
          <div className="space-y-2 mt-2">
            {status.banners.map((banner, index) => {
              const config = getSeverityConfig(banner.severity);
              const BannerIcon = config.icon;
              return (
                <div
                  key={index}
                  className={`p-3 rounded-md bg-white/50 dark:bg-black/20 border border-current/20`}
                >
                  <div className="flex items-start gap-3">
                    {getComponentIcon(banner.component)}
                    <div className="flex-1">
                      <p className="font-medium text-sm capitalize">
                        {banner.component.toUpperCase()}
                      </p>
                      <p className="text-sm opacity-90 mt-1">
                        {banner.message}
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        <span className="text-xs px-2 py-0.5 rounded bg-current/10">
                          {getActionText(banner.action)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Component Details */}
          <div className="mt-4 pt-3 border-t border-current/20">
            <p className="text-xs font-medium mb-2 opacity-75">Component Status</p>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(status.components || {}).map(([key, comp]) => (
                <div
                  key={key}
                  className={`p-2 rounded text-xs flex items-center gap-2 ${
                    comp.healthy
                      ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
                      : "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"
                  }`}
                >
                  {comp.healthy ? (
                    <span className="w-2 h-2 rounded-full bg-green-500" />
                  ) : (
                    <span className="w-2 h-2 rounded-full bg-red-500" />
                  )}
                  <span className="capitalize">{key}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="mt-4 flex items-center justify-between">
            <button
              onClick={(e) => {
                e.stopPropagation();
                window.location.reload();
              }}
              className="px-3 py-1.5 text-xs font-medium rounded bg-white/50 dark:bg-black/20 hover:bg-white/70 dark:hover:bg-black/30 transition-colors"
            >
              Refresh Status
            </button>
            <button
              onClick={handleDismiss}
              className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded bg-white/50 dark:bg-black/20 hover:bg-white/70 dark:hover:bg-black/30 transition-colors"
            >
              <X className="w-3 h-3" />
              Dismiss for 5 min
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// Hook for checking specific component availability
export function useComponentHealth(component: string) {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [message, setMessage] = useState<string>("");

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(
          apiUrl(`/api/v1/debug/health/degradation`)
        );
        if (response.ok) {
          const data = await response.json();
          const comp = data.components?.[component];
          if (comp) {
            setHealthy(comp.healthy);
            setMessage(comp.message || "");
          }
        }
      } catch {
        setHealthy(null);
        setMessage("Unable to check");
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [component]);

  return { healthy, message };
}

// Component-specific hooks for convenience
export function useASRHealth() {
  return useComponentHealth("asr");
}

export function useTTSHealth() {
  return useComponentHealth("tts");
}

export function useRAGHealth() {
  return useComponentHealth("rag");
}

export function useLLMHealth() {
  return useComponentHealth("llm");
}
