"use client";

import { useState, useCallback } from "react";
import { apiUrl } from "@/lib/api";

export interface ServiceHealth {
  name: string;
  endpoint: string;
  status: "unknown" | "checking" | "healthy" | "unhealthy" | "error";
  latency?: number;
  error?: string;
  details?: any;
}

export interface ConnectionTestResult {
  overall: "connected" | "partial" | "disconnected";
  backendUrl: string;
  services: ServiceHealth[];
  timestamp: string;
}

const DEFAULT_SERVICES: ServiceHealth[] = [
  { name: "Backend API", endpoint: "/api/v1/health", status: "unknown" },
  { name: "LLM", endpoint: "/api/v1/settings/env/test/", status: "unknown" },
  { name: "RAG", endpoint: "/api/v1/knowledge/list", status: "unknown" },
  { name: "TTS", endpoint: "/api/voice/health", status: "unknown" },
  { name: "ASR", endpoint: "/api/voice/health", status: "unknown" },
];

export function useConnectionTest() {
  const [isTesting, setIsTesting] = useState(false);
  const [results, setResults] = useState<ConnectionTestResult | null>(null);

  const testConnection = useCallback(async (backendUrl?: string) => {
    setIsTesting(true);
    setResults(null);

    const services = DEFAULT_SERVICES.map((s) => ({ ...s, status: "checking" as const }));
    const baseUrl = backendUrl || apiUrl("").replace(/\/api\/v1.*$/, "");
    const startTime = Date.now();

    for (const service of services) {
      const serviceStart = Date.now();

      try {
        // For health checks, use the correct endpoint
        let testUrl: string;
        if (service.endpoint.includes("health")) {
          testUrl = service.endpoint.startsWith("http")
            ? service.endpoint
            : `${baseUrl}${service.endpoint}`;
        } else {
          testUrl = service.endpoint.startsWith("http")
            ? service.endpoint
            : apiUrl(service.endpoint);
        }

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        const response = await fetch(testUrl, {
          method: "GET",
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        const latency = Date.now() - serviceStart;

        if (response.ok) {
          services[services.indexOf(service)].status = "healthy";
          services[services.indexOf(service)].latency = latency;

          try {
            const data = await response.json();
            services[services.indexOf(service)].details = data;
          } catch {
            // Ignore JSON parse errors
          }
        } else {
          services[services.indexOf(service)].status = "unhealthy";
          services[services.indexOf(service)].error = `HTTP ${response.status}`;
          services[services.indexOf(service)].latency = latency;
        }
      } catch (error) {
        const latency = Date.now() - serviceStart;
        services[services.indexOf(service)].status = "error";
        services[services.indexOf(service)].error =
          error instanceof Error ? error.message : "Connection failed";
        services[services.indexOf(service)].latency = latency;
      }
    }

    const totalLatency = Date.now() - startTime;
    const healthyCount = services.filter((s) => s.status === "healthy").length;
    const hasPartialConnection = services.some(
      (s) => s.status === "healthy" || s.status === "unhealthy",
    );

    const overall = healthyCount === services.length
      ? "connected"
      : healthyCount > 0 && hasPartialConnection
        ? "partial"
        : "disconnected";

    setResults({
      overall,
      backendUrl: baseUrl,
      services,
      timestamp: new Date().toISOString(),
    });

    setIsTesting(false);
    return { overall, services, totalLatency };
  }, []);

  const clearResults = useCallback(() => {
    setResults(null);
  }, []);

  return {
    testConnection,
    clearResults,
    isTesting,
    results,
  };
}

export default useConnectionTest;
