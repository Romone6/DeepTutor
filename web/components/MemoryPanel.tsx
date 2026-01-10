"use client";

import { useState, useEffect, useCallback } from "react";
import { Brain, Clock, TrendingDown, AlertCircle, CheckCircle, RotateCcw } from "lucide-react";
import { apiUrl } from "@/lib/api";

interface WeakSpot {
  subject: string;
  topic: string;
  count: number;
  avg_mastery: number;
  total_reviews: number;
}

interface ReviewItem {
  id: number;
  weakness_id: number;
  subject: string;
  topic: string;
  priority: number;
  scheduled_date: number;
  interval_hours: number;
}

interface MemoryStats {
  total_weaknesses: number;
  pending_reviews: number;
  tracked_topics: number;
  total_interactions: number;
}

interface InteractionStats {
  total_interactions: number;
  correct_count: number;
  accuracy: number;
  avg_duration_ms: number;
  days: number;
}

interface MemoryPanelProps {
  subject?: string;
}

export default function MemoryPanel({ subject }: MemoryPanelProps) {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [weakSpots, setWeakSpots] = useState<WeakSpot[]>([]);
  const [reviewQueue, setReviewQueue] = useState<ReviewItem[]>([]);
  const [interactionStats, setInteractionStats] = useState<InteractionStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"weakspots" | "review" | "stats">("weakspots");
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const baseUrl = apiUrl("/api/v1/memory");
      const headers = { "Content-Type": "application/json" };

      const [statsRes, weakSpotsRes, reviewRes, statsIntRes] = await Promise.all([
        fetch(`${baseUrl}/stats`, { headers }),
        fetch(`${baseUrl}/weak-spots${subject ? `?subject=${subject}` : ""}`, { headers }),
        fetch(`${baseUrl}/review-queue`, { headers }),
        fetch(`${baseUrl}/interaction-stats${subject ? `?subject=${subject}` : ""}`, { headers }),
      ]);

      if (!statsRes.ok) throw new Error("Failed to fetch stats");
      if (!weakSpotsRes.ok) throw new Error("Failed to fetch weak spots");
      if (!reviewRes.ok) throw new Error("Failed to fetch review queue");
      if (!statsIntRes.ok) throw new Error("Failed to fetch interaction stats");

      const [statsData, weakSpotsData, reviewData, statsIntData] = await Promise.all([
        statsRes.json(),
        weakSpotsRes.json(),
        reviewRes.json(),
        statsIntRes.json(),
      ]);

      setStats(statsData);
      setWeakSpots(weakSpotsData);
      setReviewQueue(reviewData);
      setInteractionStats(statsIntData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      console.error("Error fetching memory data:", err);
    } finally {
      setLoading(false);
    }
  }, [subject]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const completeReview = async (reviewId: number, quality: number, correct: boolean) => {
    try {
      const baseUrl = apiUrl("/api/v1/memory");
      const res = await fetch(`${baseUrl}/review/${reviewId}/complete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ quality, correct }),
      });

      if (!res.ok) throw new Error("Failed to complete review");

      fetchData();
    } catch (err) {
      console.error("Error completing review:", err);
    }
  };

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));

    if (diffHours < 0) return "Overdue";
    if (diffHours < 1) return `${Math.floor(diffMs / (1000 * 60))} min`;
    if (diffHours < 24) return `${diffHours} hours`;
    return `${Math.floor(diffHours / 24)} days`;
  };

  const getMasteryColor = (score: number) => {
    if (score >= 0.8) return "text-green-500";
    if (score >= 0.6) return "text-yellow-500";
    if (score >= 0.4) return "text-orange-500";
    return "text-red-500";
  };

  const getMasteryLabel = (score: number) => {
    if (score >= 0.8) return "Strong";
    if (score >= 0.6) return "Moderate";
    if (score >= 0.4) return "Weak";
    return "Very Weak";
  };

  if (loading && !stats) {
    return (
      <div className="p-4 animate-pulse">
        <div className="h-6 bg-gray-200 rounded w-1/3 mb-4"></div>
        <div className="space-y-3">
          <div className="h-4 bg-gray-200 rounded"></div>
          <div className="h-4 bg-gray-200 rounded w-5/6"></div>
          <div className="h-4 bg-gray-200 rounded w-4/6"></div>
        </div>
      </div>
    );
  }

  if (error && !stats) {
    return (
      <div className="p-4 text-red-500">
        <AlertCircle className="w-5 h-5 inline-block mr-2" />
        {error}
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-500" />
            <h3 className="font-semibold text-lg">Learning Memory</h3>
          </div>
          <button
            onClick={fetchData}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors"
            title="Refresh"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex border-b border-gray-200 dark:border-gray-700">
        {[
          { id: "weakspots", label: "Weak Spots", icon: TrendingDown },
          { id: "review", label: "Review Queue", icon: Clock },
          { id: "stats", label: "Statistics", icon: CheckCircle },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? "border-b-2 border-purple-500 text-purple-600 dark:text-purple-400"
                : "text-gray-500 hover:text-gray-700 dark:text-gray-400"
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      <div className="p-4">
        {activeTab === "weakspots" && (
          <div className="space-y-3">
            {weakSpots.length === 0 ? (
              <p className="text-gray-500 text-center py-8">No weak spots recorded yet</p>
            ) : (
              weakSpots.map((spot, index) => (
                <div
                  key={`${spot.subject}-${spot.topic}-${index}`}
                  className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg"
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{spot.topic}</span>
                      <span className="text-xs px-2 py-0.5 bg-gray-200 dark:bg-gray-600 rounded-full">
                        {spot.subject}
                      </span>
                    </div>
                    <div className="text-sm text-gray-500 mt-1">
                      {spot.count} issue{spot.count !== 1 ? "s" : ""} &bull;{" "}
                      {spot.total_reviews} review{spot.total_reviews !== 1 ? "s" : ""}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-lg font-semibold ${getMasteryColor(spot.avg_mastery)}`}>
                      {Math.round(spot.avg_mastery * 100)}%
                    </div>
                    <div className="text-xs text-gray-500">{getMasteryLabel(spot.avg_mastery)}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === "review" && (
          <div className="space-y-3">
            {reviewQueue.length === 0 ? (
              <p className="text-gray-500 text-center py-8">No items in review queue</p>
            ) : (
              reviewQueue.map((item) => (
                <div
                  key={item.id}
                  className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-medium">{item.topic}</div>
                    <div className="text-xs text-gray-500">{item.subject}</div>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-1 text-gray-500">
                      <Clock className="w-3 h-3" />
                      {formatDate(item.scheduled_date)}
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => completeReview(item.id, 5, true)}
                        className="px-3 py-1 bg-green-500 text-white rounded-full text-xs hover:bg-green-600 transition-colors"
                      >
                        Got it
                      </button>
                      <button
                        onClick={() => completeReview(item.id, 2, false)}
                        className="px-3 py-1 bg-red-500 text-white rounded-full text-xs hover:bg-red-600 transition-colors"
                      >
                        Still struggling
                      </button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === "stats" && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">{stats?.total_weaknesses || 0}</div>
                <div className="text-sm text-gray-500">Total Weaknesses</div>
              </div>
              <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{stats?.pending_reviews || 0}</div>
                <div className="text-sm text-gray-500">Pending Reviews</div>
              </div>
              <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{stats?.tracked_topics || 0}</div>
                <div className="text-sm text-gray-500">Tracked Topics</div>
              </div>
              <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                <div className="text-2xl font-bold text-orange-600">{stats?.total_interactions || 0}</div>
                <div className="text-sm text-gray-500">Total Interactions</div>
              </div>
            </div>

            {interactionStats && (
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <h4 className="font-medium mb-2">Recent Performance ({interactionStats.days} days)</h4>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-3xl font-bold">{interactionStats.accuracy.toFixed(1)}%</div>
                    <div className="text-sm text-gray-500">Accuracy</div>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold">{interactionStats.total_interactions}</div>
                    <div className="text-sm text-gray-500">Interactions</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
