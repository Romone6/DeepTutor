"use client";

import { useState, useEffect, useCallback } from "react";
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Clock,
  Target,
  AlertCircle,
  CheckCircle,
  BookOpen,
  Calendar,
  Filter,
  RefreshCw,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  ClipboardList,
} from "lucide-react";
import { apiUrl } from "@/lib/api";

interface DashboardSummary {
  period_days: number;
  subject: string | null;
  overall_stats: {
    total_questions: number;
    correct_questions: number;
    accuracy: number;
    total_study_time_minutes: number;
  };
  by_subject: Array<{
    subject: string;
    total_sessions: number;
    total_time_minutes: number;
    total_questions: number;
    correct_questions: number;
    accuracy: number;
    avg_accuracy: number;
    avg_confidence: number;
  }>;
  weak_topics: Array<{
    topic: string;
    subject: string;
    sessions: number;
    total_questions: number;
    correct_questions: number;
    avg_accuracy: number;
    avg_confidence: number;
  }>;
  accuracy_trend: Array<{
    date: string;
    sessions: number;
    total_questions: number;
    correct_questions: number;
    accuracy: number;
    avg_accuracy: number;
  }>;
  time_allocation: Array<{
    topic: string;
    subject: string;
    time_minutes: number;
    sessions: number;
    accuracy: number;
    percentage: number;
  }>;
}

interface ExamAnalytics {
  subject: string;
  days: number;
  session_count: number;
  average_percentage: number;
  average_marks: string;
  top_weak_topics: Array<{
    topic: string;
    times_identified: number;
    avg_score_percentage: number;
  }>;
}

interface ExamSessionSummary {
  id: string;
  subject: string;
  status: string;
  total_marks: number;
  earned_marks: number;
  percentage: number;
  created_at: string;
}

export default function DashboardPage() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [examAnalytics, setExamAnalytics] = useState<Record<string, ExamAnalytics>>({});
  const [recentExams, setRecentExams] = useState<ExamSessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSubject, setSelectedSubject] = useState<string>("");
  const [days, setDays] = useState(30);
  const [subjects, setSubjects] = useState<string[]>([]);

  const fetchDashboard = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const subjectParam = selectedSubject ? `&subject=${selectedSubject}` : "";
      const res = await fetch(
        apiUrl(`/api/v1/dashboard/summary?days=${days}${subjectParam}`),
      );

      if (!res.ok) throw new Error("Failed to fetch dashboard data");

      const data = await res.json();
      setSummary(data);

      const uniqueSubjects = Array.from(
        new Set(data.by_subject.map((s: any) => s.subject)),
      );
      setSubjects(uniqueSubjects);

      const examSubjects = ["mathematics", "biology", "business_studies", "legal_studies", "english_advanced"];
      const analyticsData: Record<string, ExamAnalytics> = {};
      const examsRes = await fetch(apiUrl("/api/v1/exam/recent?limit=5"));
      if (examsRes.ok) {
        const examsData = await examsRes.json();
        setRecentExams(examsData);
      }

      for (const subj of examSubjects) {
        try {
          const analyticsRes = await fetch(apiUrl(`/api/v1/exam/analytics/${subj}?days=${days}`));
          if (analyticsRes.ok) {
            const analytics = await analyticsRes.json();
            analyticsData[subj] = analytics;
          }
        } catch {
        }
      }
      setExamAnalytics(analyticsData);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [selectedSubject, days]);

  useEffect(() => {
    fetchDashboard();
  }, [fetchDashboard]);

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 80) return "text-emerald-600 dark:text-emerald-400";
    if (accuracy >= 60) return "text-amber-600 dark:text-amber-400";
    return "text-red-600 dark:text-red-400";
  };

  const getAccuracyBg = (accuracy: number) => {
    if (accuracy >= 80) return "bg-emerald-100 dark:bg-emerald-900/30";
    if (accuracy >= 60) return "bg-amber-100 dark:bg-amber-900/30";
    return "bg-red-100 dark:bg-red-900/30";
  };

  const getTrendIcon = (trend: number) => {
    if (trend > 2) return <ArrowUpRight className="w-4 h-4 text-emerald-500" />;
    if (trend < -2) return <ArrowDownRight className="w-4 h-4 text-red-500" />;
    return <Minus className="w-4 h-4 text-slate-400" />;
  };

  const formatTime = (minutes: number) => {
    if (minutes < 60) return `${Math.round(minutes)}m`;
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return `${hours}h ${mins}m`;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  };

  return (
    <div className="h-screen flex flex-col animate-fade-in">
      <div className="flex-1 overflow-y-auto p-6">
        {/* Header */}
        <div className="flex justify-between items-end mb-8">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 tracking-tight flex items-center gap-3">
              <BarChart3 className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              Progress Dashboard
            </h1>
            <p className="text-slate-500 dark:text-slate-400 mt-2">
              Track your learning progress by subject and topic.
            </p>
          </div>

          <div className="flex items-center gap-4">
            {/* Filters */}
            <div className="flex items-center gap-3 bg-white dark:bg-slate-800 px-4 py-2 rounded-xl border border-slate-200 dark:border-slate-700">
              <Filter className="w-4 h-4 text-slate-400" />
              <select
                value={selectedSubject}
                onChange={(e) => setSelectedSubject(e.target.value)}
                className="bg-transparent text-sm text-slate-700 dark:text-slate-300 outline-none"
              >
                <option value="">All Subjects</option>
                {subjects.map((s) => (
                  <option key={s} value={s}>
                    {s.charAt(0).toUpperCase() + s.slice(1)}
                  </option>
                ))}
              </select>

              <div className="w-px h-4 bg-slate-200 dark:bg-slate-600" />

              <select
                value={days}
                onChange={(e) => setDays(parseInt(e.target.value))}
                className="bg-transparent text-sm text-slate-700 dark:text-slate-300 outline-none"
              >
                <option value={7}>Last 7 days</option>
                <option value={14}>Last 14 days</option>
                <option value={30}>Last 30 days</option>
                <option value={90}>Last 90 days</option>
              </select>
            </div>

            <button
              onClick={fetchDashboard}
              disabled={loading}
              className="bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 px-4 py-2 rounded-xl text-sm font-medium hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors flex items-center gap-2 border border-slate-200 dark:border-slate-600"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </button>
          </div>
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400 p-4 rounded-xl border border-red-100 dark:border-red-800 mb-6 flex items-center gap-3">
            <AlertCircle className="w-5 h-5" />
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading && !summary && (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <RefreshCw className="w-8 h-8 text-blue-500 animate-spin mx-auto mb-4" />
              <p className="text-slate-500 dark:text-slate-400">Loading dashboard...</p>
            </div>
          </div>
        )}

        {/* Dashboard Content */}
        {summary && (
          <div className="space-y-6">
            {/* Overall Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-xl flex items-center justify-center">
                    <Target className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <span className={`text-xs font-medium px-2 py-1 rounded-full ${getAccuracyBg(summary.overall_stats.accuracy)} ${getAccuracyColor(summary.overall_stats.accuracy)}`}>
                    {summary.overall_stats.accuracy.toFixed(0)}%
                  </span>
                </div>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Overall Accuracy</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {summary.overall_stats.correct_questions}/{summary.overall_stats.total_questions}
                </p>
                <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">questions answered</p>
              </div>

              <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-10 h-10 bg-emerald-100 dark:bg-emerald-900/30 rounded-xl flex items-center justify-center">
                    <Clock className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                  </div>
                </div>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Study Time</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {formatTime(summary.overall_stats.total_study_time_minutes)}
                </p>
                <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">last {summary.period_days} days</p>
              </div>

              <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-xl flex items-center justify-center">
                    <BookOpen className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                  </div>
                </div>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Subjects Active</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {summary.by_subject.length}
                </p>
                <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">with recorded activity</p>
              </div>

              <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-10 h-10 bg-amber-100 dark:bg-amber-900/30 rounded-xl flex items-center justify-center">
                    <AlertCircle className="w-5 h-5 text-amber-600 dark:text-amber-400" />
                  </div>
                </div>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Weak Topics</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {summary.weak_topics.length}
                </p>
                <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">below 70% accuracy</p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Accuracy Trend Chart */}
              <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-6 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-blue-500" />
                  Accuracy Trend
                </h3>

                {summary.accuracy_trend.length > 0 ? (
                  <div className="space-y-3">
                    {summary.accuracy_trend.slice(-14).map((day, idx) => {
                      const prevDay = idx > 0 ? summary.accuracy_trend[summary.accuracy_trend.length - 14 + idx - 1] : null;
                      const trend = prevDay ? day.accuracy - prevDay.accuracy : 0;

                      return (
                        <div key={day.date} className="flex items-center gap-4">
                          <span className="text-xs text-slate-500 dark:text-slate-400 w-16">
                            {formatDate(day.date)}
                          </span>
                          <div className="flex-1">
                            <div className="h-6 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full transition-all ${getAccuracyBg(day.accuracy).replace("bg-", "bg-").replace("100", "400")} ${getAccuracyBg(day.accuracy).replace("dark:", "dark:")}`}
                                style={{ width: `${Math.min(day.accuracy, 100)}%` }}
                              />
                            </div>
                          </div>
                          <div className="flex items-center gap-2 w-24 justify-end">
                            <span className={`text-sm font-medium ${getAccuracyColor(day.accuracy)}`}>
                              {day.accuracy.toFixed(0)}%
                            </span>
                            {prevDay && getTrendIcon(trend)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-center py-12 text-slate-500 dark:text-slate-400">
                    <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-20" />
                    <p>No trend data available yet</p>
                    <p className="text-sm mt-1">Start studying to see your progress!</p>
                  </div>
                )}
              </div>

              {/* Weak Topics */}
              <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-6 flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 text-amber-500" />
                  Weak Topics
                </h3>

                {summary.weak_topics.length > 0 ? (
                  <div className="space-y-3">
                    {summary.weak_topics.slice(0, 8).map((topic, idx) => (
                      <div
                        key={idx}
                        className="p-3 bg-slate-50 dark:bg-slate-700/50 rounded-xl"
                      >
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <p className="font-medium text-slate-900 dark:text-slate-100 text-sm">
                              {topic.topic.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}
                            </p>
                            <p className="text-xs text-slate-500 dark:text-slate-400">
                              {topic.subject} • {topic.sessions} sessions
                            </p>
                          </div>
                          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${getAccuracyBg(topic.avg_accuracy)} ${getAccuracyColor(topic.avg_accuracy)}`}>
                            {topic.avg_accuracy.toFixed(0)}%
                          </span>
                        </div>
                        <div className="w-full bg-slate-200 dark:bg-slate-600 rounded-full h-1.5">
                          <div
                            className="bg-red-400 h-1.5 rounded-full transition-all"
                            style={{ width: `${Math.min(topic.avg_accuracy, 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12 text-slate-500 dark:text-slate-400">
                    <CheckCircle className="w-12 h-12 mx-auto mb-4 text-emerald-500 opacity-50" />
                    <p>No weak topics!</p>
                    <p className="text-sm mt-1">All topics above 70% accuracy</p>
                  </div>
                )}
              </div>
            </div>

            {/* Subject Breakdown */}
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-6 flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-emerald-500" />
                Subject Breakdown
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {summary.by_subject.map((subject, idx) => (
                  <div
                    key={idx}
                    className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-xl"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-slate-900 dark:text-slate-100 capitalize">
                        {subject.subject}
                      </h4>
                      <span className={`text-xs font-medium px-2 py-1 rounded-full ${getAccuracyBg(subject.accuracy)} ${getAccuracyColor(subject.accuracy)}`}>
                        {subject.accuracy.toFixed(0)}%
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <p className="text-slate-500 dark:text-slate-400">Accuracy</p>
                        <p className="font-medium text-slate-900 dark:text-slate-100">
                          {subject.correct_questions}/{subject.total_questions}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500 dark:text-slate-400">Time</p>
                        <p className="font-medium text-slate-900 dark:text-slate-100">
                          {formatTime(subject.total_time_minutes)}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500 dark:text-slate-400">Sessions</p>
                        <p className="font-medium text-slate-900 dark:text-slate-100">
                          {subject.total_sessions}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500 dark:text-slate-400">Confidence</p>
                        <p className="font-medium text-slate-900 dark:text-slate-100">
                          {(subject.avg_confidence * 20).toFixed(0)}%
                        </p>
                      </div>
                    </div>

                    <div className="mt-3 w-full bg-slate-200 dark:bg-slate-600 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all ${getAccuracyBg(subject.accuracy).replace("100", "400").replace("dark:", "dark:")}`}
                        style={{ width: `${Math.min(subject.accuracy, 100)}%` }}
                      />
                    </div>
                  </div>
                ))}

                {summary.by_subject.length === 0 && (
                  <div className="col-span-full text-center py-12 text-slate-500 dark:text-slate-400">
                    <BookOpen className="w-12 h-12 mx-auto mb-4 opacity-20" />
                    <p>No subject data yet</p>
                    <p className="text-sm mt-1">Start studying to see your progress!</p>
                  </div>
                )}
              </div>
            </div>

            {/* Exam Simulator */}
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-6 flex items-center gap-2">
                <ClipboardList className="w-5 h-5 text-blue-500" />
                Exam Simulator
              </h3>

              {recentExams.length > 0 ? (
                <div className="space-y-3">
                  {recentExams.map((exam) => {
                    const percentage = exam.percentage || 0;
                    return (
                      <div
                        key={exam.id}
                        className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-xl"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <span className={`w-10 h-10 rounded-lg flex items-center justify-center font-bold text-sm ${
                              percentage >= 80
                                ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600"
                                : percentage >= 60
                                ? "bg-amber-100 dark:bg-amber-900/30 text-amber-600"
                                : "bg-red-100 dark:bg-red-900/30 text-red-600"
                            }`}>
                              {percentage.toFixed(0)}%
                            </span>
                            <div>
                              <p className="font-medium text-slate-900 dark:text-slate-100 capitalize">
                                {exam.subject.replace(/_/g, " ")}
                              </p>
                              <p className="text-xs text-slate-500 dark:text-slate-400">
                                {new Date(exam.created_at).toLocaleDateString()}
                              </p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="font-medium text-slate-900 dark:text-slate-100">
                              {exam.earned_marks}/{exam.total_marks}
                            </p>
                            <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                              percentage >= 80
                                ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600"
                                : percentage >= 60
                                ? "bg-amber-100 dark:bg-amber-900/30 text-amber-600"
                                : "bg-red-100 dark:bg-red-900/30 text-red-600"
                            }`}>
                              {exam.status}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-12 text-slate-500 dark:text-slate-400">
                  <ClipboardList className="w-12 h-12 mx-auto mb-4 opacity-20" />
                  <p>No exams yet</p>
                  <p className="text-sm mt-1">Take an exam to see your results!</p>
                </div>
              )}

              {Object.keys(examAnalytics).length > 0 && (
                <div className="mt-6 pt-6 border-t border-slate-100 dark:border-slate-700">
                  <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-4">
                    Subject Performance
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {Object.entries(examAnalytics).map(([key, analytics]) => {
                      if (analytics.session_count === 0) return null;
                      const percentage = analytics.average_percentage || 0;
                      return (
                        <div
                          key={key}
                          className="p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg text-center"
                        >
                          <p className="font-medium text-slate-900 dark:text-slate-100 text-sm capitalize mb-1">
                            {key.replace(/_/g, " ")}
                          </p>
                          <p className={`text-xl font-bold ${
                            percentage >= 80
                              ? "text-emerald-600"
                              : percentage >= 60
                              ? "text-amber-600"
                              : "text-red-600"
                          }`}>
                            {percentage.toFixed(0)}%
                          </p>
                          <p className="text-xs text-slate-500 dark:text-slate-400">
                            {analytics.session_count} exam{analytics.session_count !== 1 ? "s" : ""}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            {/* Time Allocation */}
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-6 flex items-center gap-2">
                <Clock className="w-5 h-5 text-purple-500" />
                Time Allocation
              </h3>

              {summary.time_allocation.length > 0 ? (
                <div className="space-y-3">
                  {summary.time_allocation.slice(0, 10).map((item, idx) => (
                    <div key={idx} className="flex items-center gap-4">
                      <div className="w-48 shrink-0">
                        <p className="font-medium text-slate-900 dark:text-slate-100 text-sm truncate">
                          {item.topic.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}
                        </p>
                        <p className="text-xs text-slate-500 dark:text-slate-400 capitalize">
                          {item.subject}
                        </p>
                      </div>
                      <div className="flex-1">
                        <div className="h-8 bg-slate-100 dark:bg-slate-700 rounded-lg overflow-hidden flex">
                          <div
                            className="h-full bg-purple-400 dark:bg-purple-500 transition-all"
                            style={{ width: `${item.percentage}%` }}
                          />
                        </div>
                      </div>
                      <div className="w-32 text-right">
                        <p className="font-medium text-slate-900 dark:text-slate-100">
                          {formatTime(item.time_minutes)}
                        </p>
                        <p className="text-xs text-slate-500 dark:text-slate-400">
                          {item.percentage.toFixed(1)}%
                        </p>
                      </div>
                      <div className="w-16 text-right">
                        <span className={`text-xs font-medium ${getAccuracyColor(item.accuracy)}`}>
                          {item.accuracy.toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-slate-500 dark:text-slate-400">
                  <Clock className="w-12 h-12 mx-auto mb-4 opacity-20" />
                  <p>No time data yet</p>
                  <p className="text-sm mt-1">Start studying to see your time allocation!</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
