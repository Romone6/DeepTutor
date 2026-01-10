"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Calendar,
  Clock,
  Target,
  BookOpen,
  Plus,
  Play,
  Pause,
  CheckCircle,
  XCircle,
  ChevronRight,
  BarChart3,
  RefreshCw,
  Trash2,
  Loader2,
  TrendingUp,
  Award,
  Flame,
  Book,
  Brain,
  TestTube,
  ArrowRight,
} from "lucide-react";
import { apiUrl } from "@/lib/api";

interface SprintSubject {
  id: string;
  name: string;
  display_name: string;
}

interface SprintDay {
  day_number: number;
  date: string;
  subject: string;
  topics_covered: string[];
  activities: Array<{
    type: string;
    title: string;
    duration_minutes: number;
    description: string;
  }>;
  quiz: {
    questions: Array<{
      id: string;
      question: string;
      options: string[];
      correct_answer: number;
    }>;
  };
  checkpoint: {
    questions: Array<{
      id: string;
      question: string;
      answer: string;
      type: string;
    }>;
  };
  completion_status: string;
}

interface SprintPlan {
  id: string;
  name: string;
  subjects: string[];
  total_days: number;
  daily_hours: number;
  start_date: string;
  end_date: string;
  days: SprintDay[];
  topic_coverage: Record<string, number>;
  status: string;
  progress: {
    completed_days: number;
    total_days: number;
    percent_complete: number;
    quizzes_passed: number;
    total_quizzes: number;
    active_recall_score: number;
  };
}

interface CreateSprintRequest {
  subjects: string[];
  days: number;
  daily_hours: number;
  start_date?: string;
}

interface SessionOverview {
  title: string;
  summary: string;
  key_concepts: string[];
  learning_objectives: string[];
  estimated_time_minutes: number;
  resources: string[];
  previous_review_notes: string;
}

interface SessionCheckpoint {
  title: string;
  instructions: string;
  questions: Array<{
    id: string;
    topic: string;
    question: string;
    type: string;
    prompt: string;
  }>;
  estimated_time_minutes: number;
  guidance_tips: string[];
}

interface SessionQuiz {
  title: string;
  questions: Array<{
    id: string;
    question: string;
    options?: string[];
    topic_id: string;
    marks: number;
    type: string;
  }>;
  total_marks: number;
  time_limit_minutes: number;
  passing_score: number;
}

interface SprintSession {
  plan_id: string;
  day_number: number;
  date: string;
  subject: string;
  topics_covered: string[];
  overview: SessionOverview;
  checkpoint: SessionCheckpoint;
  quiz: SessionQuiz;
  current_phase: string;
}

interface SessionResult {
  success: boolean;
  quiz_score: number;
  quiz_total: number;
  percentage: number;
  weaknesses_found: Array<{ id: number; topic: string }>;
  review_scheduled: Array<{ id: number; topic: string; d2_review_id: number; d7_review_id: number }>;
  catch_up_suggestions: string[];
  next_session_date: string;
  message: string;
}

type SessionPhase = "overview" | "checkpoint" | "quiz" | "complete";

export default function SprintPage() {
  const [plans, setPlans] = useState<SprintPlan[]>([]);
  const [subjects, setSubjects] = useState<SprintSubject[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState<SprintPlan | null>(null);
  const [selectedDay, setSelectedDay] = useState<SprintDay | null>(null);

  const [currentSession, setCurrentSession] = useState<SprintSession | null>(null);
  const [sessionPhase, setSessionPhase] = useState<SessionPhase>("overview");
  const [quizAnswers, setQuizAnswers] = useState<Record<string, { answer: number; is_correct: boolean }>>({});
  const [checkpointRatings, setCheckpointRatings] = useState<Record<string, { understanding: number; confidence: number }>>({});
  const [sessionResult, setSessionResult] = useState<SessionResult | null>(null);
  const [sessionLoading, setSessionLoading] = useState(false);
  const [sessionModalOpen, setSessionModalOpen] = useState(false);

  const [newPlan, setNewPlan] = useState<CreateSprintRequest>({
    subjects: [],
    days: 21,
    daily_hours: 2.0,
  });

  const fetchSubjects = useCallback(async () => {
    try {
      const res = await fetch(apiUrl("/api/v1/sprint/subjects"));
      if (!res.ok) throw new Error("Failed to fetch subjects");
      const data = await res.json();
      setSubjects(data);
    } catch (err: any) {
      console.error("Error fetching subjects:", err);
    }
  }, []);

  const fetchPlans = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(apiUrl("/api/v1/sprint/list"));
      if (!res.ok) throw new Error("Failed to fetch sprint plans");
      const data = await res.json();
      setPlans(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchPlanDetails = useCallback(async (planId: string) => {
    try {
      const res = await fetch(apiUrl(`/api/v1/sprint/${planId}`));
      if (!res.ok) throw new Error("Failed to fetch plan details");
      const data = await res.json();
      setSelectedPlan(data);
      return data;
    } catch (err: any) {
      console.error("Error fetching plan details:", err);
      return null;
    }
  }, []);

  useEffect(() => {
    fetchSubjects();
    fetchPlans();
  }, [fetchSubjects, fetchPlans]);

  const handleCreateSprint = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newPlan.subjects.length === 0) {
      alert("Please select at least one subject");
      return;
    }

    setCreating(true);
    try {
      const res = await fetch(apiUrl("/api/v1/sprint/create"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newPlan),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Failed to create sprint plan");
      }

      const created = await res.json();
      setPlans((prev) => [created, ...prev]);
      setCreateModalOpen(false);
      setNewPlan({ subjects: [], days: 21, daily_hours: 2.0 });
    } catch (err: any) {
      alert(`Error creating sprint: ${err.message}`);
    } finally {
      setCreating(false);
    }
  };

  const handleUpdateDayStatus = async (
    planId: string,
    dayNumber: number,
    status: string,
  ) => {
    try {
      const res = await fetch(apiUrl(`/api/v1/sprint/${planId}/day/${dayNumber}`), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ completion_status: status }),
      });

      if (!res.ok) throw new Error("Failed to update day status");

      if (selectedPlan && selectedPlan.id === planId) {
        await fetchPlanDetails(planId);
      }
      fetchPlans();
    } catch (err: any) {
      console.error("Error updating day:", err);
    }
  };

  const handleDeletePlan = async (planId: string) => {
    if (!confirm("Are you sure you want to delete this sprint plan?")) return;

    try {
      const res = await fetch(apiUrl(`/api/v1/sprint/${planId}`), {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("Failed to delete plan");
      setPlans((prev) => prev.filter((p) => p.id !== planId));
      if (selectedPlan?.id === planId) {
        setSelectedPlan(null);
        setSelectedDay(null);
      }
    } catch (err: any) {
      alert(`Error deleting plan: ${err.message}`);
    }
  };

  const handleStartToday = async (planId: string) => {
    setSessionLoading(true);
    try {
      const res = await fetch(apiUrl(`/api/v1/sprint/${planId}/today`));
      if (!res.ok) throw new Error("Failed to load today's session");
      const data = await res.json();

      if (data.session) {
        setCurrentSession(data.session);
        setSessionPhase("overview");
        setQuizAnswers({});
        setCheckpointRatings({});
        setSessionResult(null);
        setSessionModalOpen(true);
      } else {
        alert("No session available for today. All caught up!");
      }
    } catch (err: any) {
      alert(`Error loading session: ${err.message}`);
    } finally {
      setSessionLoading(false);
    }
  };

  const handleCompleteSession = async () => {
    if (!currentSession) return;

    setSessionLoading(true);
    try {
      const quizAnswerList = Object.entries(quizAnswers).map(([qId, a]) => ({
        question_id: qId,
        answer: String(a.answer),
        is_correct: a.is_correct,
        time_spent_seconds: 30,
      }));

      const checkpointRatingList = Object.entries(checkpointRatings).map(([topic, r]) => ({
        topic,
        understanding_rating: r.understanding,
        confidence_rating: r.confidence,
        notes: "",
      }));

      const res = await fetch(
        apiUrl(`/api/v1/sprint/${currentSession.plan_id}/day/${currentSession.day_number}/complete`),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            quiz_answers: quizAnswerList,
            checkpoint_ratings: checkpointRatingList,
            total_time_seconds: 1800,
          }),
        },
      );

      if (!res.ok) throw new Error("Failed to complete session");

      const data = await res.json();
      setSessionResult(data.result);
      setSessionPhase("complete");

      await fetchPlanDetails(currentSession.plan_id);
      fetchPlans();
    } catch (err: any) {
      alert(`Error completing session: ${err.message}`);
    } finally {
      setSessionLoading(false);
    }
  };

  const handleNextPhase = () => {
    if (sessionPhase === "overview") setSessionPhase("checkpoint");
    else if (sessionPhase === "checkpoint") setSessionPhase("quiz");
    else if (sessionPhase === "quiz") handleCompleteSession();
  };

  const isPhaseComplete = () => {
    if (sessionPhase === "overview") return true;
    if (sessionPhase === "checkpoint") {
      return Object.keys(checkpointRatings).length >= (currentSession?.checkpoint.questions.length || 0) / 2;
    }
    if (sessionPhase === "quiz") {
      return Object.keys(quizAnswers).length >= (currentSession?.quiz.questions.length || 0);
    }
    return true;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400";
      case "in_progress":
        return "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400";
      case "pending":
        return "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400";
      default:
        return "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400";
    }
  };

  const getSubjectColor = (subject: string) => {
    const colors: Record<string, string> = {
      mathematics: "blue",
      physics: "purple",
      chemistry: "emerald",
      biology: "amber",
      computer_science: "rose",
    };
    return colors[subject] || "slate";
  };

  return (
    <div className="h-screen flex flex-col animate-fade-in">
      <div className="flex-1 overflow-y-auto p-6">
        {/* Header */}
        <div className="flex justify-between items-end mb-8">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 tracking-tight flex items-center gap-3">
              <Target className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              Study Sprint Planner
            </h1>
            <p className="text-slate-500 dark:text-slate-400 mt-2">
              Plan and track your 21-day learning sprints with active recall.
            </p>
          </div>
          <button
            onClick={() => {
              setNewPlan({ subjects: [], days: 21, daily_hours: 2.0 });
              setCreateModalOpen(true);
            }}
            className="bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 px-4 py-2 rounded-lg text-sm font-medium hover:bg-slate-800 dark:hover:bg-slate-200 transition-colors flex items-center gap-2 shadow-lg shadow-slate-900/20"
          >
            <Plus className="w-4 h-4" />
            New Sprint
          </button>
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400 p-4 rounded-xl border border-red-100 dark:border-red-800 mb-6 flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-red-500" />
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
          </div>
        )}

        {/* Main Content */}
        {!loading && (
          <div className="flex gap-6">
            {/* Plans List */}
            <div className="w-80 shrink-0">
              <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden">
                <div className="p-4 border-b border-slate-100 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-800/50">
                  <h2 className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                    <Calendar className="w-4 h-4" />
                    Your Sprints
                  </h2>
                </div>
                <div className="max-h-[calc(100vh-250px)] overflow-y-auto">
                  {plans.length === 0 ? (
                    <div className="p-8 text-center text-slate-500 dark:text-slate-400">
                      <Target className="w-12 h-12 mx-auto mb-4 opacity-20" />
                      <p className="text-sm">No sprint plans yet.</p>
                      <p className="text-xs mt-1">Create your first sprint to get started!</p>
                    </div>
                  ) : (
                    plans.map((plan) => (
                      <button
                        key={plan.id}
                        onClick={() => fetchPlanDetails(plan.id)}
                        className={`w-full p-4 text-left border-b border-slate-100 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors ${
                          selectedPlan?.id === plan.id ? "bg-blue-50 dark:bg-blue-900/30" : ""
                        }`}
                      >
                        <div className="flex justify-between items-start mb-2">
                          <h3 className="font-medium text-slate-900 dark:text-slate-100 text-sm">
                            {plan.name}
                          </h3>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeletePlan(plan.id);
                            }}
                            className="text-slate-400 hover:text-red-500 transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 mb-3">
                          <Clock className="w-3 h-3" />
                          <span>{plan.daily_hours}h/day</span>
                          <span className="text-slate-300 dark:text-slate-600">•</span>
                          <span>{plan.total_days} days</span>
                        </div>
                        <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-1.5 mb-2">
                          <div
                            className="bg-blue-500 h-1.5 rounded-full transition-all"
                            style={{ width: `${plan.progress.percent_complete}%` }}
                          />
                        </div>
                        <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400">
                          <span>{plan.progress.completed_days}/{plan.total_days} days</span>
                          <span>{plan.progress.percent_complete}%</span>
                        </div>
                      </button>
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* Plan Details */}
            <div className="flex-1">
              {selectedPlan ? (
                <div className="space-y-6">
                  {/* Plan Overview */}
                  <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
                    <div className="flex justify-between items-start mb-6">
                      <div>
                        <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-2">
                          {selectedPlan.name}
                        </h2>
                        <div className="flex items-center gap-4 text-sm text-slate-500 dark:text-slate-400">
                          <span className="flex items-center gap-1.5">
                            <Calendar className="w-4 h-4" />
                            {selectedPlan.start_date} → {selectedPlan.end_date}
                          </span>
                          <span className="flex items-center gap-1.5">
                            <Clock className="w-4 h-4" />
                            {selectedPlan.daily_hours}h/day
                          </span>
                        </div>
                      </div>
                      <span
                        className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(
                          selectedPlan.status,
                        )}`}
                      >
                        {selectedPlan.status.replace("_", " ")}
                      </span>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-3 mt-4">
                      <button
                        onClick={() => handleStartToday(selectedPlan.id)}
                        disabled={sessionLoading}
                        className="flex-1 py-2.5 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center gap-2 transition-colors"
                      >
                        {sessionLoading ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <>
                            <Play className="w-4 h-4" />
                            Start Today
                          </>
                        )}
                      </button>
                      <button
                        onClick={() => {
                          const today = selectedPlan.days.find(
                            (d) => d.completion_status !== "completed" && d.completion_status !== "in_progress"
                          );
                          if (today) {
                            setSelectedDay(today);
                          }
                        }}
                        className="px-4 py-2.5 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-xl font-medium hover:bg-slate-200 dark:hover:bg-slate-600 flex items-center gap-2 transition-colors"
                      >
                        <ChevronRight className="w-4 h-4" />
                        View Schedule
                      </button>
                    </div>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-4 gap-4">
                      <div className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded-xl">
                        <div className="flex items-center gap-2 mb-2">
                          <TrendingUp className="w-4 h-4 text-blue-500" />
                          <span className="text-xs text-slate-500 dark:text-slate-400 font-medium">
                            Progress
                          </span>
                        </div>
                        <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                          {selectedPlan.progress.percent_complete}%
                        </p>
                      </div>
                      <div className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded-xl">
                        <div className="flex items-center gap-2 mb-2">
                          <CheckCircle className="w-4 h-4 text-emerald-500" />
                          <span className="text-xs text-slate-500 dark:text-slate-400 font-medium">
                            Days Done
                          </span>
                        </div>
                        <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                          {selectedPlan.progress.completed_days}/{selectedPlan.total_days}
                        </p>
                      </div>
                      <div className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded-xl">
                        <div className="flex items-center gap-2 mb-2">
                          <Award className="w-4 h-4 text-amber-500" />
                          <span className="text-xs text-slate-500 dark:text-slate-400 font-medium">
                            Quizzes Passed
                          </span>
                        </div>
                        <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                          {selectedPlan.progress.quizzes_passed}/{selectedPlan.progress.total_quizzes}
                        </p>
                      </div>
                      <div className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded-xl">
                        <div className="flex items-center gap-2 mb-2">
                          <Flame className="w-4 h-4 text-orange-500" />
                          <span className="text-xs text-slate-500 dark:text-slate-400 font-medium">
                            Recall Score
                          </span>
                        </div>
                        <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                          {selectedPlan.progress.active_recall_score}%
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Days Timeline */}
                  <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden">
                    <div className="p-4 border-b border-slate-100 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-800/50">
                      <h3 className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                        <BarChart3 className="w-4 h-4" />
                        Daily Schedule
                      </h3>
                    </div>
                    <div className="divide-y divide-slate-100 dark:divide-slate-700 max-h-[calc(100vh-450px)] overflow-y-auto">
                      {selectedPlan.days.map((day) => (
                        <div
                          key={day.day_number}
                          className={`p-4 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors cursor-pointer ${
                            selectedDay?.day_number === day.day_number
                              ? "bg-blue-50 dark:bg-blue-900/30"
                              : ""
                          }`}
                          onClick={() => setSelectedDay(day)}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <div
                                className={`w-8 h-8 rounded-lg flex items-center justify-center text-sm font-medium ${
                                  day.completion_status === "completed"
                                    ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400"
                                    : day.completion_status === "in_progress"
                                    ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400"
                                    : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400"
                                }`}
                              >
                                {day.completion_status === "completed" ? (
                                  <CheckCircle className="w-4 h-4" />
                                ) : day.completion_status === "in_progress" ? (
                                  <Play className="w-4 h-4" />
                                ) : (
                                  day.day_number
                                )}
                              </div>
                              <div>
                                <p className="font-medium text-slate-900 dark:text-slate-100">
                                  Day {day.day_number} - {day.date}
                                </p>
                                <p className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-2">
                                  <BookOpen className="w-3 h-3" />
                                  <span
                                    className={`capitalize text-${getSubjectColor(day.subject)}-600 dark:text-${getSubjectColor(day.subject)}-400`}
                                  >
                                    {subjects.find((s) => s.id === day.subject)?.display_name || day.subject}
                                  </span>
                                  <span className="text-slate-300 dark:text-slate-600">•</span>
                                  <span>{day.topics_covered.length} topics</span>
                                </p>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              {day.completion_status !== "completed" && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleUpdateDayStatus(selectedPlan.id, day.day_number, "completed");
                                  }}
                                  className="px-3 py-1.5 bg-emerald-600 text-white rounded-lg text-xs font-medium hover:bg-emerald-700 transition-colors"
                                >
                                  Complete
                                </button>
                              )}
                              <ChevronRight className="w-4 h-4 text-slate-400" />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-12 text-center">
                  <Target className="w-16 h-16 mx-auto mb-4 text-slate-300 dark:text-slate-600" />
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                    Select a Sprint Plan
                  </h3>
                  <p className="text-slate-500 dark:text-slate-400">
                    Choose a sprint from the list or create a new one to get started.
                  </p>
                </div>
              )}
            </div>

            {/* Day Detail Panel */}
            {selectedDay && (
              <div className="w-96 shrink-0">
                <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden">
                  <div className="p-4 border-b border-slate-100 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-800/50 flex justify-between items-center">
                    <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                      Day {selectedDay.day_number} Details
                    </h3>
                    <button
                      onClick={() => setSelectedDay(null)}
                      className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                    >
                      <XCircle className="w-5 h-5" />
                    </button>
                  </div>
                  <div className="p-4 space-y-6 max-h-[calc(100vh-250px)] overflow-y-auto">
                    {/* Topics */}
                    <div>
                      <h4 className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-2 flex items-center gap-2">
                        <BookOpen className="w-4 h-4" />
                        Topics Covered
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {selectedDay.topics_covered.map((topic) => (
                          <span
                            key={topic}
                            className="px-2 py-1 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg text-xs"
                          >
                            {topic.replace(/_/g, " ")}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Activities */}
                    <div>
                      <h4 className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-2 flex items-center gap-2">
                        <Clock className="w-4 h-4" />
                        Activities
                      </h4>
                      <div className="space-y-2">
                        {selectedDay.activities.map((activity, idx) => (
                          <div
                            key={idx}
                            className="p-3 bg-slate-50 dark:bg-slate-700/50 rounded-xl"
                          >
                            <p className="font-medium text-slate-900 dark:text-slate-100 text-sm mb-1">
                              {activity.title}
                            </p>
                            <p className="text-xs text-slate-500 dark:text-slate-400">
                              {activity.type} • {activity.duration_minutes} min
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Quiz Preview */}
                    <div>
                      <h4 className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-2 flex items-center gap-2">
                        <Target className="w-4 h-4" />
                        Mini-Quiz
                      </h4>
                      <div className="p-3 bg-blue-50 dark:bg-blue-900/30 rounded-xl">
                        <p className="text-sm text-blue-700 dark:text-blue-300 font-medium">
                          {selectedDay.quiz.questions.length} questions
                        </p>
                        <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                          Test your understanding with a quick quiz after studying.
                        </p>
                      </div>
                    </div>

                    {/* Checkpoint */}
                    <div>
                      <h4 className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-2 flex items-center gap-2">
                        <RefreshCw className="w-4 h-4" />
                        Active Recall Checkpoint
                      </h4>
                      <div className="p-3 bg-amber-50 dark:bg-amber-900/30 rounded-xl">
                        <p className="text-sm text-amber-700 dark:text-amber-300 font-medium">
                          {selectedDay.checkpoint.questions.length} recall prompts
                        </p>
                        <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
                          Strengthen memory with active recall exercises.
                        </p>
                      </div>
                    </div>

                    {/* Status Update */}
                    <div className="pt-4 border-t border-slate-100 dark:border-slate-700">
                      <p className="text-sm text-slate-500 dark:text-slate-400 mb-2">
                        Mark this day as:
                      </p>
                      <div className="flex gap-2">
                        <button
                          onClick={() =>
                            handleUpdateDayStatus(
                              selectedPlan!.id,
                              selectedDay.day_number,
                              "in_progress",
                            )
                          }
                          className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                            selectedDay.completion_status === "in_progress"
                              ? "bg-blue-600 text-white"
                              : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600"
                          }`}
                        >
                          In Progress
                        </button>
                        <button
                          onClick={() =>
                            handleUpdateDayStatus(
                              selectedPlan!.id,
                              selectedDay.day_number,
                              "completed",
                            )
                          }
                          className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                            selectedDay.completion_status === "completed"
                              ? "bg-emerald-600 text-white"
                              : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600"
                          }`}
                        >
                          Completed
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Create Sprint Modal */}
      {createModalOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-2xl w-full max-w-lg p-6 animate-in zoom-in-95">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                <Target className="w-5 h-5 text-blue-500" />
                Create New Sprint
              </h3>
              <button
                onClick={() => setCreateModalOpen(false)}
                className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
              >
                <XCircle className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCreateSprint} className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Sprint Duration
                </label>
                <select
                  value={newPlan.days}
                  onChange={(e) =>
                    setNewPlan((prev) => ({ ...prev, days: parseInt(e.target.value) }))
                  }
                  className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
                >
                  <option value={7}>7 days (1 week)</option>
                  <option value={14}>14 days (2 weeks)</option>
                  <option value={21}>21 days (3 weeks)</option>
                  <option value={30}>30 days (1 month)</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Daily Study Time
                </label>
                <select
                  value={newPlan.daily_hours}
                  onChange={(e) =>
                    setNewPlan((prev) => ({ ...prev, daily_hours: parseFloat(e.target.value) }))
                  }
                  className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
                >
                  <option value={1.0}>1 hour/day</option>
                  <option value={1.5}>1.5 hours/day</option>
                  <option value={2.0}>2 hours/day</option>
                  <option value={2.5}>2.5 hours/day</option>
                  <option value={3.0}>3 hours/day</option>
                  <option value={4.0}>4 hours/day</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Select Subjects
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {subjects.map((subject) => (
                    <label
                      key={subject.id}
                      className={`flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-all ${
                        newPlan.subjects.includes(subject.id)
                          ? "border-blue-500 bg-blue-50 dark:bg-blue-900/30"
                          : "border-slate-200 dark:border-slate-600 hover:border-blue-300 dark:hover:border-blue-500"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={newPlan.subjects.includes(subject.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setNewPlan((prev) => ({
                              ...prev,
                              subjects: [...prev.subjects, subject.id],
                            }));
                          } else {
                            setNewPlan((prev) => ({
                              ...prev,
                              subjects: prev.subjects.filter((s) => s !== subject.id),
                            }));
                          }
                        }}
                        className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500"
                      />
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        {subject.display_name}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded-xl">
                <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Sprint Summary
                </h4>
                <div className="space-y-1 text-sm text-slate-600 dark:text-slate-400">
                  <p>
                    • {newPlan.days} days of focused learning
                  </p>
                  <p>
                    • {newPlan.daily_hours} hours per day
                  </p>
                  <p>
                    • {newPlan.subjects.length} subject{newPlan.subjects.length !== 1 ? "s" : ""}{" "}
                    selected
                  </p>
                  <p>• Daily mini-quizzes and active recall checkpoints</p>
                </div>
              </div>

              <div className="flex gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setCreateModalOpen(false)}
                  className="flex-1 py-2.5 rounded-xl border border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300 font-medium hover:bg-slate-50 dark:hover:bg-slate-700"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={newPlan.subjects.length === 0 || creating}
                  className="flex-1 py-2.5 rounded-xl bg-blue-600 text-white font-medium hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {creating ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <>
                      <Target className="w-4 h-4" />
                      Create Sprint
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Session Runner Modal */}
      {sessionModalOpen && currentSession && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden animate-in zoom-in-95">
            {/* Modal Header */}
            <div className="p-6 border-b border-slate-100 dark:border-slate-700 bg-gradient-to-r from-blue-500 to-indigo-600 text-white">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="text-xl font-bold flex items-center gap-2">
                    {sessionPhase === "overview" && <Book className="w-5 h-5" />}
                    {sessionPhase === "checkpoint" && <Brain className="w-5 h-5" />}
                    {sessionPhase === "quiz" && <TestTube className="w-5 h-5" />}
                    {sessionPhase === "complete" && <CheckCircle className="w-5 h-5" />}
                    {currentSession.overview.title}
                  </h3>
                  <p className="text-blue-100 text-sm mt-1">
                    Day {currentSession.day_number} • {currentSession.subject} • ~30 min
                  </p>
                </div>
                <button
                  onClick={() => setSessionModalOpen(false)}
                  className="text-white/80 hover:text-white"
                >
                  <XCircle className="w-6 h-6" />
                </button>
              </div>

              {/* Progress Steps */}
              <div className="flex items-center gap-2 mt-4">
                {["overview", "checkpoint", "quiz", "complete"].map((phase, idx) => {
                  const phases: SessionPhase[] = ["overview", "checkpoint", "quiz", "complete"];
                  const currentIdx = phases.indexOf(sessionPhase);
                  const isComplete = idx < currentIdx;
                  const isCurrent = idx === currentIdx;

                  return (
                    <div key={phase} className="flex items-center">
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium ${
                          isComplete
                            ? "bg-emerald-400 text-emerald-900"
                            : isCurrent
                            ? "bg-white text-blue-600"
                            : "bg-blue-400/50 text-blue-200"
                        }`}
                      >
                        {isComplete ? <CheckCircle className="w-4 h-4" /> : idx + 1}
                      </div>
                      {idx < 3 && (
                        <div
                          className={`w-12 h-0.5 ${
                            isComplete ? "bg-emerald-400" : "bg-blue-400/50"
                          }`}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Modal Content */}
            <div className="p-6 max-h-[60vh] overflow-y-auto">
              {/* Overview Phase */}
              {sessionPhase === "overview" && (
                <div className="space-y-6">
                  <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-xl">
                    <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">
                      Session Summary
                    </h4>
                    <p className="text-slate-700 dark:text-slate-300">
                      {currentSession.overview.summary}
                    </p>
                  </div>

                  <div>
                    <h4 className="font-semibold text-slate-900 dark:text-slate-100 mb-3 flex items-center gap-2">
                      <Target className="w-4 h-4" />
                      Key Concepts
                    </h4>
                    <div className="grid grid-cols-2 gap-2">
                      {currentSession.overview.key_concepts.map((concept, idx) => (
                        <div
                          key={idx}
                          className="p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg text-sm text-slate-700 dark:text-slate-300"
                        >
                          {concept}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold text-slate-900 dark:text-slate-100 mb-3 flex items-center gap-2">
                      <TrendingUp className="w-4 h-4" />
                      Learning Objectives
                    </h4>
                    <ul className="space-y-2">
                      {currentSession.overview.learning_objectives.map((obj, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-slate-700 dark:text-slate-300">
                          <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                          {obj}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {currentSession.overview.previous_review_notes && (
                    <div className="bg-amber-50 dark:bg-amber-900/30 p-4 rounded-xl">
                      <p className="text-amber-700 dark:text-amber-300 text-sm">
                        {currentSession.overview.previous_review_notes}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Checkpoint Phase */}
              {sessionPhase === "checkpoint" && (
                <div className="space-y-6">
                  <div className="bg-amber-50 dark:bg-amber-900/30 p-4 rounded-xl">
                    <p className="text-amber-700 dark:text-amber-300 text-sm">
                      {currentSession.checkpoint.instructions}
                    </p>
                  </div>

                  {currentSession.checkpoint.questions
                    .filter((q) => q.type === "self-assessment")
                    .map((q, idx) => (
                      <div key={q.id} className="space-y-3">
                        <label className="block font-medium text-slate-900 dark:text-slate-100">
                          {q.question}
                        </label>
                        <div className="flex gap-2">
                          {[1, 2, 3, 4, 5].map((rating) => (
                            <button
                              key={rating}
                              onClick={() =>
                                setCheckpointRatings((prev) => ({
                                  ...prev,
                                  [q.topic]: {
                                    ...prev[q.topic],
                                    understanding: rating,
                                    confidence: rating,
                                  },
                                }))
                              }
                              className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
                                checkpointRatings[q.topic]?.understanding === rating
                                  ? "bg-blue-600 text-white"
                                  : "bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600"
                              }`}
                            >
                              {rating}
                            </button>
                          ))}
                        </div>
                        <p className="text-xs text-slate-500">
                          1 = Very uncertain → 5 = Very confident
                        </p>
                      </div>
                    ))}

                  <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-xl">
                    <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">
                      Guidance Tips
                    </h4>
                    <ul className="space-y-1 text-sm text-blue-600 dark:text-blue-400">
                      {currentSession.checkpoint.guidance_tips.map((tip, idx) => (
                        <li key={idx}>• {tip}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {/* Quiz Phase */}
              {sessionPhase === "quiz" && (
                <div className="space-y-6">
                  <div className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded-xl flex justify-between items-center">
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      {currentSession.quiz.questions.length} questions •{" "}
                      {currentSession.quiz.time_limit_minutes} min limit
                    </span>
                    <span className="text-sm font-medium text-slate-900 dark:text-slate-100">
                      Passing: {currentSession.quiz.passing_score}%
                    </span>
                  </div>

                  {currentSession.quiz.questions.map((q, idx) => (
                    <div key={q.id} className="space-y-3">
                      <p className="font-medium text-slate-900 dark:text-slate-100">
                        {idx + 1}. {q.question}
                      </p>
                      {q.options && (
                        <div className="space-y-2">
                          {q.options.map((option, optIdx) => (
                            <button
                              key={optIdx}
                              onClick={() =>
                                setQuizAnswers((prev) => ({
                                  ...prev,
                                  [q.id]: { answer: optIdx, is_correct: optIdx === 0 },
                                }))
                              }
                              className={`w-full p-3 text-left rounded-lg transition-colors ${
                                quizAnswers[q.id]?.answer === optIdx
                                  ? "bg-blue-100 dark:bg-blue-900/30 border-2 border-blue-500 text-blue-700 dark:text-blue-300"
                                  : "bg-slate-50 dark:bg-slate-700/50 border-2 border-transparent hover:bg-slate-100 dark:hover:bg-slate-700"
                              }`}
                            >
                              {String.fromCharCode(65 + optIdx)}. {option}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Complete Phase */}
              {sessionPhase === "complete" && sessionResult && (
                <div className="space-y-6 text-center">
                  <div
                    className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center ${
                      sessionResult.percentage >= 70
                        ? "bg-emerald-100 dark:bg-emerald-900/30"
                        : "bg-amber-100 dark:bg-amber-900/30"
                    }`}
                  >
                    {sessionResult.percentage >= 70 ? (
                      <CheckCircle className="w-10 h-10 text-emerald-600 dark:text-emerald-400" />
                    ) : (
                      <TrendingUp className="w-10 h-10 text-amber-600 dark:text-amber-400" />
                    )}
                  </div>

                  <div>
                    <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">
                      {sessionResult.percentage >= 70 ? "Great Job!" : "Keep Learning!"}
                    </h3>
                    <p className="text-slate-600 dark:text-slate-400">{sessionResult.message}</p>
                  </div>

                  <div className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded-xl">
                    <p className="text-sm text-slate-500 dark:text-slate-400 mb-2">Quiz Score</p>
                    <p className="text-3xl font-bold text-slate-900 dark:text-slate-100">
                      {sessionResult.quiz_score}/{sessionResult.quiz_total} ({sessionResult.percentage.toFixed(0)}%)
                    </p>
                  </div>

                  {sessionResult.weaknesses_found.length > 0 && (
                    <div className="bg-amber-50 dark:bg-amber-900/30 p-4 rounded-xl text-left">
                      <h4 className="font-semibold text-amber-700 dark:text-amber-300 mb-2">
                        Topics to Review
                      </h4>
                      <p className="text-sm text-amber-600 dark:text-amber-400">
                        Reviews have been scheduled for D+2 and D+7 to help strengthen these areas.
                      </p>
                    </div>
                  )}

                  {sessionResult.catch_up_suggestions.length > 0 && (
                    <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-xl text-left">
                      <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">
                        Suggestions for Improvement
                      </h4>
                      <ul className="space-y-1 text-sm text-blue-600 dark:text-blue-400">
                        {sessionResult.catch_up_suggestions.map((s, idx) => (
                          <li key={idx}>• {s}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="p-6 border-t border-slate-100 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 flex justify-between">
              <button
                onClick={() => setSessionModalOpen(false)}
                className="px-4 py-2 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100"
              >
                {sessionPhase === "complete" ? "Close" : "Cancel"}
              </button>

              {sessionPhase !== "complete" && (
                <button
                  onClick={handleNextPhase}
                  disabled={!isPhaseComplete() || sessionLoading}
                  className="px-6 py-2.5 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
                >
                  {sessionLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <>
                      {sessionPhase === "quiz" ? "Complete Session" : "Continue"}
                      <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
