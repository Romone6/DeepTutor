"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  ClipboardList,
  Clock,
  CheckCircle,
  XCircle,
  Play,
  Pause,
  RotateCcw,
  ChevronRight,
  ChevronLeft,
  AlertCircle,
  Target,
  BarChart3,
  Brain,
  Timer,
} from "lucide-react";
import { apiUrl } from "@/lib/api";

type ExamSubject = "mathematics" | "biology" | "business_studies" | "legal_studies" | "english_advanced";
type DifficultyLevel = "easy" | "medium" | "hard" | "exam_standard";
type QuestionType = "short_answer" | "extended_response" | "essay" | "calculation";
type SessionStatus = "created" | "in_progress" | "submitted" | "marked" | "completed";

interface ExamQuestion {
  id: string;
  question_type: QuestionType;
  topic: string;
  command_term: string;
  question_text: string;
  marks: number;
  rubric: {
    criteria: string[];
    structure_elements: string[];
  };
}

interface StudentAnswer {
  question_id: string;
  answer_text: string;
  time_spent_seconds: number;
  started_at: string;
  submitted_at: string;
}

interface QuestionMarking {
  question_id: string;
  structure_score: number;
  depth_score: number;
  overall_score: number;
  total_marks: number;
  feedback: string;
  weak_topics: string[];
}

interface ExamSession {
  id: string;
  subject: ExamSubject;
  time_limit_minutes: number;
  topic_scope: string;
  difficulty: DifficultyLevel;
  question_count: number;
  status: SessionStatus;
  questions: ExamQuestion[];
  answers: StudentAnswer[];
  markings: QuestionMarking[];
  created_at: string;
  started_at: string | null;
  submitted_at: string | null;
  marked_at: string | null;
}

interface ExamSettings {
  subject: ExamSubject;
  time_limit_minutes: number;
  difficulty: DifficultyLevel;
  topic_scope: string;
  question_count: number;
}

const SUBJECT_INFO: Record<ExamSubject, { name: string; icon: string; topics: string[] }> = {
  mathematics: {
    name: "Mathematics",
    icon: "📐",
    topics: ["calculus", "algebra", "probability", "statistics", "geometry", "functions", "sequences"],
  },
  biology: {
    name: "Biology",
    icon: "🧬",
    topics: ["cells", "genetics", "evolution", "ecology", "human_body", "plants", "microbiology"],
  },
  business_studies: {
    name: "Business Studies",
    icon: "💼",
    topics: ["marketing", "finance", "operations", "human_resources", "entrepreneurship", "globalization"],
  },
  legal_studies: {
    name: "Legal Studies",
    icon: "⚖️",
    topics: ["criminal_law", "civil_law", "torts", "contract_law", "constitutional_law", "human_rights"],
  },
  english_advanced: {
    name: "English Advanced",
    icon: "📚",
    topics: ["novel", "poetry", "drama", "nonfiction", "creative_writing", "analysis", "language"],
  },
};

const DIFFICULTY_INFO: Record<DifficultyLevel, { name: string; description: string }> = {
  easy: { name: "Easy", description: "Basic understanding" },
  medium: { name: "Medium", description: "Solid knowledge required" },
  hard: { name: "Hard", description: "Deep understanding needed" },
  exam_standard: { name: "Exam Standard", description: "HSC exam level" },
};

export default function ExamPage() {
  const [view, setView] = useState<"settings" | "taking" | "results">("settings");
  const [settings, setSettings] = useState<ExamSettings>({
    subject: "mathematics",
    time_limit_minutes: 60,
    difficulty: "exam_standard",
    topic_scope: "",
    question_count: 5,
  });
  const [session, setSession] = useState<ExamSession | null>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [timeRemaining, setTimeRemaining] = useState<number>(0);
  const [timerRunning, setTimerRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const currentQuestion = session?.questions[currentQuestionIndex];

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const calculateTimeSpent = useCallback((questionId: string) => {
    if (!session) return 0;
    const answer = session.answers.find((a) => a.question_id === questionId);
    return answer?.time_spent_seconds || 0;
  }, [session]);

  useEffect(() => {
    if (timerRunning && timeRemaining > 0) {
      timerRef.current = setInterval(() => {
        setTimeRemaining((prev) => Math.max(0, prev - 1));
      }, 1000);
    } else if (timeRemaining === 0 && timerRunning) {
      setTimerRunning(false);
      handleSubmitExam();
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [timerRunning, timeRemaining]);

  const startExam = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(apiUrl("/api/v1/exam/create"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });
      if (!res.ok) throw new Error("Failed to create exam session");
      const data = await res.json();

      const startRes = await fetch(apiUrl(`/api/v1/exam/${data.id}/start`), {
        method: "POST",
      });
      if (!startRes.ok) throw new Error("Failed to start exam");

      const sessionData = await startRes.json();

      setSession(sessionData);
      setTimeRemaining(settings.time_limit_minutes * 60);
      setTimerRunning(true);
      setView("taking");
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAnswerChange = (questionId: string, value: string) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

  const nextQuestion = () => {
    if (session && currentQuestionIndex < session.questions.length - 1) {
      setCurrentQuestionIndex((prev) => prev + 1);
    }
  };

  const prevQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex((prev) => prev - 1);
    }
  };

  const handleSubmitExam = async () => {
    if (!session) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(apiUrl(`/api/v1/exam/${session.id}/submit`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answers }),
      });
      if (!res.ok) throw new Error("Failed to submit exam");

      const markRes = await fetch(apiUrl(`/api/v1/exam/${session.id}/mark`), {
        method: "POST",
      });
      if (!markRes.ok) throw new Error("Failed to mark exam");

      const results = await markRes.json();
      setSession(results);
      setView("results");
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
      setTimerRunning(false);
    }
  };

  const resetExam = () => {
    setView("settings");
    setSession(null);
    setCurrentQuestionIndex(0);
    setAnswers({});
    setTimeRemaining(0);
    setTimerRunning(false);
  };

  const getScoreColor = (score: number, total: number) => {
    const percentage = (score / total) * 100;
    if (percentage >= 80) return "text-emerald-600";
    if (percentage >= 60) return "text-amber-600";
    return "text-red-600";
  };

  return (
    <div className="h-screen flex flex-col animate-fade-in">
      <div className="flex-1 overflow-y-auto p-6">
        {view === "settings" && (
          <>
            <div className="max-w-2xl mx-auto">
              <div className="mb-8">
                <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 tracking-tight flex items-center gap-3">
                  <ClipboardList className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                  Exam Simulator
                </h1>
                <p className="text-slate-500 dark:text-slate-400 mt-2">
                  Practice under timed exam conditions with instant feedback and weakness analysis.
                </p>
              </div>

              {error && (
                <div className="bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400 p-4 rounded-xl border border-red-100 dark:border-red-800 mb-6 flex items-center gap-3">
                  <AlertCircle className="w-5 h-5" />
                  {error}
                </div>
              )}

              <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6 space-y-6">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
                    Subject
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {Object.entries(SUBJECT_INFO).map(([key, info]) => (
                      <button
                        key={key}
                        onClick={() =>
                          setSettings((prev) => ({
                            ...prev,
                            subject: key as ExamSubject,
                            topic_scope: "",
                          }))
                        }
                        className={`p-4 rounded-xl border text-left transition-all ${
                          settings.subject === key
                            ? "border-blue-500 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
                            : "border-slate-200 dark:border-slate-600 hover:border-slate-300 dark:hover:border-slate-500"
                        }`}
                      >
                        <span className="text-2xl">{info.icon}</span>
                        <p className="font-medium mt-2">{info.name}</p>
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
                    Topic
                  </label>
                  <select
                    value={settings.topic_scope}
                    onChange={(e) => setSettings((prev) => ({ ...prev, topic_scope: e.target.value }))}
                    className="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">All {SUBJECT_INFO[settings.subject].name} Topics</option>
                    {SUBJECT_INFO[settings.subject].topics.map((topic) => (
                      <option key={topic} value={topic}>
                        {topic.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
                      Time Limit
                    </label>
                    <select
                      value={settings.time_limit_minutes}
                      onChange={(e) =>
                        setSettings((prev) => ({
                          ...prev,
                          time_limit_minutes: parseInt(e.target.value),
                        }))
                      }
                      className="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value={30}>30 minutes</option>
                      <option value={45}>45 minutes</option>
                      <option value={60}>60 minutes</option>
                      <option value={90}>90 minutes</option>
                      <option value={120}>120 minutes</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
                      Difficulty
                    </label>
                    <select
                      value={settings.difficulty}
                      onChange={(e) =>
                        setSettings((prev) => ({
                          ...prev,
                          difficulty: e.target.value as DifficultyLevel,
                        }))
                      }
                      className="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      {Object.entries(DIFFICULTY_INFO).map(([key, info]) => (
                        <option key={key} value={key}>
                          {info.name} - {info.description}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
                    Number of Questions: {settings.question_count}
                  </label>
                  <input
                    type="range"
                    min={3}
                    max={10}
                    value={settings.question_count}
                    onChange={(e) =>
                      setSettings((prev) => ({
                        ...prev,
                        question_count: parseInt(e.target.value),
                      }))
                    }
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mt-1">
                    <span>3</span>
                    <span>10</span>
                  </div>
                </div>

                <button
                  onClick={startExam}
                  disabled={loading}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-4 rounded-xl transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {loading ? (
                    <RotateCcw className="w-5 h-5 animate-spin" />
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      Start Exam
                    </>
                  )}
                </button>
              </div>
            </div>
          </>
        )}

        {view === "taking" && session && currentQuestion && (
          <>
            <div className="max-w-4xl mx-auto">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-4">
                  <span className="text-2xl">{SUBJECT_INFO[session.subject].icon}</span>
                  <div>
                    <h1 className="text-xl font-bold text-slate-900 dark:text-slate-100">
                      {SUBJECT_INFO[session.subject].name} Exam
                    </h1>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      Question {currentQuestionIndex + 1} of {session.questions.length}
                    </p>
                  </div>
                </div>

                <div
                  className={`flex items-center gap-3 px-6 py-3 rounded-xl font-mono text-lg ${
                    timeRemaining < 300
                      ? "bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400"
                      : "bg-slate-100 dark:bg-slate-700"
                  }`}
                >
                  <Timer
                    className={`w-5 h-5 ${timeRemaining < 300 ? "animate-pulse" : ""}`}
                  />
                  {formatTime(timeRemaining)}
                </div>
              </div>

              <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
                {session.questions.map((q, idx) => (
                  <button
                    key={q.id}
                    onClick={() => setCurrentQuestionIndex(idx)}
                    className={`w-10 h-10 rounded-lg font-medium text-sm transition-all flex-shrink-0 ${
                      idx === currentQuestionIndex
                        ? "bg-blue-600 text-white"
                        : answers[q.id]
                        ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400"
                        : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
                    }`}
                  >
                    {idx + 1}
                  </button>
                ))}
              </div>

              <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6 mb-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 mb-3">
                      {currentQuestion.question_type.replace(/_/g, " ")}
                      <span className="w-px h-3 bg-blue-300 dark:bg-blue-600" />
                      {currentQuestion.marks} mark{currentQuestion.marks > 1 ? "s" : ""}
                    </span>
                    <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                      {currentQuestion.question_text}
                    </h2>
                  </div>
                </div>

                <div className="bg-slate-50 dark:bg-slate-700/50 rounded-xl p-4 mb-4">
                  <p className="text-sm text-slate-600 dark:text-slate-300">
                    <span className="font-medium">Topic:</span>{" "}
                    {currentQuestion.topic.replace(/_/g, " ")}
                  </p>
                  <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                    <span className="font-medium">Command term:</span>{" "}
                    {currentQuestion.command_term}
                  </p>
                </div>

                {currentQuestion.question_type === "essay" ? (
                  <textarea
                    value={answers[currentQuestion.id] || ""}
                    onChange={(e) => handleAnswerChange(currentQuestion.id, e.target.value)}
                    placeholder="Write your response..."
                    className="w-full h-64 px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                ) : (
                  <textarea
                    value={answers[currentQuestion.id] || ""}
                    onChange={(e) => handleAnswerChange(currentQuestion.id, e.target.value)}
                    placeholder="Write your answer..."
                    className="w-full h-40 px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                )}
              </div>

              <div className="flex items-center justify-between">
                <button
                  onClick={prevQuestion}
                  disabled={currentQuestionIndex === 0}
                  className="flex items-center gap-2 px-6 py-3 rounded-xl border border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ChevronLeft className="w-5 h-5" />
                  Previous
                </button>

                {currentQuestionIndex === session.questions.length - 1 ? (
                  <button
                    onClick={handleSubmitExam}
                    disabled={loading}
                    className="flex items-center gap-2 px-6 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-medium disabled:opacity-50"
                  >
                    <CheckCircle className="w-5 h-5" />
                    Submit Exam
                  </button>
                ) : (
                  <button
                    onClick={nextQuestion}
                    className="flex items-center gap-2 px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-medium"
                  >
                    Next
                    <ChevronRight className="w-5 h-5" />
                  </button>
                )}
              </div>
            </div>
          </>
        )}

        {view === "results" && session && (
          <>
            <div className="max-w-4xl mx-auto">
              <div className="mb-8">
                <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 tracking-tight flex items-center gap-3">
                  <BarChart3 className="w-8 h-8 text-emerald-600 dark:text-emerald-400" />
                  Exam Results
                </h1>
                <p className="text-slate-500 dark:text-slate-400 mt-2">
                  {SUBJECT_INFO[session.subject].name} - {session.questions.length} questions
                </p>
              </div>

              {error && (
                <div className="bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400 p-4 rounded-xl border border-red-100 dark:border-red-800 mb-6 flex items-center gap-3">
                  <AlertCircle className="w-5 h-5" />
                  {error}
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-xl flex items-center justify-center">
                      <Target className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                    </div>
                  </div>
                  <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Total Score</p>
                  <p className={`text-3xl font-bold ${getScoreColor(
                    session.markings.reduce((sum, m) => sum + m.overall_score, 0),
                    session.markings.reduce((sum, m) => sum + m.total_marks, 0)
                  )}`}>
                    {session.markings.reduce((sum, m) => sum + m.overall_score, 0)}/
                    {session.markings.reduce((sum, m) => sum + m.total_marks, 0)}
                  </p>
                  <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
                    {((session.markings.reduce((sum, m) => sum + m.overall_score, 0) /
                      session.markings.reduce((sum, m) => sum + m.total_marks, 0)) * 100).toFixed(1)}%
                  </p>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-10 h-10 bg-emerald-100 dark:bg-emerald-900/30 rounded-xl flex items-center justify-center">
                      <Clock className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                    </div>
                  </div>
                  <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Time Used</p>
                  <p className="text-3xl font-bold text-slate-900 dark:text-slate-100">
                    {formatTime(
                      settings.time_limit_minutes * 60 - timeRemaining
                    )}
                  </p>
                  <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
                    of {settings.time_limit_minutes} minutes
                  </p>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-xl flex items-center justify-center">
                      <Brain className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                    </div>
                  </div>
                  <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Weak Topics</p>
                  <p className="text-3xl font-bold text-slate-900 dark:text-slate-100">
                    {session.markings.filter(m => m.weak_topics.length > 0).length}
                  </p>
                  <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
                    identified from responses
                  </p>
                </div>
              </div>

              <div className="space-y-6">
                {session.questions.map((question, idx) => {
                  const marking = session.markings.find((m) => m.question_id === question.id);
                  const answer = answers[question.id];

                  return (
                    <div
                      key={question.id}
                      className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 p-6"
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex items-center gap-3">
                          <span
                            className={`w-8 h-8 rounded-lg flex items-center justify-center font-bold text-sm ${
                              marking && marking.overall_score >= marking.total_marks * 0.7
                                ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600"
                                : marking && marking.overall_score >= marking.total_marks * 0.4
                                ? "bg-amber-100 dark:bg-amber-900/30 text-amber-600"
                                : "bg-red-100 dark:bg-red-900/30 text-red-600"
                            }`}
                          >
                            {marking?.overall_score || 0}/{marking?.total_marks || 0}
                          </span>
                          <div>
                            <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                              Question {idx + 1}
                            </h3>
                            <p className="text-sm text-slate-500 dark:text-slate-400">
                              {question.topic.replace(/_/g, " ")} • {question.command_term}
                            </p>
                          </div>
                        </div>
                        <span className="text-xs text-slate-400">{question.question_type}</span>
                      </div>

                      <div className="bg-slate-50 dark:bg-slate-700/50 rounded-xl p-4 mb-4">
                        <p className="text-slate-700 dark:text-slate-300">{question.question_text}</p>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2 uppercase tracking-wide">
                            Your Answer
                          </p>
                          <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-3 text-sm text-slate-700 dark:text-slate-300 min-h-[80px]">
                            {answer || "No answer provided"}
                          </div>
                        </div>

                        <div>
                          <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2 uppercase tracking-wide">
                            Feedback
                          </p>
                          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-3 text-sm text-emerald-700 dark:text-emerald-300 min-h-[80px]">
                            {marking?.feedback || "No feedback available"}
                          </div>
                        </div>
                      </div>

                      {marking && marking.weak_topics.length > 0 && (
                        <div className="mt-4 flex flex-wrap gap-2">
                          {marking.weak_topics.map((topic) => (
                            <span
                              key={topic}
                              className="px-3 py-1 rounded-full text-xs bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400"
                            >
                              {topic.replace(/_/g, " ")}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              <div className="mt-8 flex justify-center">
                <button
                  onClick={resetExam}
                  className="flex items-center gap-2 px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-medium"
                >
                  <RotateCcw className="w-5 h-5" />
                  New Exam
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
