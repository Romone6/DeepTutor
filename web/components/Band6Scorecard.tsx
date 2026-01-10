"use client";

import { useState } from "react";
import {
  Award,
  CheckCircle,
  XCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  BookOpen,
  Target,
  TrendingUp,
  Lightbulb,
} from "lucide-react";

interface RubricCriterionResult {
  id: string;
  description: string;
  score: number;
  passed: boolean;
  feedback: string;
}

interface RubricDimensionResult {
  dimension: string;
  score: number;
  passed: boolean;
  criteria: RubricCriterionResult[];
}

interface Band6ScorecardProps {
  bandEstimate: string;
  overallScore: number;
  rubricChecklist: RubricDimensionResult[];
  topFixes: Array<{
    dimension: string;
    criterion: string;
    issue: string;
    fix: string;
    priority: string;
  }>;
  improvementTips: string[];
  feedbackSummary: string;
  subject?: string;
  onDismiss?: () => void;
}

export default function Band6Scorecard({
  bandEstimate,
  overallScore,
  rubricChecklist,
  topFixes,
  improvementTips,
  feedbackSummary,
  subject,
  onDismiss,
}: Band6ScorecardProps) {
  const [expanded, setExpanded] = useState(true);

  const getBandColor = (band: string) => {
    if (band.includes("6")) return "text-purple-600 bg-purple-50 border-purple-200";
    if (band.includes("5")) return "text-green-600 bg-green-50 border-green-200";
    if (band.includes("4")) return "text-blue-600 bg-blue-50 border-blue-200";
    if (band.includes("3")) return "text-yellow-600 bg-yellow-50 border-yellow-200";
    return "text-gray-600 bg-gray-50 border-gray-200";
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return "text-purple-600";
    if (score >= 80) return "text-green-600";
    if (score >= 70) return "text-blue-600";
    if (score >= 60) return "text-yellow-600";
    return "text-red-600";
  };

  const getProgressColor = (score: number) => {
    if (score >= 90) return "bg-purple-500";
    if (score >= 80) return "bg-green-500";
    if (score >= 70) return "bg-blue-500";
    if (score >= 60) return "bg-yellow-500";
    return "bg-red-500";
  };

  const passedCount = rubricChecklist.filter((d) => d.passed).length;
  const failedCount = rubricChecklist.length - passedCount;

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Award className="w-5 h-5 text-purple-600" />
          <span className="font-semibold text-gray-900">Band 6 Scorecard</span>
          {subject && (
            <span className="text-sm text-gray-500 px-2 py-0.5 bg-gray-100 rounded-full">
              {subject.replace("_", " ")}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">Passed:</span>
            <span className="text-sm font-medium text-green-600">{passedCount}</span>
            <span className="text-sm text-gray-300">/</span>
            <span className="text-sm font-medium text-gray-500">{rubricChecklist.length}</span>
          </div>
          <button
            onClick={() => setExpanded(!expanded)}
            className="p-1 hover:bg-gray-200 rounded transition-colors"
          >
            {expanded ? (
              <ChevronUp className="w-4 h-4 text-gray-500" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-500" />
            )}
          </button>
          {onDismiss && (
            <button
              onClick={onDismiss}
              className="p-1 hover:bg-gray-200 rounded transition-colors"
            >
              <XCircle className="w-4 h-4 text-gray-400" />
            </button>
          )}
        </div>
      </div>

      {expanded && (
        <div className="p-4">
          {/* Band Estimate and Overall Score */}
          <div className="flex items-center gap-4 mb-4">
            <div
              className={`px-4 py-2 rounded-lg border-2 ${getBandColor(bandEstimate)}`}
            >
              <span className="text-2xl font-bold">{bandEstimate}</span>
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm text-gray-600">Overall Score</span>
                <span className={`text-xl font-bold ${getScoreColor(overallScore)}`}>
                  {overallScore.toFixed(1)}%
                </span>
              </div>
              <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full ${getProgressColor(overallScore)} transition-all duration-500`}
                  style={{ width: `${overallScore}%` }}
                />
              </div>
            </div>
          </div>

          {/* Feedback Summary */}
          <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-100">
            <div className="flex items-start gap-2">
              <BookOpen className="w-4 h-4 text-blue-600 mt-0.5" />
              <p className="text-sm text-blue-800">{feedbackSummary}</p>
            </div>
          </div>

          {/* Rubric Checklist */}
          <div className="space-y-3 mb-4">
            <h4 className="text-sm font-medium text-gray-700 flex items-center gap-2">
              <Target className="w-4 h-4" />
              Rubric Checklist
            </h4>
            {rubricChecklist.map((dimension) => (
              <div
                key={dimension.dimension}
                className="border border-gray-200 rounded-lg overflow-hidden"
              >
                <div
                  className={`px-3 py-2 flex items-center justify-between ${
                    dimension.passed ? "bg-green-50" : "bg-red-50"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {dimension.passed ? (
                      <CheckCircle className="w-4 h-4 text-green-600" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-600" />
                    )}
                    <span className="text-sm font-medium text-gray-800">
                      {dimension.dimension}
                    </span>
                  </div>
                  <span className={`text-sm font-bold ${getScoreColor(dimension.score)}`}>
                    {dimension.score.toFixed(0)}%
                  </span>
                </div>
                <div className="p-3 bg-white">
                  <div className="grid gap-2">
                    {dimension.criteria.map((criterion) => (
                      <div
                        key={criterion.id}
                        className="flex items-start gap-2 text-sm"
                      >
                        {criterion.passed ? (
                          <CheckCircle className="w-3.5 h-3.5 text-green-500 mt-0.5" />
                        ) : (
                          <AlertTriangle className="w-3.5 h-3.5 text-yellow-500 mt-0.5" />
                        )}
                        <span className="text-gray-700">{criterion.description}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Top Fixes */}
          {topFixes.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-medium text-gray-700 flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4" />
                Top Priority Fixes
              </h4>
              <div className="space-y-2">
                {topFixes.map((fix, index) => (
                  <div
                    key={index}
                    className={`p-3 rounded-lg border ${
                      fix.priority === "high"
                        ? "bg-red-50 border-red-100"
                        : "bg-yellow-50 border-yellow-100"
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      <span
                        className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
                          fix.priority === "high"
                            ? "bg-red-100 text-red-700"
                            : "bg-yellow-100 text-yellow-700"
                        }`}
                      >
                        {index + 1}
                      </span>
                      <div className="flex-1">
                        <p className="text-sm font-medium text-gray-800">{fix.criterion}</p>
                        <p className="text-sm text-gray-600 mt-0.5">{fix.issue}</p>
                        <p className="text-sm text-blue-700 mt-1 flex items-center gap-1">
                          <Lightbulb className="w-3.5 h-3.5" />
                          {fix.fix}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Improvement Tips */}
          {improvementTips.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 flex items-center gap-2 mb-2">
                <Lightbulb className="w-4 h-4" />
                Improvement Tips
              </h4>
              <ul className="space-y-1">
                {improvementTips.slice(0, 4).map((tip, index) => (
                  <li key={index} className="text-sm text-gray-600 flex items-start gap-2">
                    <span className="text-blue-500 mt-1">•</span>
                    {tip}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
