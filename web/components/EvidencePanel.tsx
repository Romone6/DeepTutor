"use client";

import { useState } from "react";
import {
  BookOpen,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  FileText,
  GraduationCap,
  Scale,
} from "lucide-react";

export interface Citation {
  key: string;
  label: string;
  doc_id: string;
  page?: number;
  topic: string;
  preview: string;
  full_text: string;
  score: number;
  source_type: string;
}

export interface EvidenceMapData {
  query: string;
  kb_name: string;
  total_retrieved: number;
  used_for_answer: number;
  kb_empty: boolean;
  timestamp: string;
  snippets: Citation[];
}

interface EvidencePanelProps {
  evidence_map: EvidenceMapData | null;
  showByDefault?: boolean;
  compact?: boolean;
}

const sourceIcons = {
  syllabus: GraduationCap,
  exam_paper: FileText,
  marking_guide: Scale,
  textbook: BookOpen,
  notes: FileText,
  unknown: FileText,
};

const sourceColors = {
  syllabus: "bg-blue-100 text-blue-800 border-blue-200",
  exam_paper: "bg-purple-100 text-purple-800 border-purple-200",
  marking_guide: "bg-amber-100 text-amber-800 border-amber-200",
  textbook: "bg-green-100 text-green-800 border-green-200",
  notes: "bg-gray-100 text-gray-800 border-gray-200",
  unknown: "bg-gray-100 text-gray-800 border-gray-200",
};

export default function EvidencePanel({
  evidence_map,
  showByDefault = false,
  compact = false,
}: EvidencePanelProps) {
  const [isExpanded, setIsExpanded] = useState(showByDefault);
  const [selectedCitation, setSelectedCitation] = useState<string | null>(null);

  if (!evidence_map || evidence_map.kb_empty) {
    return null;
  }

  const { snippets = [] } = evidence_map;

  if (snippets.length === 0) {
    return null;
  }

  const toggleExpand = () => setIsExpanded(!isExpanded);

  const handleCitationClick = (key: string) => {
    setSelectedCitation(selectedCitation === key ? null : key);
  };

  if (compact) {
    return (
      <div className="mt-2">
        <button
          onClick={toggleExpand}
          className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
        >
          <BookOpen className="w-4 h-4" />
          <span>
            {isExpanded ? "Hide" : "Show"} evidence ({snippets.length} sources)
          </span>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </button>

        {isExpanded && (
          <div className="mt-2 p-3 bg-gray-50 rounded-lg border border-gray-200">
            <CitationList
              citations={snippets}
              selectedKey={selectedCitation}
              onSelect={handleCitationClick}
            />
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="mt-4 border border-gray-200 rounded-lg overflow-hidden">
      <button
        onClick={toggleExpand}
        className="w-full flex items-center justify-between p-3 bg-gray-50 hover:bg-gray-100 transition-colors"
      >
        <div className="flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-gray-700" />
          <span className="font-medium text-gray-900">
            Evidence & Sources
          </span>
          <span className="text-sm text-gray-500">
            ({snippets.length} {snippets.length === 1 ? "source" : "sources"})
          </span>
        </div>
        <div className="flex items-center gap-2">
          {evidence_map.kb_name && (
            <span className="text-xs text-gray-500 bg-gray-200 px-2 py-1 rounded">
              {evidence_map.kb_name}
            </span>
          )}
          {isExpanded ? (
            <ChevronUp className="w-5 h-5 text-gray-500" />
          ) : (
            <ChevronDown className="w-5 h-5 text-gray-500" />
          )}
        </div>
      </button>

      {isExpanded && (
        <div className="p-4 bg-white">
          <div className="space-y-3">
            <CitationList
              citations={snippets}
              selectedKey={selectedCitation}
              onSelect={handleCitationClick}
            />
          </div>

          <div className="mt-4 pt-3 border-t border-gray-100 text-xs text-gray-500">
            <p>
              Generated at:{" "}
              {evidence_map.timestamp
                ? new Date(evidence_map.timestamp).toLocaleString()
                : "N/A"}
            </p>
            <p className="mt-1">
              Retrieved {evidence_map.total_retrieved} snippets, using{" "}
              {evidence_map.used_for_answer} for this answer.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function CitationList({
  citations,
  selectedKey,
  onSelect,
}: {
  citations: Citation[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
}) {
  return (
    <div className="space-y-2">
      {citations.map((citation) => {
        const SourceIcon = sourceIcons[
          citation.source_type as keyof typeof sourceIcons
        ] || FileText;
        const colorClass =
          sourceColors[
            citation.source_type as keyof typeof sourceColors
          ] || sourceColors.unknown;

        const isSelected = selectedKey === citation.key;

        return (
          <div
            key={citation.key}
            className={`rounded-lg border transition-all ${
              isSelected
                ? "border-blue-300 bg-blue-50"
                : "border-gray-200 hover:border-gray-300"
            }`}
          >
            <button
              onClick={() => onSelect(citation.key)}
              className="w-full p-3 text-left"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span
                    className={`flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${colorClass}`}
                  >
                    <SourceIcon className="w-3 h-3" />
                    {citation.key}
                  </span>
                  <span className="font-medium text-gray-900">
                    {citation.label}
                  </span>
                  {citation.doc_id && citation.doc_id !== "unknown" && (
                    <span className="text-xs text-gray-500">
                      ({citation.doc_id}
                      {citation.page && `, p.${citation.page}`})
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-1">
                  {citation.score > 0.8 && (
                    <span className="text-xs px-1.5 py-0.5 bg-green-100 text-green-700 rounded">
                      High relevance
                    </span>
                  )}
                </div>
              </div>

              {citation.topic && (
                <p className="mt-1 text-xs text-gray-500">
                  Topic: {citation.topic}
                </p>
              )}

              <p className="mt-2 text-sm text-gray-700 line-clamp-2">
                {citation.preview}
              </p>
            </button>

            {isSelected && (
              <div className="px-3 pb-3">
                <div className="pt-2 border-t border-gray-200">
                  <p className="text-sm text-gray-800 whitespace-pre-wrap">
                    {citation.full_text}
                  </p>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

interface EvidenceToggleProps {
  showEvidence: boolean;
  onToggle: (show: boolean) => void;
  count?: number;
}

export function EvidenceToggle({
  showEvidence,
  onToggle,
  count,
}: EvidenceToggleProps) {
  return (
    <button
      onClick={() => onToggle(!showEvidence)}
      className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
        showEvidence
          ? "bg-blue-100 text-blue-800 hover:bg-blue-200"
          : "bg-gray-100 text-gray-700 hover:bg-gray-200"
      }`}
    >
      <BookOpen className="w-4 h-4" />
      {showEvidence ? "Hide" : "Show"} Evidence
      {count !== undefined && count > 0 && (
        <span className="ml-1 px-1.5 py-0.5 bg-white/50 rounded text-xs">
          {count}
        </span>
      )}
    </button>
  );
}

interface EvidenceBadgeProps {
  citations: string[];
}

export function EvidenceBadge({ citations }: EvidenceBadgeProps) {
  if (!citations || citations.length === 0) {
    return null;
  }

  return (
    <div className="flex items-center gap-1 flex-wrap">
      <span className="text-xs text-gray-500 mr-1">Sources:</span>
      {citations.map((citation) => (
        <span
          key={citation}
          className="inline-flex items-center px-1.5 py-0.5 bg-gray-100 text-gray-700 text-xs rounded"
        >
          {citation}
        </span>
      ))}
    </div>
  );
}
