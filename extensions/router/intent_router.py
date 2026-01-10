"""
Intent Router for DeepTutor

Determines user intent and routes messages to appropriate handlers.

Features:
- Hard keyword-based classification (fast, deterministic)
- Subject detection with keyword mapping
- Confidence scoring (high/medium/low)
- Clarification questions for low confidence
- Optional LLM fallback for medium confidence
- Strict JSON output schema

Subjects:
- solve: Problem solving, math, coding, exercises
- question: Quizzes, assessments, exam prep
- research: Information gathering, web search
- knowledge: Knowledge base queries, RAG
- ideagen: Creative ideas, brainstorming
- chat: General conversation
"""

import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from extensions.utils.schemas import RouteDecision
except ImportError:
    # Fallback if schemas not available
    class RouteDecision(BaseModel):
        target: str = "router"
        handler: str = ""
        confidence: float = 0.0
        metadata: dict = field(default_factory=dict)


# =============================================================================
# Configuration
# =============================================================================


class ConfidenceLevel(str, Enum):
    """Confidence level enumeration."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Subject(str, Enum):
    """Subject enumeration matching DeepTutor routes."""

    SOLVE = "solve"
    QUESTION = "question"
    RESEARCH = "research"
    KNOWLEDGE = "knowledge"
    IDEAGEN = "ideagen"
    CHAT = "chat"


# =============================================================================
# Keyword Rules
# =============================================================================


# Subject keyword mappings with weights
SUBJECT_KEYWORDS: dict[Subject, list[tuple[str, float]]] = {
    Subject.SOLVE: [
        # Math keywords
        (r"\b(solve|calculate|compute|evaluate|simplify)\b", 1.0),
        (r"\b(equation|formula|integral|derivative|function)\b", 0.9),
        (r"\b(math|mathematics|algebra|geometry|calculus)\b", 0.8),
        (r"\b(problem|exercise|practice|step by step)\b", 0.7),
        (r"\b(x|y|z|variable|unknown)\b", 0.6),
        # Coding keywords
        (r"\b(code|program|debug|function|algorithm)\b", 0.9),
        (r"\b(python|javascript|java|c\+\+|sql)\b", 0.8),
        (r"\b(bug|error|exception|stack trace)\b", 0.7),
        (r"\b(api|endpoint|request|response)\b", 0.6),
    ],
    Subject.QUESTION: [
        (r"\b(quiz|test|exam|assessment)\b", 1.0),
        (r"\b(question|multiple choice|true false)\b", 0.9),
        (r"\b(what is|who is|where is|when did)\b", 0.6),
        (r"\b(explain the difference|compare)\b", 0.7),
        (r"\b(practice|drill|exercise)\b", 0.8),
        (r"\b(check my answer|verify)\b", 0.9),
    ],
    Subject.RESEARCH: [
        (r"\b(research|study|survey|investigate)\b", 1.0),
        (r"\b(find information|look up|search for)\b", 0.9),
        (r"\b(web|internet|online)\b", 0.7),
        (r"\b(trends|statistics|data|analysis)\b", 0.8),
        (r"\b(latest news|recent|current)\b", 0.7),
    ],
    Subject.KNOWLEDGE: [
        (r"\b(knowledge base|kb|document|source)\b", 1.0),
        (r"\b(explain|describe|what is|how does)\b", 0.6),
        (r"\b(context|reference|according to)\b", 0.7),
        (r"\b(textbook|manual|guide|documentation)\b", 0.8),
        (r"\b(based on|from the|from your)\b", 0.5),
    ],
    Subject.IDEAGEN: [
        (r"\b(idea|creative|innovative|brainstorm)\b", 1.0),
        (r"\b(generate|create|come up with|think of)\b", 0.9),
        (r"\b(suggestion|recommendation|tip|advice)\b", 0.7),
        (r"\b(alternative|option|possibility)\b", 0.6),
        (r"\b(what if|imagine| envision)\b", 0.7),
    ],
    Subject.CHAT: [
        (r"\b(hi|hello|hey|good morning|good afternoon)\b", 0.9),
        (r"\b(thanks|thank you|appreciate)\b", 0.7),
        (r"\b(bye|goodbye|see you)\b", 0.8),
        (r"\b(how are you|what's up)\b", 0.8),
        (r"\b(just|really|actually|honestly)\b", 0.3),
    ],
}

# Negative keywords that reduce confidence
NEGATIVE_KEYWORDS: list[tuple[re.Pattern, float]] = [
    (r"\b(not |don't |doesn't |didn't |can't |won't |wouldn't |shouldn't |couldn't )", -0.3),
    (r"\b(i don't know|i'm not sure|i have no idea)", -0.5),
    (r"\b(maybe|perhaps|possibly|i guess)", -0.2),
]

# Intent patterns
INTENT_PATTERNS: dict[str, list[re.Pattern]] = {
    "question": [
        r"\?$",
        r"\b(what|who|where|when|why|how|which)\b",
        r"\b(is there|are there|do you know)\b",
    ],
    "request": [
        r"\b(please|could you|would you|can you)\b",
        r"\b(i need|i want|i'd like)\b",
    ],
    "statement": [
        r"\b(i think|i believe|in my opinion)\b",
        r".*\.$",  # Ends with period
    ],
}

# Clarification templates by subject
CLARIFICATION_TEMPLATES: dict[Subject, str] = {
    Subject.SOLVE: "What specific problem would you like me to help you solve? Please provide the full question or equation.",
    Subject.QUESTION: "What type of question is this? Is it multiple choice, true/false, or open-ended?",
    Subject.RESEARCH: "What specific information are you looking for? Any particular topic or keywords?",
    Subject.KNOWLEDGE: "Which knowledge base or document should I refer to? Or what topic would you like explained?",
    Subject.IDEAGEN: "What challenge or topic would you like creative ideas for?",
    Subject.CHAT: "How can I help you today?",
}


# =============================================================================
# Pydantic Models
# =============================================================================


class IntentRouterConfig(BaseModel):
    """Configuration for intent router."""

    enable_llm_fallback: bool = True
    llm_threshold: float = 0.6  # Use LLM when confidence >= this
    high_threshold: float = 0.8  # High confidence threshold
    medium_threshold: float = 0.5  # Medium confidence threshold
    default_subject: Subject = Subject.CHAT
    max_clarification_length: int = 200


class IntentClassification(BaseModel):
    """Result of intent classification."""

    subject: Subject = Subject.CHAT
    intent: str = "statement"
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    matched_keywords: list[str] = Field(default_factory=list)
    matched_patterns: list[str] = Field(default_factory=list)
    reasoning: str = ""
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


class RouterResponse(BaseModel):
    """Final router response with strict schema."""

    decision: RouteDecision
    classification: IntentClassification
    original_message: str
    timestamp: str
    version: str = "1.0.0"

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "decision": {
                        "target": "router",
                        "handler": "solve",
                        "confidence": 0.95,
                        "metadata": {},
                    },
                    "classification": {
                        "subject": "solve",
                        "intent": "question",
                        "confidence": 0.95,
                        "confidence_level": "high",
                        "matched_keywords": ["solve", "equation"],
                        "matched_patterns": ["?"],
                        "reasoning": "Math keywords detected",
                        "needs_clarification": False,
                    },
                    "original_message": "Solve the equation 2x + 5 = 15",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "version": "1.0.0",
                }
            ]
        }


# =============================================================================
# Intent Router Class
# =============================================================================


class IntentRouter:
    """
    Intent classification router for DeepTutor.

    Uses deterministic keyword matching with optional LLM fallback
    for medium-confidence classifications.
    """

    def __init__(self, config: Optional[IntentRouterConfig] = None):
        """
        Initialize the intent router.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or IntentRouterConfig()
        self._compile_regexes()

    def _compile_regexes(self):
        """Compile all regex patterns for efficiency."""
        self._subject_keywords: dict[Subject, list[tuple[re.Pattern, float]]] = {}
        for subject, keywords in SUBJECT_KEYWORDS.items():
            self._subject_keywords[subject] = [
                (re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in keywords
            ]

        self._negative_keywords: list[tuple[re.Pattern, float]] = [
            (re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in NEGATIVE_KEYWORDS
        ]

        self._intent_patterns: dict[str, list[re.Pattern]] = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in INTENT_PATTERNS.items()
        }

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        return text.strip().lower()

    def _detect_intent(self, text: str) -> str:
        """Detect the intent type from text."""
        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return intent
        return "statement"

    def _calculate_subject_scores(self, text: str) -> dict[Subject, float]:
        """Calculate subject scores based on keyword matching."""
        scores: dict[Subject, float] = {s: 0.0 for s in Subject}
        text_lower = text.lower()

        for subject, keywords in self._subject_keywords.items():
            matched = []
            for pattern, base_weight in keywords:
                if pattern.search(text_lower):
                    scores[subject] += base_weight
                    matched.append(pattern.pattern)
            # Bonus for multiple matches in same subject
            if len(matched) >= 2:
                scores[subject] *= 1.1
            elif len(matched) >= 3:
                scores[subject] *= 1.2

        # Apply negative keywords
        for pattern, penalty in self._negative_keywords:
            if pattern.search(text_lower):
                for subject in scores:
                    scores[subject] = max(0.0, scores[subject] + penalty)

        # Normalize scores to 0-1 range
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            for subject in scores:
                scores[subject] = min(1.0, scores[subject] / max_score)

        return scores

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map confidence score to level."""
        if confidence >= self.config.high_threshold:
            return ConfidenceLevel.HIGH
        elif confidence >= self.config.medium_threshold:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _determine_subject(self, scores: dict[Subject, float]) -> tuple[Subject, float]:
        """Determine the subject with highest score."""
        if not scores:
            return self.config.default_subject, 0.0

        # Find highest scoring subject
        best_subject = max(scores, key=lambda s: scores[s])
        best_score = scores[best_subject]

        # If all scores are very low, use default
        if best_score < 0.1:
            return self.config.default_subject, 0.3

        return best_subject, best_score

    def _generate_reasoning(
        self,
        subject: Subject,
        confidence: float,
        matched_keywords: list[str],
        matched_patterns: list[str],
    ) -> str:
        """Generate human-readable reasoning for the classification."""
        parts = []

        if matched_keywords:
            parts.append(f"Keywords: {', '.join(matched_keywords[:3])}")

        if matched_patterns:
            parts.append(f"Patterns: {', '.join(matched_patterns[:2])}")

        if confidence >= 0.8:
            parts.append("High confidence match")
        elif confidence >= 0.5:
            parts.append("Medium confidence match")
        else:
            parts.append("Low confidence - clarification recommended")

        return "; ".join(parts) if parts else "Default classification"

    def _should_clarify(self, classification: IntentClassification) -> tuple[bool, Optional[str]]:
        """
        Determine if clarification is needed.

        Returns:
            Tuple of (needs_clarification, clarification_question)
        """
        # Don't clarify for high confidence
        if classification.confidence_level == ConfidenceLevel.HIGH:
            return False, None

        # Don't clarify for chat/small messages
        if len(classification.original_message or "") < 20:
            return False, None

        # Generate clarification question based on subject
        template = CLARIFICATION_TEMPLATES.get(
            classification.subject, "Could you please clarify what you'd like help with?"
        )

        # Check if message is too vague
        if len(classification.original_message or "") < 10:
            return True, template

        # Medium confidence with vague message
        if classification.confidence_level == ConfidenceLevel.LOW:
            return True, template

        return False, None

    def _create_route_decision(
        self,
        subject: Subject,
        confidence: float,
    ) -> RouteDecision:
        """Create a RouteDecision from classification."""
        return RouteDecision(
            target="router",
            handler=subject.value,
            confidence=confidence,
            metadata={
                "subject": subject.value,
                "router_version": "1.0.0",
            },
        )

    def classify(self, message: str) -> IntentClassification:
        """
        Classify a message into intent and subject.

        Args:
            message: The user message to classify

        Returns:
            IntentClassification with subject, intent, confidence, etc.
        """
        if not message or not message.strip():
            return IntentClassification(
                subject=self.config.default_subject,
                intent="statement",
                confidence=0.0,
                confidence_level=ConfidenceLevel.LOW,
                reasoning="Empty message",
                needs_clarification=True,
                clarification_question=CLARIFICATION_TEMPLATES[self.config.default_subject],
            )

        # Normalize message
        text = self._normalize_text(message)

        # Detect intent
        intent = self._detect_intent(text)

        # Calculate subject scores
        scores = self._calculate_subject_scores(text)

        # Determine subject and confidence
        subject, base_confidence = self._determine_subject(scores)

        # Collect matched keywords
        matched_keywords = []
        for pattern, _ in self._subject_keywords.get(subject, []):
            if pattern.search(text):
                # Extract matched keyword (simplified)
                match = pattern.search(text)
                if match:
                    matched_keywords.append(match.group(0))

        # Get matched intent patterns
        matched_patterns = []
        for ipattern in self._intent_patterns.get(intent, []):
            if ipattern.search(text):
                matched_patterns.append(ipattern.pattern)

        # Adjust confidence based on matched keywords count
        keyword_boost = min(0.2, len(matched_keywords) * 0.05)
        confidence = min(1.0, base_confidence + keyword_boost)

        confidence_level = self._get_confidence_level(confidence)

        reasoning = self._generate_reasoning(
            subject, confidence, matched_keywords, matched_patterns
        )

        classification = IntentClassification(
            subject=subject,
            intent=intent,
            confidence=confidence,
            confidence_level=confidence_level,
            matched_keywords=matched_keywords[:5],  # Limit to top 5
            matched_patterns=matched_patterns[:3],  # Limit to top 3
            reasoning=reasoning,
            needs_clarification=False,  # Set in should_clarify
            clarification_question=None,
        )

        # Check if clarification needed
        needs_clarify, question = self._should_clarify(classification)
        classification.needs_clarification = needs_clarify
        classification.clarification_question = question

        return classification

    def route(self, message: str) -> RouterResponse:
        """
        Route a message to the appropriate handler.

        Args:
            message: The user message to route

        Returns:
            RouterResponse with decision and classification
        """
        classification = self.classify(message)

        # Check if LLM fallback should be used
        use_llm_fallback = (
            self.config.enable_llm_fallback
            and classification.confidence_level == ConfidenceLevel.MEDIUM
        )

        if use_llm_fallback:
            # Use LLM to improve classification
            llm_classification = self._llm_fallback_classify(message)
            if llm_classification:
                # Use LLM classification if higher confidence
                if llm_classification.confidence > classification.confidence:
                    classification = llm_classification

        # Create route decision
        decision = self._create_route_decision(
            classification.subject,
            classification.confidence,
        )

        return RouterResponse(
            decision=decision,
            classification=classification,
            original_message=message,
            timestamp=self._get_timestamp(),
        )

    def _llm_fallback_classify(self, message: str) -> Optional[IntentClassification]:
        """
        Use LLM to improve medium-confidence classifications.

        This is a placeholder - implement with actual LLM call if needed.
        """
        # TODO: Implement LLM-based classification for medium confidence cases
        # For now, return None to use deterministic classification
        return None

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"

    def batch_classify(self, messages: list[str]) -> list[IntentClassification]:
        """
        Classify multiple messages.

        Args:
            messages: List of messages to classify

        Returns:
            List of IntentClassification results
        """
        return [self.classify(msg) for msg in messages]

    def get_subjects(self) -> list[str]:
        """Get list of supported subjects."""
        return [s.value for s in Subject]

    def get_statistics(self, classifications: list[IntentClassification]) -> dict:
        """Get statistics about classifications."""
        if not classifications:
            return {}

        total = len(classifications)
        high_conf = sum(1 for c in classifications if c.confidence_level == ConfidenceLevel.HIGH)
        medium_conf = sum(
            1 for c in classifications if c.confidence_level == ConfidenceLevel.MEDIUM
        )
        low_conf = sum(1 for c in classifications if c.confidence_level == ConfidenceLevel.LOW)

        subject_counts: dict[str, int] = {}
        for c in classifications:
            subject_counts[c.subject.value] = subject_counts.get(c.subject.value, 0) + 1

        return {
            "total": total,
            "high_confidence_count": high_conf,
            "medium_confidence_count": medium_conf,
            "low_confidence_count": low_conf,
            "high_confidence_percent": round(high_conf / total * 100, 1) if total > 0 else 0,
            "subject_distribution": subject_counts,
            "average_confidence": round(sum(c.confidence for c in classifications) / total, 3)
            if total > 0
            else 0,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_intent_router(
    enable_llm_fallback: bool = True,
    high_threshold: float = 0.8,
    medium_threshold: float = 0.5,
) -> IntentRouter:
    """
    Create an intent router with specified configuration.

    Args:
        enable_llm_fallback: Enable LLM fallback for medium confidence
        high_threshold: Threshold for high confidence (0.0-1.0)
        medium_threshold: Threshold for medium confidence (0.0-1.0)

    Returns:
        Configured IntentRouter instance
    """
    config = IntentRouterConfig(
        enable_llm_fallback=enable_llm_fallback,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
    )
    return IntentRouter(config)


def classify_message(message: str) -> IntentClassification:
    """
    Convenience function to classify a single message.

    Args:
        message: Message to classify

    Returns:
        IntentClassification result
    """
    router = create_intent_router()
    return router.classify(message)


def route_message(message: str) -> RouterResponse:
    """
    Convenience function to route a single message.

    Args:
        message: Message to route

    Returns:
        RouterResponse with decision
    """
    router = create_intent_router()
    return router.route(message)
