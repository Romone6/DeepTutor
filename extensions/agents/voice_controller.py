"""
Voice Controller Agent for Sprint Mode
=======================================

Hands-free voice control for sprint sessions with strict action schemas.

Voice Macros:
- "start today" - Start today's sprint session
- "next checkpoint" - Move to checkpoint phase
- "previous" / "go back" - Go back to previous phase
- "repeat" / "say again" - Repeat current content
- "slower" - Slow down explanation (reduce TTS speed)
- "faster" - Speed up explanation (increase TTS speed)
- "harder" - Increase difficulty level
- "easier" - Decrease difficulty level
- "mark this" / "done" / "complete" - Mark current item as complete
- "skip this" / "skip" - Skip current item
- "pause" / "wait" - Pause the session
- "resume" / "continue" - Resume the session
- "stop" / "end" / "quit" - End the session
- "help" / "what can I say" - List available commands

Usage:
    from extensions.agents.voice_controller import VoiceControllerAgent, VoiceAction

    controller = VoiceControllerAgent()
    action = controller.process_command("start today")
    if action:
        controller.execute_action(action, session_context)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class SprintVoiceAction(str, Enum):
    """Strict action schema for sprint voice commands."""

    START_TODAY = "start_today"
    NEXT_PHASE = "next_phase"
    PREVIOUS_PHASE = "previous_phase"
    REPEAT = "repeat"
    SLOWER = "slower"
    FASTER = "faster"
    HARDER = "harder"
    EASIER = "easier"
    MARK_COMPLETE = "mark_complete"
    SKIP_ITEM = "skip_item"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class VoiceCommandMatch:
    """Result of matching a voice command."""

    action: SprintVoiceAction
    confidence: float  # 0.0 to 1.0
    matched_phrase: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "matched_phrase": self.matched_phrase,
            "parameters": self.parameters,
        }


@dataclass
class SprintSessionContext:
    """Context for executing sprint actions."""

    plan_id: Optional[str] = None
    day_number: Optional[int] = None
    current_phase: str = "overview"
    current_topic: Optional[str] = None
    current_activity: Optional[str] = None
    difficulty_level: str = "medium"
    tts_speed: float = 1.0
    is_paused: bool = False
    session_active: bool = False


class VoiceControllerAgent:
    """
    Voice controller for hands-free sprint session control.

    Features:
    - Strict command matching with fuzzy fallback
    - Action schema with parameters
    - Context-aware execution
    - Help system for discovering commands
    """

    def __init__(self):
        self._command_patterns: dict[SprintVoiceAction, list[tuple[re.Pattern, float]]] = {}
        self._action_handlers: dict[SprintVoiceAction, Callable] = {}
        self._setup_command_patterns()
        self._setup_default_handlers()

        self.context = SprintSessionContext()
        self.last_command_time: float = 0
        self.command_cooldown_seconds: float = 0.5

    def _setup_command_patterns(self) -> None:
        """Set up regex patterns for voice command matching."""

        patterns: dict[SprintVoiceAction, list[tuple[str, float]]] = {
            SprintVoiceAction.START_TODAY: [
                (r"\bstart\s*(today|now|session)?\b", 0.95),
                (r"\bbegin\s*(the\s*)?(today\s*)?(session|day)?\b", 0.85),
                (r"\bready\s*to\s*start\b", 0.75),
            ],
            SprintVoiceAction.NEXT_PHASE: [
                (r"\bnext\s*(checkpoint|phase|step|part)?\b", 0.95),
                (r"\bcontinue\b", 0.80),
                (r"\bgo\s*to\s*(the\s*)?next\b", 0.85),
                (r"\bproceed\b", 0.80),
            ],
            SprintVoiceAction.PREVIOUS_PHASE: [
                (r"\b(go\s*)?back(ward)?\b", 0.95),
                (r"\bprevious\b", 0.95),
                (r"\bgo\s*back\b", 0.90),
                (r"\blast\s*(one|part|step)?\b", 0.80),
            ],
            SprintVoiceAction.REPEAT: [
                (r"\brepeat\b", 0.95),
                (r"\bsay\s*again\b", 0.95),
                (r"\bagain\b", 0.85),
                (r"\bonce\s*more\b", 0.80),
                (r"\bone\s*more\s*time\b", 0.75),
            ],
            SprintVoiceAction.SLOWER: [
                (r"\bslower\b", 0.95),
                (r"\bslow\s*down\b", 0.95),
                (r"\btoo\s*fast\b", 0.85),
                (r"\bslow\s*the\s*pace\b", 0.80),
            ],
            SprintVoiceAction.FASTER: [
                (r"\bfaster\b", 0.95),
                (r"\bspeed\s*up\b", 0.95),
                (r"\btoo\s*slow\b", 0.85),
                (r"\bpick\s*up\s*the\s*pace\b", 0.75),
            ],
            SprintVoiceAction.HARDER: [
                (r"\bharder\b", 0.95),
                (r"\bmore\s*difficult\b", 0.90),
                (r"\bincrease\s*difficulty\b", 0.85),
                (r"\bchallenge\s*me\b", 0.75),
                (r"\btougher\b", 0.80),
            ],
            SprintVoiceAction.EASIER: [
                (r"\beasier\b", 0.95),
                (r"\bmore\s*(simple|basic|easier)\b", 0.90),
                (r"\bdecrease\s*difficulty\b", 0.85),
                (r"\bsimpler\b", 0.80),
                (r"\btoo\s*hard\b", 0.80),
            ],
            SprintVoiceAction.MARK_COMPLETE: [
                (r"\bmark\s*(this|it|done|complete)\b", 0.95),
                (r"\b(i\s*)?(got|gotta|have\s*)?done\b", 0.85),
                (r"\bcomplete(d)?\b", 0.80),
                (r"\bfinished\b", 0.85),
                (r"\bthat.?s\s*it\b", 0.75),
                (r"\b(i\s*)?finished\s*(with|this)?\b", 0.80),
            ],
            SprintVoiceAction.SKIP_ITEM: [
                (r"\bskip\b", 0.95),
                (r"\bskip\s*(this|that|it)\b", 0.95),
                (r"\bnext\s*(one|item)?\b", 0.80),
                (r"\bcome\s*back\s*later\b", 0.70),
            ],
            SprintVoiceAction.PAUSE: [
                (r"\bpause\b", 0.95),
                (r"\bwait\b", 0.85),
                (r"\bhold\s*on\b", 0.80),
                (r"\bstop\s*(for\s*a\s*sec(ond)?|moment|bit)?\b", 0.75),
            ],
            SprintVoiceAction.RESUME: [
                (r"\bresume\b", 0.95),
                (r"\bcontinue\b", 0.85),
                (r"\bgo\s*(back\s*to\s*)?work\b", 0.75),
                (r"\bkeep\s*going\b", 0.85),
                (r"\bkeep\s*on\b", 0.75),
                (r"\bstart\s*(again|over)?\b", 0.80),
            ],
            SprintVoiceAction.STOP: [
                (r"\bstop\b", 0.95),
                (r"\bend\b", 0.90),
                (r"\bquit\b", 0.90),
                (r"\bfinish\b", 0.80),
                (r"\bthat's?\s*all\b", 0.75),
                (r"\bdone\s*(for\s*)?(now|today)?\b", 0.80),
            ],
            SprintVoiceAction.HELP: [
                (r"\bhelp\b", 0.95),
                (r"\bwhat\s*(can\s*i|to)\s*say\b", 0.90),
                (r"\bcommands\b", 0.85),
                (r"\boptions\b", 0.80),
                (r"\bwhat\s*are\s*(my|the)\s*options\b", 0.85),
            ],
        }

        for action, pattern_list in patterns.items():
            self._command_patterns[action] = [
                (re.compile(p, re.IGNORECASE), confidence) for p, confidence in pattern_list
            ]

    def _setup_default_handlers(self) -> None:
        """Set up default action handlers."""

        def default_handler(action: SprintVoiceAction, context: SprintSessionContext) -> dict:
            return {
                "success": True,
                "action": action.value,
                "message": f"Action '{action.value}' executed",
                "context": context.__dict__,
            }

        for action in SprintVoiceAction:
            if action not in self._action_handlers:
                self._action_handlers[action] = lambda a, ctx, d=default_handler: d(a, ctx)

    def register_handler(
        self,
        action: SprintVoiceAction,
        handler: Callable[[SprintSessionContext], dict],
    ) -> None:
        """Register a custom handler for an action."""
        self._action_handlers[action] = handler

    def process_command(self, transcript: str) -> VoiceCommandMatch:
        """
        Process a voice transcript and return the matched command.

        Args:
            transcript: The transcribed voice text

        Returns:
            VoiceCommandMatch with action, confidence, and parameters
        """
        current_time = time.time()

        if current_time - self.last_command_time < self.command_cooldown_seconds:
            return VoiceCommandMatch(
                action=SprintVoiceAction.UNKNOWN,
                confidence=0.0,
                matched_phrase="",
                parameters={"reason": "cooldown"},
            )

        best_match: tuple[SprintVoiceAction, float, str] | None = None
        best_confidence = 0.0

        normalized = transcript.lower().strip()

        for action, pattern_list in self._command_patterns.items():
            for pattern, base_confidence in pattern_list:
                match = pattern.search(normalized)
                if match:
                    match_length = len(match.group())
                    length_bonus = min(match_length / len(normalized), 0.1)
                    confidence = min(base_confidence + length_bonus, 1.0)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = (
                            action,
                            confidence,
                            match.group(0),
                        )

        if best_match:
            self.last_command_time = current_time
            return VoiceCommandMatch(
                action=best_match[0],
                confidence=best_match[1],
                matched_phrase=best_match[2],
            )

        return VoiceCommandMatch(
            action=SprintVoiceAction.UNKNOWN,
            confidence=0.0,
            matched_phrase="",
            parameters={"original_transcript": transcript},
        )

    def execute_action(
        self,
        command: VoiceCommandMatch,
        context: Optional[SprintSessionContext] = None,
    ) -> dict:
        """
        Execute the action from a matched command.

        Args:
            command: The matched voice command
            context: Optional session context

        Returns:
            Result dictionary with success status and any output
        """
        action = command.action

        if action == SprintVoiceAction.UNKNOWN:
            return {
                "success": False,
                "action": "unknown",
                "message": "I didn't understand that command. Say 'help' for available commands.",
                "confidence": command.confidence,
                "original_transcript": command.parameters.get("original_transcript", ""),
            }

        if context is None:
            context = self.context

        if action == SprintVoiceAction.HELP:
            return self._get_help()

        if action == SprintVoiceAction.PAUSE:
            context.is_paused = True

        if action == SprintVoiceAction.RESUME:
            context.is_paused = False
            context.session_active = True

        if action == SprintVoiceAction.START_TODAY:
            context.session_active = True
            context.current_phase = "overview"
            context.is_paused = False

        if action == SprintVoiceAction.STOP:
            context.session_active = False
            context.is_paused = False

        if action == SprintVoiceAction.NEXT_PHASE:
            phases = ["overview", "checkpoint", "quiz", "complete"]
            if context.current_phase in phases:
                current_idx = phases.index(context.current_phase)
                if current_idx < len(phases) - 1:
                    context.current_phase = phases[current_idx + 1]

        if action == SprintVoiceAction.PREVIOUS_PHASE:
            phases = ["overview", "checkpoint", "quiz", "complete"]
            if context.current_phase in phases:
                current_idx = phases.index(context.current_phase)
                if current_idx > 0:
                    context.current_phase = phases[current_idx - 1]

        if action == SprintVoiceAction.HARDER:
            difficulty_levels = ["easy", "medium", "hard", "advanced"]
            if context.difficulty_level in difficulty_levels:
                current_idx = difficulty_levels.index(context.difficulty_level)
                if current_idx < len(difficulty_levels) - 1:
                    context.difficulty_level = difficulty_levels[current_idx + 1]

        if action == SprintVoiceAction.EASIER:
            difficulty_levels = ["easy", "medium", "hard", "advanced"]
            if context.difficulty_level in difficulty_levels:
                current_idx = difficulty_levels.index(context.difficulty_level)
                if current_idx > 0:
                    context.difficulty_level = difficulty_levels[current_idx - 1]

        if action == SprintVoiceAction.SLOWER:
            context.tts_speed = max(0.5, context.tts_speed - 0.1)

        if action == SprintVoiceAction.FASTER:
            context.tts_speed = min(2.0, context.tts_speed + 0.1)

        handler = self._action_handlers.get(
            action, self._action_handlers[SprintVoiceAction.UNKNOWN]
        )
        result = handler(context)

        result["command"] = command.to_dict()
        return result

    def _get_help(self) -> dict:
        """Get help text for available commands."""
        return {
            "success": True,
            "action": "help",
            "message": "Voice commands available:",
            "commands": {
                "start today": "Start today's sprint session",
                "next checkpoint": "Move to next phase",
                "previous": "Go back to previous phase",
                "repeat": "Repeat current content",
                "slower": "Slow down speech",
                "faster": "Speed up speech",
                "harder": "Increase difficulty",
                "easier": "Decrease difficulty",
                "mark this / done": "Mark item as complete",
                "skip": "Skip current item",
                "pause": "Pause the session",
                "resume": "Continue the session",
                "stop / end": "End the session",
                "help": "Show this help",
            },
            "context": self.context.__dict__,
        }

    def set_context(self, context: SprintSessionContext) -> None:
        """Set the current session context."""
        self.context = context

    def get_context(self) -> SprintSessionContext:
        """Get the current session context."""
        return self.context

    def update_phase(self, phase: str) -> None:
        """Update the current phase in context."""
        self.context.current_phase = phase

    def update_difficulty(self, difficulty: str) -> None:
        """Update the difficulty level in context."""
        self.context.difficulty_level = difficulty

    def update_tts_speed(self, speed: float) -> None:
        """Update the TTS speed in context."""
        self.context.tts_speed = max(0.5, min(2.0, speed))


def match_voice_command(transcript: str) -> VoiceCommandMatch:
    """
    Convenience function to match a voice command.

    Usage:
        match = match_voice_command("start today")
        if match.action == SprintVoiceAction.START_TODAY:
            # Execute action
    """
    controller = VoiceControllerAgent()
    return controller.process_command(transcript)


def execute_voice_action(
    transcript: str,
    context: Optional[SprintSessionContext] = None,
) -> dict:
    """
    Convenience function to process and execute a voice command.

    Usage:
        result = execute_voice_action("start today", context)
        if result["success"]:
            print(result["message"])
    """
    controller = VoiceControllerAgent()
    match = controller.process_command(transcript)
    return controller.execute_action(match, context)


__all__ = [
    "VoiceControllerAgent",
    "VoiceCommandMatch",
    "SprintSessionContext",
    "SprintVoiceAction",
    "match_voice_command",
    "execute_voice_action",
]
