"""
Policy Enforcement Decorators and Helpers

Provides decorators and utilities for enforcing policy settings
in DeepTutor modules without rewriting core logic.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from .engine import PolicyEngine, PolicySession, get_policy_engine, PolicyConfig


def enforce_hinting(max_hints_param: str = "max_hints") -> Callable:
    """
    Decorator to enforce hinting policy on a function.

    Usage:
        @enforce_hinting()
        async def provide_hint(self, context, hint_level=1):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            self_arg = args[0] if args else None
            policy_id = getattr(self_arg, "policy_id", None) or getattr(
                self_arg, "current_policy_id", "term"
            )

            engine = get_policy_engine()
            policy = engine.get_policy(policy_id)

            if policy:
                hinting = policy.hinting
                kwargs[max_hints_param] = hinting.max_hints

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            self_arg = args[0] if args else None
            policy_id = getattr(self_arg, "policy_id", None) or getattr(
                self_arg, "current_policy_id", "term"
            )

            engine = get_policy_engine()
            policy = engine.get_policy(policy_id)

            if policy:
                hinting = policy.hinting
                kwargs[max_hints_param] = hinting.max_hints

            return func(*args, **kwargs)

        return async_wrapper if hasattr(func, "__wrapped__") else sync_wrapper

    return decorator


def enforce_strictness(score_param: str = "score") -> Callable:
    """
    Decorator to evaluate answer against policy strictness.

    Usage:
        @enforce_strictness()
        def evaluate_answer(self, answer, expected):
            score = calculate_score(answer, expected)
            return score
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self_arg = args[0] if args else None
            policy_id = getattr(self_arg, "policy_id", None) or getattr(
                self_arg, "current_policy_id", "term"
            )

            engine = get_policy_engine()
            policy = engine.get_policy(policy_id)

            if policy:
                strictness = policy.strictness
                kwargs["pass_threshold"] = strictness.pass_threshold
                kwargs["evaluation_strictness"] = strictness.evaluation_strictness

            return func(*args, **kwargs)

        return wrapper

    return decorator


def apply_policy_prompt(original_prompt_param: str = "system_prompt") -> Callable:
    """
    Decorator to apply policy system prompt suffix.

    Usage:
        @apply_policy_prompt()
        async def call_llm(self, system_prompt, user_prompt):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            self_arg = args[0] if args else None
            policy_id = getattr(self_arg, "policy_id", None) or getattr(
                self_arg, "current_policy_id", "term"
            )

            if original_prompt_param in kwargs:
                base_prompt = kwargs[original_prompt_param]
                engine = get_policy_engine()
                kwargs[original_prompt_param] = engine.apply_policy_to_prompt(
                    policy_id, base_prompt
                )

            return await func(*args, **kwargs)

        return async_wrapper

    return decorator


def get_policy_params() -> Callable:
    """
    Decorator to inject policy parameters into a method.

    Usage:
        @get_policy_params()
        def configure_solver(self, policy, temperature, max_tokens):
            solver.temperature = temperature
            solver.max_tokens = max_tokens
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self_arg = args[0] if args else None
            policy_id = getattr(self_arg, "policy_id", None) or getattr(
                self_arg, "current_policy_id", "term"
            )

            engine = get_policy_engine()
            policy = engine.get_policy(policy_id)

            if policy:
                params = policy.get_llm_params()
                kwargs["policy_params"] = params
                kwargs["policy"] = policy

            return func(*args, **kwargs)

        return wrapper

    return decorator


class PolicyAwareAgent:
    """
    Mixin class for agents that need policy awareness.

    Usage:
        class MyAgent(PolicyAwareAgent, BaseAgent):
            def __init__(self, *args, policy_id="term", **kwargs):
                super().__init__(*args, **kwargs)
                self.policy_id = policy_id

            async def answer_question(self, question):
                if self.should_use_active_recall():
                    question = self.add_recall_prompt(question)
                ...
    """

    policy_id: str = "term"

    def get_policy(self) -> PolicyConfig | None:
        """Get the current policy."""
        return get_policy_engine().get_policy(self.policy_id)

    def get_session(self) -> PolicySession | None:
        """Get the current policy session."""
        session_id = getattr(self, "session_id", None)
        if session_id:
            return get_policy_engine().get_session(session_id)
        return None

    def apply_policy_to_prompt(self, base_prompt: str) -> str:
        """Apply policy to a system prompt."""
        return get_policy_engine().apply_policy_to_prompt(self.policy_id, base_prompt)

    def get_policy_llm_params(self) -> dict[str, Any]:
        """Get LLM parameters for current policy."""
        return get_policy_engine().get_llm_params_for_policy(self.policy_id)

    def should_use_active_recall(self) -> bool:
        """Check if active recall should be used."""
        policy = self.get_policy()
        if policy:
            return policy.active_recall.recall_frequency.value in ("often", "always")
        return False

    def should_start_with_recall(self) -> bool:
        """Check if session should start with recall."""
        policy = self.get_policy()
        if policy:
            return policy.active_recall.start_with_recall
        return False

    def is_exploration_allowed(self) -> bool:
        """Check if off-topic exploration is allowed."""
        policy = self.get_policy()
        if policy:
            return policy.engagement.allow_exploration
        return False

    def get_evaluation_strictness(self) -> float:
        """Get the evaluation strictness score."""
        policy = self.get_policy()
        if policy:
            return policy.strictness.evaluation_strictness
        return 0.6

    def get_pass_threshold(self) -> float:
        """Get the pass threshold score."""
        policy = self.get_policy()
        if policy:
            return policy.strictness.pass_threshold
        return 0.7

    def get_max_hints(self) -> int:
        """Get the maximum number of hints."""
        policy = self.get_policy()
        if policy:
            return policy.hinting.max_hints
        return 3

    def get_check_frequency(self) -> str:
        """Get the checkpoint frequency."""
        policy = self.get_policy()
        if policy:
            return policy.checkpoints.check_frequency.value
        return "regular"

    def add_recall_prompt(self, question: str) -> str:
        """Add a recall prompt to a question."""
        return f"Recall: {question}"


class PolicyContextManager:
    """Context manager for temporary policy changes."""

    def __init__(
        self,
        original_policy_id: str,
        temp_policy_id: str | None = None,
        subject: str | None = None,
    ):
        self.original_policy_id = original_policy_id
        self.temp_policy_id = temp_policy_id
        self.subject = subject
        self.original_session = None
        self.temp_session = None

    def __enter__(self) -> PolicySession:
        engine = get_policy_engine()

        if self.temp_policy_id:
            import uuid

            session_id = f"temp_{uuid.uuid4().hex[:8]}"
            self.temp_session = engine.create_session(
                self.temp_policy_id,
                session_id,
                subject=self.subject,
            )
            return self.temp_session

        self.original_session = engine.get_session(self.original_policy_id)
        return self.original_session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_session:
            get_policy_engine().end_session(self.temp_session.policy.policy_id)


def with_policy(policy_id: str) -> Callable:
    """
    Decorator to temporarily use a specific policy.

    Usage:
        @with_policy("exam")
        async def run_exam_simulation(self):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            self_arg = args[0] if args else None
            original_policy_id = getattr(self_arg, "policy_id", "term")

            if hasattr(self_arg, "policy_id"):
                self_arg.policy_id = policy_id

            try:
                return await func(*args, **kwargs)
            finally:
                if hasattr(self_arg, "policy_id"):
                    self_arg.policy_id = original_policy_id

        return async_wrapper

    return decorator
