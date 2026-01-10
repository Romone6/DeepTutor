from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ErrorTolerance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class HintGranularity(str, Enum):
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"
    COMPLETE = "complete"


class RecallFrequency(str, Enum):
    NEVER = "never"
    RARE = "rare"
    SOMETIMES = "sometimes"
    OFTEN = "often"
    ALWAYS = "always"


class ExplanationDepth(str, Enum):
    BRIEF = "brief"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"


class Tone(str, Enum):
    ENCOURAGING = "encouraging"
    NEUTRAL = "neutral"
    DEMANDING = "demanding"


class TimePressure(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CheckFrequency(str, Enum):
    NEVER = "never"
    SOMETIMES = "sometimes"
    REGULAR = "regular"
    FREQUENT = "frequent"


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    PLAIN = "plain"
    STRUCTURED = "structured"


class HintingConfig(BaseModel):
    max_hints: int = 3
    min_hints: int = 1
    hint_granularity: HintGranularity = HintGranularity.MEDIUM
    include_examples: bool = True
    scaffold_steps: bool = True
    allow_skip_hints: bool = True


class StrictnessConfig(BaseModel):
    error_tolerance: ErrorTolerance = ErrorTolerance.MEDIUM
    evaluation_strictness: float = Field(0.6, ge=0.0, le=1.0)
    partial_credit: bool = True
    pass_threshold: float = Field(0.7, ge=0.0, le=1.0)
    penalize_mistakes: bool = False
    feedback_verbosity: str = "detailed"
    explain_errors: bool = True


class ActiveRecallConfig(BaseModel):
    recall_frequency: RecallFrequency = RecallFrequency.SOMETIMES
    start_with_recall: bool = False
    spaced_repetition: bool = True
    repeat_interval_hours: int = 24
    quiz_previous: bool = False


class DepthConfig(BaseModel):
    explanation_depth: ExplanationDepth = ExplanationDepth.STANDARD
    include_background: bool = True
    include_applications: bool = True
    connect_concepts: bool = True
    max_explanation_chars: int = 0
    ask_followups: bool = True
    followup_count: int = 2


class EngagementConfig(BaseModel):
    tone: Tone = Tone.NEUTRAL
    use_emoji: bool = True
    celebrate_success: bool = True
    encourage_mistakes: bool = True
    allow_exploration: bool = False
    session_warning_minutes: int = 0


class TimeConfig(BaseModel):
    time_pressure: TimePressure = TimePressure.NONE
    show_timer: bool = False
    time_per_question: int = 0
    auto_submit: bool = False
    allow_extension: bool = True


class CheckpointsConfig(BaseModel):
    check_frequency: CheckFrequency = CheckFrequency.REGULAR
    min_checkpoints: int = 2
    max_checkpoints: int = 4
    checkpoint_types: list[str] = Field(default_factory=lambda: ["quiz", "self_assessment"])
    allow_skip: bool = True


class ResponseProfileConfig(BaseModel):
    system_prompt_suffix: str = ""
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = 1500
    response_format: ResponseFormat = ResponseFormat.MARKDOWN


class ModuleOverridesConfig(BaseModel):
    solve: dict[str, Any] = Field(default_factory=dict)
    guide: dict[str, Any] = Field(default_factory=dict)
    question: dict[str, Any] = Field(default_factory=dict)
    research: dict[str, Any] = Field(default_factory=dict)
    co_writer: dict[str, Any] = Field(default_factory=dict)


class PolicyConfig(BaseModel):
    policy_id: str
    name: str
    description: str
    version: str = "1.0.0"
    effective_date: str = "2024-01-01"

    hinting: HintingConfig = Field(default_factory=HintingConfig)
    strictness: StrictnessConfig = Field(default_factory=StrictnessConfig)
    active_recall: ActiveRecallConfig = Field(default_factory=ActiveRecallConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    engagement: EngagementConfig = Field(default_factory=EngagementConfig)
    time: TimeConfig = Field(default_factory=TimeConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    response_profile: ResponseProfileConfig = Field(default_factory=ResponseProfileConfig)
    modules: ModuleOverridesConfig = Field(default_factory=ModuleOverridesConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyConfig":
        hinting_data = data.get("hinting", {})
        strictness_data = data.get("strictness", {})
        active_recall_data = data.get("active_recall", {})
        depth_data = data.get("depth", {})
        engagement_data = data.get("engagement", {})
        time_data = data.get("time", {})
        checkpoints_data = data.get("checkpoints", {})
        response_profile_data = data.get("response_profile", {})
        modules_data = data.get("modules", {})

        return cls(
            policy_id=data["policy_id"],
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            effective_date=data.get("effective_date", "2024-01-01"),
            hinting=HintingConfig(**hinting_data),
            strictness=StrictnessConfig(**strictness_data),
            active_recall=ActiveRecallConfig(**active_recall_data),
            depth=DepthConfig(**depth_data),
            engagement=EngagementConfig(**engagement_data),
            time=TimeConfig(**time_data),
            checkpoints=CheckpointsConfig(**checkpoints_data),
            response_profile=ResponseProfileConfig(**response_profile_data),
            modules=ModuleOverridesConfig(**modules_data),
        )

    def get_system_prompt(self, base_prompt: str) -> str:
        suffix = self.response_profile.system_prompt_suffix
        if suffix:
            return f"{base_prompt}\n\n{suffix}"
        return base_prompt

    def get_llm_params(self) -> dict[str, Any]:
        return {
            "temperature": self.response_profile.temperature,
            "max_tokens": self.response_profile.max_tokens,
        }


@dataclass
class PolicySession:
    """Session-bound policy state."""

    policy: PolicyConfig
    subject: str | None = None
    module: str | None = None
    started_at: float = field(default_factory=lambda: __import__("time").time())

    def get_hint_config(self) -> HintingConfig:
        return self.policy.hinting

    def get_strictness(self) -> StrictnessConfig:
        return self.policy.strictness

    def get_active_recall(self) -> ActiveRecallConfig:
        return self.policy.active_recall

    def get_depth(self) -> DepthConfig:
        return self.policy.depth

    def get_engagement(self) -> EngagementConfig:
        return self.policy.engagement

    def get_time(self) -> TimeConfig:
        return self.policy.time

    def get_checkpoints(self) -> CheckpointsConfig:
        return self.policy.checkpoints

    def get_module_override(self, module_name: str) -> dict[str, Any]:
        module_overrides = getattr(self.policy.modules, module_name, {})
        if isinstance(module_overrides, dict):
            return module_overrides
        return {}

    def should_celebrate_success(self) -> bool:
        return self.policy.engagement.celebrate_success

    def should_allow_exploration(self) -> bool:
        return self.policy.engagement.allow_exploration

    def get_evaluation_strictness(self) -> float:
        return self.policy.strictness.evaluation_strictness

    def get_pass_threshold(self) -> float:
        return self.policy.strictness.pass_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy.policy_id,
            "policy_name": self.policy.name,
            "subject": self.subject,
            "module": self.module,
            "started_at": self.started_at,
            "hinting": {
                "max_hints": self.policy.hinting.max_hints,
                "min_hints": self.policy.hinting.min_hints,
                "granularity": self.policy.hinting.hint_granularity.value,
            },
            "strictness": {
                "evaluation_strictness": self.policy.strictness.evaluation_strictness,
                "pass_threshold": self.policy.strictness.pass_threshold,
            },
            "active_recall": {
                "frequency": self.policy.active_recall.recall_frequency.value,
                "spaced_repetition": self.policy.active_recall.spaced_repetition,
            },
        }


class PolicyEngine:
    """Loads and manages learning policies."""

    def __init__(self, policies_dir: str | None = None):
        if policies_dir is None:
            policies_dir = Path(__file__).parent
        self.policies_dir = Path(policies_dir)
        self._policies: dict[str, PolicyConfig] = {}
        self._sessions: dict[str, PolicySession] = {}
        self._load_all_policies()

    def _load_all_policies(self) -> None:
        """Load all policy files from the policies directory."""
        policy_files = list(self.policies_dir.glob("*.yaml"))
        for policy_file in policy_files:
            if policy_file.name.startswith("."):
                continue
            try:
                with open(policy_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data and "policy_id" in data:
                    policy = PolicyConfig.from_dict(data)
                    self._policies[policy.policy_id] = policy
            except Exception as e:
                print(f"Warning: Failed to load policy from {policy_file}: {e}")

    def list_policies(self) -> list[dict[str, str]]:
        """List all available policies."""
        return [
            {
                "id": pid,
                "name": p.name,
                "description": p.description[:100] + "..."
                if len(p.description) > 100
                else p.description,
            }
            for pid, p in self._policies.items()
        ]

    def get_policy(self, policy_id: str) -> PolicyConfig | None:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    def get_policy_names(self) -> list[str]:
        """Get list of policy IDs."""
        return list(self._policies.keys())

    def create_session(
        self,
        policy_id: str,
        session_id: str,
        subject: str | None = None,
        module: str | None = None,
    ) -> PolicySession | None:
        """Create a new policy session."""
        policy = self._policies.get(policy_id)
        if policy is None:
            return None

        session = PolicySession(
            policy=policy,
            subject=subject,
            module=module,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> PolicySession | None:
        """Get an existing session."""
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        """End and clean up a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def apply_policy_to_prompt(
        self,
        policy_id: str,
        base_system_prompt: str,
    ) -> str:
        """Apply policy to a system prompt."""
        policy = self._policies.get(policy_id)
        if policy is None:
            return base_system_prompt
        return policy.get_system_prompt(base_system_prompt)

    def get_llm_params_for_policy(self, policy_id: str) -> dict[str, Any]:
        """Get LLM parameters for a policy."""
        policy = self._policies.get(policy_id)
        if policy is None:
            return {"temperature": 0.7, "max_tokens": 1500}
        return policy.get_llm_params()


_engine: PolicyEngine | None = None


def get_policy_engine(policies_dir: str | None = None) -> PolicyEngine:
    """Get the global policy engine instance."""
    global _engine
    if _engine is None:
        _engine = PolicyEngine(policies_dir=policies_dir)
    return _engine


def reset_policy_engine() -> None:
    """Reset the global policy engine (for testing)."""
    global _engine
    _engine = None


def get_policy(policy_id: str) -> PolicyConfig | None:
    """Convenience function to get a policy by ID."""
    return get_policy_engine().get_policy(policy_id)


def create_session(
    policy_id: str,
    session_id: str,
    subject: str | None = None,
    module: str | None = None,
) -> PolicySession | None:
    """Convenience function to create a policy session."""
    return get_policy_engine().create_session(policy_id, session_id, subject, module)
