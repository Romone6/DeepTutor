"""Extensions policies module.

Policy management for controlling strictness, depth, hinting, and active recall behavior.

Usage:
    from extensions.policies import get_policy, create_session, PolicyAwareAgent

    # Get a policy
    policy = get_policy("term")

    # Create a session with a policy
    session = create_session("exam", session_id="sess_123", subject="math")

    # Use PolicyAwareAgent mixin
    class MyAgent(PolicyAwareAgent):
        def answer(self, question):
            if self.should_use_active_recall():
                question = self.add_recall_prompt(question)
            ...
"""

from .engine import (
    PolicyEngine,
    PolicyConfig,
    PolicySession,
    get_policy_engine,
    get_policy,
    create_session,
    reset_policy_engine,
    HintingConfig,
    StrictnessConfig,
    ActiveRecallConfig,
    DepthConfig,
    EngagementConfig,
    TimeConfig,
    CheckpointsConfig,
    ResponseProfileConfig,
)

from .enforce import (
    PolicyAwareAgent,
    PolicyContextManager,
    enforce_hinting,
    enforce_strictness,
    apply_policy_prompt,
    get_policy_params,
    with_policy,
)

__all__ = [
    "PolicyEngine",
    "PolicyConfig",
    "PolicySession",
    "get_policy_engine",
    "get_policy",
    "create_session",
    "reset_policy_engine",
    "HintingConfig",
    "StrictnessConfig",
    "ActiveRecallConfig",
    "DepthConfig",
    "EngagementConfig",
    "TimeConfig",
    "CheckpointsConfig",
    "ResponseProfileConfig",
    "PolicyAwareAgent",
    "PolicyContextManager",
    "enforce_hinting",
    "enforce_strictness",
    "apply_policy_prompt",
    "get_policy_params",
    "with_policy",
]
