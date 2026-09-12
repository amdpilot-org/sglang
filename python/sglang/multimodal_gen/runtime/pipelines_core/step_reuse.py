"""Side-effect-aware prediction reuse for iterative diffusion pipelines."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass(frozen=True, slots=True)
class StepReusePolicy:
    policy_name: str
    observation_point: str
    history_size: int
    max_skip_steps: int
    force_real_steps: frozenset[str] = frozenset()
    state_scope: tuple[str, ...] = ("request",)
    trace_only: bool = False

    def __post_init__(self) -> None:
        if not self.policy_name or not self.observation_point:
            raise ValueError("step-reuse policy and observation point must be named")
        if self.history_size < 1:
            raise ValueError("step-reuse history_size must be at least 1")
        if self.max_skip_steps < 0:
            raise ValueError("step-reuse max_skip_steps cannot be negative")
        if not self.state_scope or len(set(self.state_scope)) != len(self.state_scope):
            raise ValueError("step-reuse state_scope must contain unique scope fields")
        unknown = self.force_real_steps - {"first_two", "terminal"}
        if unknown:
            raise ValueError(f"unknown forced-real step rules: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class StepSideEffectContract:
    terminal_write_required: bool = False
    write_tags: frozenset[str] = frozenset()


@dataclass(slots=True)
class StepReuseState:
    last_real_prediction: torch.Tensor | None = None
    real_history: deque[torch.Tensor] = field(default_factory=deque)
    skip_remaining: int = 0
    real_forwards: int = 0
    reused_steps: int = 0
    trace_only_reuse_steps: int = 0
    possible_forwards: int = 0
    scheduler_steps: int = 0
    reuse_streak: int = 0
    reuse_streaks: list[int] = field(default_factory=list)
    forced_real_reasons: list[str] = field(default_factory=list)
    decision_statistics: list[Mapping[str, float]] = field(default_factory=list)
    terminal_write_verified: bool | None = None


@dataclass(frozen=True, slots=True)
class StepReuseDecision:
    reuse: bool
    reason: str
    skip_remaining: int = 0


class StepReuseAdapter(Protocol):
    policy: StepReusePolicy
    side_effect_contract: StepSideEffectContract

    def scope_values(self, batch: Any) -> Mapping[str, Any]: ...

    def observe_real_prediction(
        self, prediction: torch.Tensor, *, step_index: int, state: StepReuseState
    ) -> torch.Tensor: ...

    def decide_reuse(
        self, *, step_index: int, state: StepReuseState
    ) -> int | tuple[int, Mapping[str, float]]: ...

    def verify_side_effects(self, *, state: StepReuseState) -> bool: ...


DecisionSynchronizer = Callable[[StepReuseDecision | None], StepReuseDecision]


class StepReuseController:
    """Own request-local reuse state; model adapters own similarity semantics."""

    def __init__(
        self,
        adapter: StepReuseAdapter,
        *,
        synchronize_decision: DecisionSynchronizer | None = None,
        is_control_rank: bool = True,
    ) -> None:
        self.adapter = adapter
        self.policy = adapter.policy
        self.contract = adapter.side_effect_contract
        self._synchronize = synchronize_decision
        self._is_control_rank = is_control_rank
        self._states: dict[tuple[Any, ...], StepReuseState] = {}
        self._active_scope: tuple[Any, ...] | None = None

    @property
    def state(self) -> StepReuseState:
        if self._active_scope is None:
            raise RuntimeError("begin_scope must be called before using step reuse")
        return self._states[self._active_scope]

    def begin_scope(self, batch: Any) -> StepReuseState:
        values = self.adapter.scope_values(batch)
        missing = set(self.policy.state_scope) - set(values)
        extra = set(values) - set(self.policy.state_scope)
        if missing or extra:
            raise ValueError(
                f"step-reuse scope mismatch (missing={sorted(missing)}, extra={sorted(extra)})"
            )
        key = tuple(values[name] for name in self.policy.state_scope)
        state = StepReuseState(real_history=deque(maxlen=self.policy.history_size))
        self._states[key] = state
        self._active_scope = key
        return state

    def before_step(self, step_index: int, num_steps: int) -> StepReuseDecision:
        state = self.state
        state.possible_forwards += 1
        local = None
        if self._is_control_rank:
            reason = self._forced_real_reason(step_index, num_steps)
            if reason is not None:
                local = StepReuseDecision(False, reason, state.skip_remaining)
            elif state.skip_remaining > 0:
                local = StepReuseDecision(
                    True, "reuse_budget", state.skip_remaining - 1
                )
            else:
                local = StepReuseDecision(False, "budget_exhausted", 0)
        decision = self._synchronize(local) if self._synchronize else local
        if not isinstance(decision, StepReuseDecision):
            raise RuntimeError(
                "step-reuse decision synchronization returned no decision"
            )
        state.skip_remaining = max(
            0, min(decision.skip_remaining, self.policy.max_skip_steps)
        )
        if not decision.reuse and decision.reason != "budget_exhausted":
            state.forced_real_reasons.append(decision.reason)
        return decision

    def _forced_real_reason(self, step_index: int, num_steps: int) -> str | None:
        state = self.state
        if step_index == num_steps - 1 and (
            "terminal" in self.policy.force_real_steps
            or self.contract.terminal_write_required
        ):
            return "terminal_side_effect"
        if (
            state.last_real_prediction is None
            or len(state.real_history) < self.policy.history_size
        ):
            return "history_warmup"
        if "first_two" in self.policy.force_real_steps and step_index < 2:
            return "first_two"
        return None

    def reused_prediction(self) -> torch.Tensor:
        state = self.state
        if state.last_real_prediction is None:
            raise RuntimeError("cannot reuse before a real prediction")
        state.reused_steps += 1
        state.reuse_streak += 1
        return state.last_real_prediction

    def after_real_forward(self, prediction: torch.Tensor, step_index: int) -> None:
        state = self.state
        if state.reuse_streak:
            state.reuse_streaks.append(state.reuse_streak)
            state.reuse_streak = 0
        observation = self.adapter.observe_real_prediction(
            prediction, step_index=step_index, state=state
        )
        if not isinstance(observation, torch.Tensor):
            raise TypeError("observe_real_prediction must return a tensor")
        # The scheduler may mutate or retain its model_output argument. Keep the
        # reusable observation independently owned for the rest of this scope.
        observation = observation.detach().clone()
        state.last_real_prediction = observation
        state.real_history.append(observation)
        state.real_forwards += 1
        if (
            len(state.real_history) < self.policy.history_size
            or not self._is_control_rank
        ):
            state.skip_remaining = 0
            return
        result = self.adapter.decide_reuse(step_index=step_index, state=state)
        statistics: Mapping[str, float] = {}
        if isinstance(result, tuple):
            budget, statistics = result
        else:
            budget = result
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError("decide_reuse must return an integer skip budget")
        if budget < 0 or budget > self.policy.max_skip_steps:
            raise ValueError(
                f"adapter skip budget {budget} is outside [0, {self.policy.max_skip_steps}]"
            )
        state.skip_remaining = budget
        if statistics:
            state.decision_statistics.append(dict(statistics))

    def after_scheduler_step(self) -> None:
        self.state.scheduler_steps += 1

    def record_trace_only_reuse(self) -> None:
        self.state.trace_only_reuse_steps += 1

    def finalize_scope(self) -> dict[str, Any]:
        state = self.state
        if state.reuse_streak:
            state.reuse_streaks.append(state.reuse_streak)
            state.reuse_streak = 0
        if self.contract.terminal_write_required:
            state.terminal_write_verified = bool(
                self.adapter.verify_side_effects(state=state)
            )
            if not state.terminal_write_verified:
                raise RuntimeError(
                    "mandatory terminal side effects were not verified: "
                    + str(sorted(self.contract.write_tags))
                )
        result = {
            "policy_name": self.policy.policy_name,
            "observation_point": self.policy.observation_point,
            "possible_forwards": state.possible_forwards,
            "real_forwards": state.real_forwards,
            "reused_steps": state.reused_steps,
            "trace_only_reuse_steps": state.trace_only_reuse_steps,
            "reuse_streaks": list(state.reuse_streaks),
            "forced_real_reasons": list(state.forced_real_reasons),
            "decision_statistics": list(state.decision_statistics),
            "scheduler_steps": state.scheduler_steps,
            "terminal_write_verified": state.terminal_write_verified,
            "trace_only": self.policy.trace_only,
        }
        self._states.pop(self._active_scope, None)
        self._active_scope = None
        return result

    def abort_scope(self) -> None:
        if self._active_scope is not None:
            self._states.pop(self._active_scope, None)
            self._active_scope = None
