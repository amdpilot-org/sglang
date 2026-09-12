# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic contract for action candidate trajectory ensembling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CandidateTrajectorySpec:
    """One logical action prediction sampled as an ordered candidate batch."""

    count: int = 1
    reducer: str = "none"
    return_candidates: bool = False
    seed_policy: str = "per_candidate"

    @classmethod
    def from_value(cls, value: Any) -> CandidateTrajectorySpec | None:
        if value is None:
            return None
        if isinstance(value, cls):
            spec = value
        elif isinstance(value, dict):
            unknown = sorted(
                set(value) - {"count", "reducer", "return_candidates", "seed_policy"}
            )
            if unknown:
                raise ValueError(f"Unknown candidate_trajectory fields: {unknown}")
            spec = cls(**value)
        else:
            raise ValueError("candidate_trajectory must be an object")
        spec.validate()
        return spec

    def validate(self) -> None:
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 1
        ):
            raise ValueError(
                f"candidate_trajectory.count must be a positive int, got {self.count!r}"
            )
        if self.reducer not in ("none", "mean"):
            raise ValueError(
                "candidate_trajectory.reducer must be 'none', 'mean', or a "
                f"model-registered reducer; got {self.reducer!r}"
            )
        if self.seed_policy != "per_candidate":
            raise ValueError(
                "candidate_trajectory.seed_policy currently supports only "
                f"'per_candidate', got {self.seed_policy!r}"
            )
        if not isinstance(self.return_candidates, bool):
            raise ValueError("candidate_trajectory.return_candidates must be a bool")
        if self.count > 1 and self.reducer == "none":
            raise ValueError(
                "candidate_trajectory.count > 1 requires a model-supported reducer"
            )


@dataclass(frozen=True)
class ActionCandidateCapability:
    tensor: str
    reduction_order: str
    reducers: tuple[str, ...]
    output_dtype: str
    output_shape: str
    candidate_invariant_conditioning: tuple[str, ...]
    auxiliary_branch_required: bool


def reduce_action_candidates(
    candidates: torch.Tensor,
    spec: CandidateTrajectorySpec,
    capability: ActionCandidateCapability,
) -> torch.Tensor:
    """Apply a validated model-declared reduction without hiding batch errors."""
    if candidates.ndim < 1 or candidates.shape[0] != spec.count:
        raise RuntimeError(
            "candidate action tensor has an invalid candidate axis: expected "
            f"{spec.count}, got shape {tuple(candidates.shape)}"
        )
    if spec.reducer not in capability.reducers:
        raise ValueError(
            f"Reducer {spec.reducer!r} is not supported for {capability.tensor}; "
            f"supported reducers: {list(capability.reducers)}"
        )
    # Preserve the legacy action tensor bit-for-bit for the compatibility case.
    if spec.count == 1:
        return candidates[0]
    if spec.reducer == "none":
        return candidates[0]
    if spec.reducer == "mean":
        return candidates.mean(dim=0)
    raise ValueError(f"No registered implementation for reducer {spec.reducer!r}")
