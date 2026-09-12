"""Memory accounting helpers for Apple's unified-memory MPS backend."""

from __future__ import annotations

from typing import Any

import psutil


def get_mps_recommended_memory(mps: Any) -> int:
    """Return Metal's recommended working-set size.

    Host RAM is not a safe fallback because Metal's working-set limit is
    smaller and device-specific.  Fail closed when PyTorch cannot provide a
    positive limit instead of inviting an allocation beyond Metal's limit.
    """
    recommended_max_memory = getattr(mps, "recommended_max_memory", None)
    if not callable(recommended_max_memory):
        raise RuntimeError(
            "Cannot determine the safe MPS memory limit because "
            "torch.mps.recommended_max_memory() is unavailable. Upgrade to a "
            "PyTorch build that exposes the Metal recommended working-set limit."
        )

    try:
        recommended = int(recommended_max_memory())
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "Cannot determine the safe MPS memory limit because "
            "torch.mps.recommended_max_memory() failed."
        ) from exc

    if recommended <= 0:
        raise RuntimeError(
            "Cannot determine the safe MPS memory limit because "
            "torch.mps.recommended_max_memory() returned a non-positive value."
        )
    return recommended


def get_mps_available_memory(mps: Any) -> int:
    """Return safe additional MPS allocation headroom in bytes.

    Both constraints matter on UMA: system pressure can make less memory
    available than Metal's limit, while existing MPS allocations consume part
    of that limit even if the host still reports plenty of available RAM.
    """
    system_available = int(psutil.virtual_memory().available)
    recommended = get_mps_recommended_memory(mps)

    driver_allocated_memory = getattr(mps, "driver_allocated_memory", None)
    if not callable(driver_allocated_memory):
        raise RuntimeError(
            "Cannot determine safe MPS allocation headroom because "
            "torch.mps.driver_allocated_memory() is unavailable."
        )

    try:
        allocated = int(driver_allocated_memory())
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "Cannot determine safe MPS allocation headroom because "
            "torch.mps.driver_allocated_memory() failed."
        ) from exc
    if allocated < 0:
        raise RuntimeError(
            "Cannot determine safe MPS allocation headroom because "
            "torch.mps.driver_allocated_memory() returned a negative value."
        )
    return min(system_available, max(recommended - allocated, 0))
