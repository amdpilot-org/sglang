"""Memory accounting helpers for Apple's unified-memory MPS backend."""

from __future__ import annotations

from typing import Any

import psutil


def get_mps_recommended_memory(mps: Any) -> int:
    """Return Metal's recommended working-set size, if PyTorch exposes it.

    Older PyTorch builds do not provide ``recommended_max_memory``.  Keeping
    the system-memory fallback preserves compatibility with those builds,
    while current builds use Metal's per-device guardrail.
    """
    recommended_max_memory = getattr(mps, "recommended_max_memory", None)
    if recommended_max_memory is not None:
        try:
            recommended = int(recommended_max_memory())
        except (RuntimeError, TypeError, ValueError):
            recommended = 0
        if recommended > 0:
            return recommended
    return int(psutil.virtual_memory().total)


def get_mps_available_memory(mps: Any) -> int:
    """Return safe additional MPS allocation headroom in bytes.

    Both constraints matter on UMA: system pressure can make less memory
    available than Metal's limit, while existing MPS allocations consume part
    of that limit even if the host still reports plenty of available RAM.
    """
    system_available = int(psutil.virtual_memory().available)
    recommended = get_mps_recommended_memory(mps)

    driver_allocated_memory = getattr(mps, "driver_allocated_memory", None)
    if driver_allocated_memory is None:
        return min(system_available, recommended)

    try:
        allocated = max(int(driver_allocated_memory()), 0)
    except (RuntimeError, TypeError, ValueError):
        allocated = 0
    return min(system_available, max(recommended - allocated, 0))
