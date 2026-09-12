"""State for opt-in out-of-order prefill admission after a blocked queue head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class _HeldLock:
    head: Any
    node: Any
    receipt: Any
    pinned_tokens: int


class PrefillLookahead:
    """Keep the blocked head's matched prefix resident across batch execution.

    The scheduler owns request ordering and admission.  This object only owns
    the cross-pass aging and cache-lock lifecycle, and deliberately depends on
    the cache through its public duck-typed lock/size contract.
    """

    def __init__(
        self,
        tree_cache: Any,
        max_candidates: int,
        aging_passes: int,
        reserve_tokens: int,
    ):
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if aging_passes <= 0:
            raise ValueError("aging_passes must be positive")
        if reserve_tokens < 0:
            raise ValueError("reserve_tokens must be non-negative")
        self.tree_cache = tree_cache
        self.max_candidates = max_candidates
        self.aging_passes = aging_passes
        self.reserve_tokens = reserve_tokens
        self._head: Optional[Any] = None
        self._no_token_passes = 0
        self._held: Optional[_HeldLock] = None

    @property
    def held_head(self) -> Optional[Any]:
        return self._held.head if self._held is not None else None

    @property
    def pinned_tokens(self) -> int:
        return self._held.pinned_tokens if self._held is not None else 0

    def _available(self) -> int:
        allocator = self.tree_cache.token_to_kv_pool_allocator
        full_available = getattr(allocator, "full_available_size", None)
        return int(
            full_available()
            if full_available is not None
            else allocator.available_size()
        )

    def _evictable(self) -> int:
        full_evictable = getattr(self.tree_cache, "full_evictable_size", None)
        return int(
            full_evictable()
            if full_evictable is not None
            else self.tree_cache.evictable_size()
        )

    def release(self) -> None:
        held, self._held = self._held, None
        if held is not None:
            self.tree_cache.dec_lock_ref(held.node, held.receipt)

    def begin_pass(self, head: Optional[Any]) -> None:
        # A request is re-matched during each admission attempt.  Drop the old
        # anchor before that happens; this also balances locks when the head was
        # admitted, retracted, aborted, or otherwise removed between passes.
        self.release()
        if head is not self._head:
            self._head = head
            self._no_token_passes = 0

    def head_admitted(self, head: Any) -> None:
        if head is self._head:
            self.release()
            self._head = None
            self._no_token_passes = 0

    def head_blocked(self, head: Any, node: Any) -> bool:
        """Return whether candidates may be considered after this rejection."""
        if head is not self._head:
            self.begin_pass(head)
        self._no_token_passes += 1
        if self._no_token_passes > self.aging_passes:
            return False
        if not self.tree_cache.is_tree_cache() or node is None:
            return False

        before = self._evictable()
        acquire = self.tree_cache.inc_lock_ref(node)
        receipt = acquire.to_dec_params()
        after = self._evictable()
        delta = getattr(acquire, "delta", None)
        pinned = abs(int(delta)) if delta is not None else max(0, before - after)
        self._held = _HeldLock(head, node, receipt, pinned)

        # This is intentionally checked after acquiring: it uses the exact
        # cache-specific effect of the lock, including shared ancestors and
        # component-specific lock behavior.
        if self._available() + after < self.reserve_tokens:
            self.release()
            return False
        return True
