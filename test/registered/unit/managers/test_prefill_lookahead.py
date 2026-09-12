import unittest
from array import array

import torch

from sglang.srt.managers.prefill_lookahead import PrefillLookahead
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Receipt:
    def __init__(self, delta):
        self.delta = delta

    def to_dec_params(self):
        return self


class _Allocator:
    def __init__(self, available):
        self.available = available
        self.device = "cpu"

    def available_size(self):
        return self.available

    def full_available_size(self):
        return self.available


class _Cache:
    def __init__(self, available=100, evictable=100, pin=20, delta_sign=-1, tree=True):
        self.token_to_kv_pool_allocator = _Allocator(available)
        self.evictable = evictable
        self.pin = pin
        self.delta_sign = delta_sign
        self.tree = tree
        self.acquires = []
        self.releases = []

    def is_tree_cache(self):
        return self.tree

    def full_evictable_size(self):
        return self.evictable

    def inc_lock_ref(self, node):
        self.acquires.append(node)
        self.evictable -= self.pin
        delta = None if self.delta_sign is None else self.delta_sign * self.pin
        return _Receipt(delta)

    def dec_lock_ref(self, node, receipt):
        self.releases.append(node)
        self.evictable += self.pin


class TestPrefillLookahead(unittest.TestCase):
    def make(self, cache=None, aging=3, reserve=0):
        cache = cache or _Cache()
        return cache, PrefillLookahead(cache, 4, aging, reserve)

    def test_default_policy_values_are_validated(self):
        cache = _Cache()
        for args in ((0, 1, 0), (1, 0, 0), (1, 1, -1)):
            with self.assertRaises(ValueError):
                PrefillLookahead(cache, *args)

    def test_lock_survives_batch_then_releases_before_rematch(self):
        cache, policy = self.make()
        head = object()
        policy.begin_pass(head)
        self.assertTrue(policy.head_blocked(head, "node-a"))
        self.assertEqual(policy.held_head, head)
        policy.begin_pass(head)
        self.assertIsNone(policy.held_head)
        self.assertEqual(cache.acquires, ["node-a"])
        self.assertEqual(cache.releases, ["node-a"])

    def test_head_change_and_admission_balance_locks(self):
        cache, policy = self.make()
        first, second = object(), object()
        policy.begin_pass(first)
        self.assertTrue(policy.head_blocked(first, "first-node"))
        policy.begin_pass(second)
        self.assertEqual(cache.releases, ["first-node"])
        self.assertTrue(policy.head_blocked(second, "second-node"))
        policy.head_admitted(second)
        self.assertEqual(cache.releases, ["first-node", "second-node"])
        self.assertIsNone(policy.held_head)

    def test_retraction_or_abort_reconciles_when_head_leaves_queue(self):
        cache, policy = self.make()
        head = object()
        policy.begin_pass(head)
        self.assertTrue(policy.head_blocked(head, "node"))
        policy.begin_pass(None)
        self.assertEqual(cache.releases, ["node"])

    def test_aging_withholds_after_bounded_number_of_passes(self):
        cache, policy = self.make(aging=2)
        head = object()
        for expected in (True, True, False, False):
            policy.begin_pass(head)
            self.assertEqual(policy.head_blocked(head, "node"), expected)
        self.assertEqual(len(cache.acquires), 2)
        policy.begin_pass(object())
        new_head = policy._head
        self.assertTrue(policy.head_blocked(new_head, "new-node"))

    def test_documented_pin_reserve_oom_sequence_is_refused(self):
        # Observed failure sequence: pin=114752, next chunk reserve=4096,
        # available=1792.  The lock consumes every evictable token, so it must
        # be immediately unwound and no candidate may interlope.
        cache = _Cache(available=1792, evictable=114752, pin=114752)
        cache, policy = self.make(cache=cache, reserve=4096)
        head = object()
        policy.begin_pass(head)
        self.assertFalse(policy.head_blocked(head, "large-prefix"))
        self.assertEqual(cache.acquires, ["large-prefix"])
        self.assertEqual(cache.releases, ["large-prefix"])
        self.assertEqual(cache.evictable, 114752)

    def test_exact_reserve_is_allowed(self):
        cache = _Cache(available=1792, evictable=117056, pin=114752)
        cache, policy = self.make(cache=cache, reserve=4096)
        policy.begin_pass("head")
        self.assertTrue(policy.head_blocked("head", "node"))

    def test_pin_accounting_accepts_delta_sign_variants_and_none(self):
        for sign in (-1, 1, None):
            with self.subTest(sign=sign):
                cache = _Cache(pin=23, delta_sign=sign)
                cache, policy = self.make(cache=cache)
                policy.begin_pass("head")
                self.assertTrue(policy.head_blocked("head", "node"))
                self.assertEqual(policy.pinned_tokens, 23)
                policy.release()

    def test_chunk_cache_and_missing_match_never_enable_lookahead(self):
        for cache, node in ((_Cache(tree=False), "node"), (_Cache(), None)):
            policy = PrefillLookahead(cache, 1, 1, 0)
            policy.begin_pass("head")
            self.assertFalse(policy.head_blocked("head", node))
            self.assertEqual(cache.acquires, [])

    def test_real_radix_cache_lock_contract_and_balance(self):
        allocator = _Allocator(available=50)
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3, 4])),
                value=torch.tensor([10, 11, 12, 13]),
            )
        )
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4])))
        )
        self.assertEqual(cache.evictable_size(), 4)

        policy = PrefillLookahead(
            cache, max_candidates=2, aging_passes=2, reserve_tokens=50
        )
        policy.begin_pass("head")
        self.assertTrue(policy.head_blocked("head", match.last_device_node))
        self.assertEqual(policy.pinned_tokens, 4)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 4)

        policy.begin_pass("head")
        self.assertEqual(cache.evictable_size(), 4)
        self.assertEqual(cache.protected_size(), 0)


if __name__ == "__main__":
    unittest.main()
