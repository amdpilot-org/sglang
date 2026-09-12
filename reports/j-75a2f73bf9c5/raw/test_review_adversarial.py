import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


HELPER_PATH = Path(__file__).with_name("test_lmcache_mp_async_store.py")
spec = importlib.util.spec_from_file_location("candidate_helpers", HELPER_PATH)
helpers = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(helpers)

mod = helpers.mod


def test_completed_store_is_not_reaped_at_scheduler_budget_boundary():
    cache = helpers._bare_cache()
    node = object()
    future = helpers.ControlledFuture(result=True)
    cache._in_flight_mp_stores = [(node, future, "request-1")]
    cache._mp_load_back_markers["request-1"] = object()

    # Model a tree whose only cached slots are still locked. The ordinary
    # radix budget therefore reports no evictable capacity and no allocation
    # path has a reason to call evict().
    with patch.object(mod.RadixCache, "evictable_size", return_value=0):
        assert cache.evictable_size() == 0

    # Contract expectation: once completion is observable, the scheduling
    # budget boundary must release the pin so these slots can be reused.
    assert future.called is True
    cache.dec_lock_ref.assert_called_once_with(node)
    assert cache._in_flight_mp_stores == []


def test_reconcile_drains_all_entries_after_first_future_fails():
    cache = helpers._bare_cache()
    first = object()
    second = object()
    events = []

    class FailedFuture:
        def result(self):
            events.append("first_future")
            raise RuntimeError("first failed")

    class SuccessfulFuture:
        def result(self):
            events.append("second_future")
            return True

    cache._in_flight_mp_stores = [
        (first, FailedFuture(), "request-1"),
        (second, SuccessfulFuture(), "request-2"),
    ]
    cache._mp_load_back_markers = {"request-1": object(), "request-2": object()}
    cache.dec_lock_ref.side_effect = lambda node: events.append(
        "unlock_first" if node is first else "unlock_second"
    )
    cache.lmcache_connector.end_session.side_effect = (
        lambda rid: events.append(f"end_{rid}")
    )

    error = cache._reconcile_in_flight_stores()

    assert isinstance(error, RuntimeError)
    assert str(error) == "first failed"
    assert events == [
        "first_future",
        "unlock_first",
        "end_request-1",
        "second_future",
        "unlock_second",
        "end_request-2",
    ]
    assert cache._in_flight_mp_stores == []
    assert cache._mp_load_back_markers == {}


def test_ip_sync_failure_still_unlocks_ip_and_reconciles_mp():
    cache = helpers._bare_cache(mode=mod.LMCacheMode.IP)
    ip_node = object()
    mp_node = object()
    future = helpers.ControlledFuture(result=True)
    cache._in_flight_nodes = [ip_node]
    cache._in_flight_mp_stores = [(mp_node, future, "request-1")]
    cache._mp_load_back_markers["request-1"] = object()
    cache.store_stream.synchronize.side_effect = RuntimeError("sync failed")

    error = cache._reconcile_in_flight_stores()

    assert isinstance(error, RuntimeError)
    assert str(error) == "sync failed"
    assert future.called is True
    assert cache.dec_lock_ref.call_count == 2
    cache.lmcache_connector.end_session.assert_called_once_with("request-1")
    assert cache._in_flight_nodes == []
    assert cache._in_flight_mp_stores == []
