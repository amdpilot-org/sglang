import importlib
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


def _import_lmc_radix_cache():
    """Import the cache with the absent optional LMCache dependency stubbed."""
    modules = {
        "lmcache": types.ModuleType("lmcache"),
        "lmcache.integration": types.ModuleType("lmcache.integration"),
        "lmcache.integration.sglang": types.ModuleType("lmcache.integration.sglang"),
        "lmcache.integration.sglang.multi_process_adapter": types.ModuleType(
            "lmcache.integration.sglang.multi_process_adapter"
        ),
        "lmcache.integration.sglang.sglang_adapter": types.ModuleType(
            "lmcache.integration.sglang.sglang_adapter"
        ),
        "lmcache.integration.sglang.utils": types.ModuleType(
            "lmcache.integration.sglang.utils"
        ),
    }
    modules[
        "lmcache.integration.sglang.multi_process_adapter"
    ].LMCacheMPConnector = object
    adapter = modules["lmcache.integration.sglang.sglang_adapter"]
    adapter.LMCacheLayerwiseConnector = object
    adapter.LoadMetadata = SimpleNamespace
    adapter.StoreMetadata = SimpleNamespace
    modules["lmcache.integration.sglang.utils"].lmcache_get_config = MagicMock()
    with patch.dict(sys.modules, modules):
        sys.modules.pop(
            "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache", None
        )
        return importlib.import_module(
            "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache"
        )


mod = _import_lmc_radix_cache()


class ControlledFuture:
    def __init__(self, result=True, error=None):
        self.value = result
        self.error = error
        self.called = False

    def result(self):
        self.called = True
        if self.error is not None:
            raise self.error
        return self.value


def _bare_cache(mode=mod.LMCacheMode.MP):
    cache = mod.LMCRadixCache.__new__(mod.LMCRadixCache)
    cache.disable = False
    cache._mode = mode
    cache._node_lock = threading.Lock()
    cache._in_flight_nodes = []
    cache._in_flight_mp_stores = []
    cache._mp_load_back_markers = {}
    cache.store_stream = MagicMock()
    cache.lmcache_connector = MagicMock()
    cache.dec_lock_ref = MagicMock()
    return cache


def test_mp_finish_submits_without_waiting_and_keeps_node_locked():
    cache = _bare_cache()
    node = object()
    future = ControlledFuture()
    cache.lmcache_connector.store_kv_async.return_value = future
    cache.inc_lock_ref = MagicMock()
    cache.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[7, 8, 9]], dtype=torch.int64)
    )
    req = SimpleNamespace(
        rid="request-1",
        origin_input_ids=[1, 2, 3],
        output_ids=[],
        extra_key=None,
        cache_salt=None,
        kv=SimpleNamespace(req_pool_idx=0, kv_committed_len=3),
    )

    with (
        patch.object(mod.RadixCache, "cache_finished_req"),
        patch.object(
            mod.RadixCache,
            "match_prefix",
            return_value=SimpleNamespace(last_device_node=node),
        ),
        patch.object(
            mod, "get_spec", return_value=SimpleNamespace(speculative_eagle_topk=None)
        ),
    ):
        cache.cache_finished_req(req, kv_len_to_handle=3)

    assert future.called is False
    cache.inc_lock_ref.assert_called_once_with(node)
    cache.dec_lock_ref.assert_not_called()
    cache.lmcache_connector.end_session.assert_not_called()
    assert cache._in_flight_mp_stores == [(node, future, "request-1")]


def test_evict_resolves_future_before_unlock_cleanup_and_slot_reuse():
    cache = _bare_cache()
    events = []
    node = object()

    class OrderedFuture:
        def result(self):
            events.append("future")
            return True

    cache._in_flight_mp_stores = [(node, OrderedFuture(), "request-1")]
    cache._mp_load_back_markers["request-1"] = object()
    cache.dec_lock_ref.side_effect = lambda _: events.append("unlock")
    cache.lmcache_connector.end_session.side_effect = lambda _: events.append("end")

    with patch.object(
        mod.RadixCache, "evict", side_effect=lambda _: events.append("evict")
    ):
        cache.evict(mod.EvictParams(num_tokens=1))

    assert events == ["future", "unlock", "end", "evict"]
    assert "request-1" not in cache._mp_load_back_markers
    assert cache._in_flight_mp_stores == []


def test_submit_failure_releases_lock_marker_and_session():
    cache = _bare_cache()
    node = object()
    cache.lmcache_connector.store_kv_async.side_effect = RuntimeError("send failed")
    cache.inc_lock_ref = MagicMock()
    cache.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[7]], dtype=torch.int64)
    )
    cache._mp_load_back_markers["request-1"] = object()
    req = SimpleNamespace(
        rid="request-1",
        origin_input_ids=[1],
        output_ids=[],
        extra_key=None,
        cache_salt=None,
        kv=SimpleNamespace(req_pool_idx=0, kv_committed_len=1),
    )

    with (
        patch.object(mod.RadixCache, "cache_finished_req"),
        patch.object(
            mod.RadixCache,
            "match_prefix",
            return_value=SimpleNamespace(last_device_node=node),
        ),
        patch.object(
            mod, "get_spec", return_value=SimpleNamespace(speculative_eagle_topk=None)
        ),
        pytest.raises(RuntimeError, match="send failed"),
    ):
        cache.cache_finished_req(req, kv_len_to_handle=1)

    cache.dec_lock_ref.assert_called_once_with(node)
    cache.lmcache_connector.end_session.assert_called_once_with("request-1")
    assert cache._mp_load_back_markers == {}


@pytest.mark.parametrize(
    "future,match",
    [
        (ControlledFuture(result=False), "store failed"),
        (ControlledFuture(error=ValueError("daemon error")), "daemon error"),
    ],
)
def test_failed_future_still_unlocks_ends_session_and_clears_marker(future, match):
    cache = _bare_cache()
    node = object()
    cache._in_flight_mp_stores = [(node, future, "request-1")]
    cache._mp_load_back_markers["request-1"] = object()

    with pytest.raises(Exception, match=match):
        cache.evict(mod.EvictParams(num_tokens=1))

    cache.dec_lock_ref.assert_called_once_with(node)
    cache.lmcache_connector.end_session.assert_called_once_with("request-1")
    assert cache._in_flight_mp_stores == []
    assert cache._mp_load_back_markers == {}


def test_reset_drains_before_tree_reset_and_shutdown_drains_before_close():
    for operation in ("reset", "release_host_resources"):
        cache = _bare_cache()
        events = []
        node = object()
        cache._in_flight_mp_stores = [(node, ControlledFuture(), "request-1")]
        cache.dec_lock_ref.side_effect = lambda _: events.append("unlock")
        cache.lmcache_connector.end_session.side_effect = lambda _: events.append("end")
        cache.lmcache_connector.close.side_effect = lambda: events.append("close")

        with patch.object(
            mod.RadixCache, "reset", side_effect=lambda: events.append("reset")
        ):
            getattr(cache, operation)()

        expected_tail = "reset" if operation == "reset" else "close"
        assert events == ["unlock", "end", expected_tail]


def test_ip_reconciliation_is_unchanged_and_does_not_touch_mp_sessions():
    cache = _bare_cache(mode=mod.LMCacheMode.IP)
    node = object()
    cache._in_flight_nodes = [node]

    with patch.object(mod.RadixCache, "evict", return_value=mod.EvictResult()):
        cache.evict(mod.EvictParams(num_tokens=0))

    cache.store_stream.synchronize.assert_called_once_with()
    cache.dec_lock_ref.assert_called_once_with(node)
    cache.lmcache_connector.end_session.assert_not_called()
