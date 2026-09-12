"""Contract tests for cache_salt at the SGLang -> LMCache MP boundary."""

import ast
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


SOURCE = Path(__file__).resolve().parents[4] / (
    "python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py"
)


def _load_method(name):
    tree = ast.parse(SOURCE.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LMCRadixCache"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0), method],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"logger": logging.getLogger(__name__)}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[name]


def _lookup_tree(connector):
    tree = SimpleNamespace(
        lmcache_connector=connector,
        _mp_load_back_markers={},
    )
    tree._mp_supports_cache_salt = MethodType(
        _load_method("_mp_supports_cache_salt"), tree
    )
    tree._mp_match_prefix = MethodType(_load_method("_mp_match_prefix"), tree)
    return tree


def _lookup_args(cache_salt):
    key = SimpleNamespace(
        cache_salt=cache_salt,
        token_ids=[11, 12, 13],
        raw_token_ids=lambda: [11, 12, 13],
    )
    base_res = object()
    return key, base_res, torch.empty(0), object(), SimpleNamespace(rid="request-1")


def test_salted_mp_lookup_is_forwarded_and_legacy_connectors_fail_closed():
    capable = SimpleNamespace(
        supports_cache_salt=True,
        lookup_kv=Mock(return_value=0),
        release_pending=Mock(),
    )
    _lookup_tree(capable)._mp_match_prefix(*_lookup_args("tenant-a"))
    capable.lookup_kv.assert_called_once_with(
        [11, 12, 13], "request-1", cache_salt="tenant-a"
    )

    legacy = SimpleNamespace(lookup_kv=Mock(return_value=0), release_pending=Mock())
    args = _lookup_args("tenant-a")
    assert _lookup_tree(legacy)._mp_match_prefix(*args) is args[1]
    legacy.lookup_kv.assert_not_called()


def test_unsalted_lookup_remains_compatible_with_legacy_connector():
    legacy = SimpleNamespace(lookup_kv=Mock(return_value=0), release_pending=Mock())
    _lookup_tree(legacy)._mp_match_prefix(*_lookup_args(None))
    legacy.lookup_kv.assert_called_once_with([11, 12, 13], "request-1")


def test_salted_mp_store_is_forwarded_and_legacy_connectors_fail_closed():
    source = SOURCE.read_text()
    assert 'store_metadata_kwargs["cache_salt"] = req.cache_salt or ""' in source
    assert "req.cache_salt and not supports_cache_salt" in source
    store_guard = source.index("req.cache_salt and not supports_cache_salt")
    store_call = source.index("self.lmcache_connector.store_kv", store_guard)
    assert store_guard < store_call


def test_mp_store_cleanup_is_exception_safe():
    source = SOURCE.read_text()
    store_call = source.index("self.lmcache_connector.store_kv")
    cleanup = source.index("self.dec_lock_ref(new_last_node)", store_call)
    assert "finally:" in source[store_call:cleanup]


def test_radix_key_documents_connector_specific_external_namespacing():
    radix_source = (SOURCE.parents[2] / "radix_cache.py").read_text()
    assert "External storage namespacing is connector-specific" in radix_source
