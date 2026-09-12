"""CPU coverage for hybrid-prefix queries in the HiCache file backend."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _query(kv_pages, pools):
    with TemporaryDirectory() as directory:
        backend = HiCacheFile.__new__(HiCacheFile)
        backend.file_path = directory
        backend.config_suffix = "_batch_exists_test"
        backend.metadata_cache = None
        keys = ["p1", "p2", "p3"]

        for name, pages, _ in [(PoolName.KV, kv_pages, None), *pools]:
            for page in pages:
                Path(backend._get_component_path(keys[page - 1], name)).touch()

        transfers = [
            PoolTransfer(name=name, keys=[keys[-1]], hit_policy=policy)
            for name, _, policy in pools
        ]
        return backend.batch_exists_v2(keys, transfers)


@pytest.mark.parametrize(
    ("swa_pages", "mamba_pages", "expected", "restorable"),
    [
        ({3}, {3}, 3, [3]),
        ({3}, {2}, 0, []),
        ({1, 3}, {1, 2}, 1, [1]),
    ],
)
def test_trailing_pools_use_common_valid_endpoint(
    swa_pages, mamba_pages, expected, restorable
):
    result = _query(
        {1, 2, 3},
        [
            (PoolName.SWA, swa_pages, PoolHitPolicy.TRAILING_PAGES),
            (PoolName.MAMBA, mamba_pages, PoolHitPolicy.TRAILING_PAGES),
        ],
    )

    assert result.kv_hit_pages == expected
    assert result.restorable_prefix_pages == restorable


def test_empty_kv_prefix_is_not_restorable():
    result = _query(
        set(),
        [(PoolName.SWA, {1, 2, 3}, PoolHitPolicy.TRAILING_PAGES)],
    )

    assert result.kv_hit_pages == 0
    assert result.extra_pool_hit_pages == {}
    assert result.restorable_prefix_pages == []


@pytest.mark.parametrize(
    ("trailing_pages", "expected", "restorable"),
    [
        ({3}, 0, []),
        ({1, 2, 3}, 2, [1, 2]),
    ],
)
def test_all_pages_boundary_intersects_trailing_endpoints(
    trailing_pages, expected, restorable
):
    result = _query(
        {1, 2, 3},
        [
            (PoolName.INDEXER, {1, 2}, PoolHitPolicy.ALL_PAGES),
            (PoolName.SWA, trailing_pages, PoolHitPolicy.TRAILING_PAGES),
        ],
    )

    assert result.kv_hit_pages == expected
    assert result.restorable_prefix_pages == restorable
