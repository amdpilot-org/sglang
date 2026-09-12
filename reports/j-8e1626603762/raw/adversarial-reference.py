import itertools
from pathlib import Path
from tempfile import TemporaryDirectory

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)


N = 4
KEYS = [f"p{i}" for i in range(1, N + 1)]


def legal_endpoints(kv_prefix, pages, policy, trailing):
    endpoints = set()
    for endpoint in range(1, kv_prefix + 1):
        if policy == PoolHitPolicy.ALL_PAGES:
            required = range(1, endpoint + 1)
        else:
            required = range(max(1, endpoint - trailing + 1), endpoint + 1)
        if all(page in pages for page in required):
            endpoints.add(endpoint)
    return endpoints


def query(kv_prefix, specs):
    with TemporaryDirectory() as directory:
        backend = HiCacheFile.__new__(HiCacheFile)
        backend.file_path = directory
        backend.config_suffix = "_adversarial"
        backend.metadata_cache = None
        for page in range(1, kv_prefix + 1):
            Path(backend._get_component_path(KEYS[page - 1], PoolName.KV)).touch()
        transfers = []
        for name, pages, policy, trailing in specs:
            for page in pages:
                Path(backend._get_component_path(KEYS[page - 1], name)).touch()
            transfers.append(
                PoolTransfer(
                    name=name,
                    keys=KEYS[:trailing] if trailing else None,
                    hit_policy=policy,
                )
            )
        return backend.batch_exists_v2(KEYS, transfers)


checked = 0
names = [PoolName.SWA, PoolName.MAMBA]
policies = [PoolHitPolicy.ALL_PAGES, PoolHitPolicy.TRAILING_PAGES]
page_sets = [
    {i + 1 for i, bit in enumerate(bits) if bit}
    for bits in itertools.product([False, True], repeat=N)
]

for kv_prefix in range(N + 1):
    for policy_pair in itertools.product(policies, repeat=2):
        trailing_ranges = [
            range(1, 4) if policy == PoolHitPolicy.TRAILING_PAGES else [1]
            for policy in policy_pair
        ]
        for trailing_pair in itertools.product(*trailing_ranges):
            for pages_a in page_sets:
                for pages_b in page_sets:
                    specs = [
                        (names[0], pages_a, policy_pair[0], trailing_pair[0]),
                        (names[1], pages_b, policy_pair[1], trailing_pair[1]),
                    ]
                    expected = set(range(1, kv_prefix + 1))
                    for _, pages, policy, trailing in specs:
                        expected &= legal_endpoints(
                            kv_prefix, pages, policy, trailing
                        )
                    result = query(kv_prefix, specs)
                    expected_list = sorted(expected)
                    assert result.restorable_prefix_pages == expected_list, (
                        kv_prefix,
                        specs,
                        expected_list,
                        result,
                    )
                    assert result.kv_hit_pages == (
                        expected_list[-1] if expected_list else 0
                    ), (kv_prefix, specs, expected_list, result)
                    checked += 1

print(f"checked={checked}")
