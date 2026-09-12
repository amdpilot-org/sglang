import importlib.util
from pathlib import Path


test_path = Path("/job/review-evidence-j-8dc5a591c8dc/candidate_test.py")
spec = importlib.util.spec_from_file_location("candidate_test", test_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

case = module.TestExternalCacheEffectiveKVLength()
case.setUpClass()
try:
    print(f"lmcache_source={Path(case.lmc.__file__).resolve()}")
    print(f"flexkv_source={Path(case.flexkv.__file__).resolve()}")
    scenarios = [
        (None, 1, 1),
        (1, 4, 4),
        (1, 9, 9),
        (1, 20, 9),
        (2, 1, 1),
        (2, 8, 8),
        (2, 9, 8),
        (8, 20, 8),
    ]
    for cache_cls, is_lmcache in (
        (case.lmc.LMCRadixCache, True),
        (case.flexkv.FlexKVRadixCache, False),
    ):
        for topk, limit, expected in scenarios:
            token_ids, kv_indices = case._stored_token_ids_and_indices(
                cache_cls, is_lmcache, topk, limit
            )
            assert len(token_ids) == expected, (cache_cls, topk, limit, token_ids)
            assert len(kv_indices) == expected, (cache_cls, topk, limit, kv_indices)
            assert token_ids == list(range(1, expected + 1))
            assert kv_indices == list(range(expected))
            print(
                f"PASS backend={cache_cls.__name__} topk={topk} "
                f"limit={limit} stored={expected}"
            )
finally:
    case.tearDownClass()
