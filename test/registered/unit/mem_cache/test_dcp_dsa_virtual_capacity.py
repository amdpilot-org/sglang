import pytest

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.runtime_context import get_parallel


@pytest.mark.parametrize(
    "is_draft_worker, expected",
    [(False, 800), (True, None)],
)
def test_dsa_index_buffer_spans_virtual_locs_only_for_sharded_target(
    is_draft_worker, expected
):
    configurator = object.__new__(KVCacheConfigurator)
    configurator.is_draft_worker = is_draft_worker
    with get_parallel().override(attn_dcp_size=8):
        assert configurator._dsa_index_buf_size(100) == expected


def test_dsa_index_buffer_keeps_default_without_dcp():
    configurator = object.__new__(KVCacheConfigurator)
    configurator.is_draft_worker = False
    with get_parallel().override(attn_dcp_size=1):
        assert configurator._dsa_index_buf_size(100) is None
