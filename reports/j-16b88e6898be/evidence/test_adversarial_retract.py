from array import array
from collections import deque
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import PauseGenerationReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


def test_idle_retract_invalidates_finished_old_weight_prefixes():
    allocator = MagicMock()
    allocator.device = torch.device("cpu")
    cache = RadixCache.create_simulated(mock_allocator=allocator)
    key = RadixKey(array("q", [1, 2, 3, 4]))
    cache.insert(InsertParams(key=key, value=torch.tensor([10, 11, 12, 13])))
    assert len(cache.match_prefix(MatchPrefixParams(key=key)).device_indices) == 4

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._engine_paused = False
    scheduler.enable_overlap = False
    scheduler.last_batch = None
    scheduler.cur_batch_for_debug = None
    scheduler.chunked_req = None
    scheduler.running_batch = MagicMock()
    scheduler.running_batch.reqs = []
    scheduler.running_batch.batch_is_full = False
    scheduler.tree_cache = cache
    scheduler.req_to_token_pool = MagicMock()
    scheduler.token_to_kv_pool_allocator = MagicMock()
    scheduler.hisparse_coordinator = None
    scheduler.result_queue = deque()
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.waiting_queue = []
    scheduler.metrics_reporter = MagicMock()
    scheduler.metrics_reporter.current_scheduler_metrics_enabled = False
    scheduler.kv_events_publisher = MagicMock()

    scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

    rematch = cache.match_prefix(MatchPrefixParams(key=key))
    assert len(rematch.device_indices) == 0, (
        "retract pause left an old, finished-request prefix matchable when "
        "there were no active requests"
    )
