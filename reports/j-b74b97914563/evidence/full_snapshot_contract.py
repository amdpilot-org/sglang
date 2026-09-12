import os
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer


class Spec:
    def is_none(self):
        return True


class PoolStats:
    def get_kv_token_stats(self):
        return (0, 0.0)


def batch(*rids):
    return NS(reqs=[NS(rid=rid) for rid in rids])


slots = [batch("r0"), batch("r1", "r2"), batch("r3")]
kwargs = dict(
    disaggregation_mode=DisaggregationMode.NULL,
    ps=NS(dp_rank=0),
    server_args=NS(),
    max_total_num_tokens=100,
    max_running_requests=100,
    pool_stats_observer=NS(get_pool_stats=lambda: PoolStats()),
    tp_worker=NS(model_runner=NS(weight_load_mem_usage=0), graph_memory_usage={}),
    token_to_kv_pool_allocator=NS(get_kvcache=lambda: NS(mem_usage=0)),
    spec_algorithm=Spec(),
    get_waiting_queue=lambda: [],
    waiting_queue_prefix_matched=lambda: True,
    get_recent_cache_hit_rate=lambda: 0.0,
    get_stats=lambda: NS(
        spec_accept_rate=0.0,
        lora_pool_slots_used=0,
        lora_pool_slots_total=0,
        lora_pool_utilization=0.0,
        kv_transfer_speed_gb_s=0.0,
        kv_transfer_latency_ms=0.0,
        num_grammar_queue_reqs=0,
        num_paused_reqs=0,
        num_retracted_reqs=0,
        gen_throughput=0.0,
        cache_hit_rate=0.0,
        utilization=0.0,
    ),
    get_chunked_req=lambda: None,
    get_disagg_prefill_bootstrap_queue=lambda: NS(queue=[]),
    get_disagg_prefill_inflight_queue=lambda: [],
    get_disagg_decode_prealloc_queue=lambda: NS(queue=[], retracted_queue=[]),
    get_disagg_decode_transfer_queue=lambda: NS(queue=[]),
    get_spec_total_num_accept_tokens=lambda: 0,
    get_spec_total_num_forward_ct=lambda: 0,
    get_total_prefill_uncached_tokens=lambda: 0,
    get_total_prefill_busy_us=lambda: 0,
    get_decode_moment_totals=lambda: (0, 0, 0),
)
if "get_running_batches" in SchedulerLoadInquirer.__dataclass_fields__:
    kwargs["get_running_batches"] = lambda: slots
else:
    kwargs["get_running_batch"] = lambda: slots[0]

with patch(
    "sglang.srt.managers.scheduler_components.load_inquirer.get_lora",
    return_value=NS(enable_lora=False),
):
    observed = SchedulerLoadInquirer(**kwargs).get_loads().num_running_reqs
expected = int(os.environ["EXPECTED_RUNNING"])
print(f"observed={observed} expected={expected} unique_across_slots=4")
assert observed == expected
