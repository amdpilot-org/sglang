from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Collector:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _init_context(
    *,
    attn_tp_rank=0,
    attn_cp_rank=0,
    attn_cp_size=1,
    enable_metrics=True,
    enable_metrics_for_all_schedulers=False,
    enable_kv_cache_events=False,
):
    ps = ParallelState.trivial(
        tp_rank=attn_cp_rank,
        tp_size=attn_cp_size,
        attn_tp_rank=attn_tp_rank,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
    )
    observability = SimpleNamespace(
        enable_metrics=enable_metrics,
        enable_metrics_for_all_schedulers=enable_metrics_for_all_schedulers,
        kv_events_config=object() if enable_kv_cache_events else None,
        extra_metric_labels=None,
    )

    with (
        patch(
            "sglang.srt.observability.metrics_collector.get_observability",
            return_value=observability,
        ),
        patch(
            "sglang.srt.observability.metrics_collector.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode="null"),
        ),
        patch(
            "sglang.srt.observability.metrics_collector.get_serving",
            return_value=SimpleNamespace(
                served_model_name="test-model", enable_streaming_session=False
            ),
        ),
        patch(
            "sglang.srt.observability.metrics_collector.DisaggregationMode.to_engine_type",
            return_value="unified",
        ),
        patch(
            "sglang.srt.observability.metrics_collector.resolve_collector_class",
            return_value=_Collector,
        ),
    ):
        return SchedulerMetricsCollector.init_new(
            server_args=SimpleNamespace(),
            ps=ps,
            tp_rank=ps.tp_rank,
            pp_rank=ps.pp_rank,
            dp_rank=None,
            enable_priority_scheduling=False,
            enable_lora=False,
            enable_hierarchical_cache=False,
        )


def test_cp8_exports_one_copy_of_request_metrics():
    contexts = [_init_context(attn_cp_rank=rank, attn_cp_size=8) for rank in range(8)]

    assert sum(context.is_stats_logging_rank for context in contexts) == 1
    assert sum(context.current_scheduler_metrics_enabled for context in contexts) == 1


@pytest.mark.parametrize(
    ("attn_tp_rank", "attn_cp_rank", "expected"),
    [
        (0, 0, True),
        (0, 1, False),
        (1, 0, False),
        (1, 1, False),
    ],
)
def test_stats_logging_requires_leading_attention_rank(
    attn_tp_rank, attn_cp_rank, expected
):
    context = _init_context(
        attn_tp_rank=attn_tp_rank,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=2,
    )

    assert context.is_stats_logging_rank is expected
    assert context.current_scheduler_metrics_enabled is expected


def test_single_rank_metrics_remain_enabled():
    context = _init_context()

    assert context.is_stats_logging_rank is True
    assert context.current_scheduler_metrics_enabled is True


def test_all_schedulers_override_exports_without_duplicate_human_logs():
    context = _init_context(
        attn_cp_rank=1,
        attn_cp_size=2,
        enable_metrics_for_all_schedulers=True,
    )

    assert context.is_stats_logging_rank is False
    assert context.current_scheduler_metrics_enabled is True


def test_metrics_disabled_keeps_rank_selection_but_disables_export():
    context = _init_context(enable_metrics=False)

    assert context.is_stats_logging_rank is True
    assert context.current_scheduler_metrics_enabled is False
    assert context.collector is None


@pytest.mark.parametrize("attn_cp_rank", [0, 1])
def test_kv_event_publisher_uses_same_cp_leader_rule(attn_cp_rank):
    context = _init_context(
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=2,
        enable_kv_cache_events=True,
    )

    assert context.enable_kv_cache_events is (attn_cp_rank == 0)
