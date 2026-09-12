from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)


class DecodeMode:
    def is_decode(self):
        return True

    def is_extend(self):
        return False

    def is_mixed(self):
        return False


def batch(reqs, seq_lens_cpu):
    return SimpleNamespace(
        forward_mode=DecodeMode(),
        reqs=reqs,
        decoding_reqs=[],
        prefill_stats=None,
        seq_lens_cpu=seq_lens_cpu,
    )


reporter = object.__new__(SchedulerMetricsReporter)

# Original contract: absent CPU mirror falls back to request sequence lengths.
metrics = reporter._build_scheduled_request_metrics(
    batch([SimpleNamespace(seqlen=1), SimpleNamespace(seqlen=9), SimpleNamespace(seqlen=14)], None)
)
assert (metrics.num_decode_requests, metrics.sum_decode_kv_tokens) == (3, 24)
assert metrics.var_decode_kv_tokens == 28.666666666666668

# A real CPU tensor remains authoritative even when request values disagree.
metrics = reporter._build_scheduled_request_metrics(
    batch([SimpleNamespace(seqlen=1000), SimpleNamespace(seqlen=2000)], torch.tensor([4, 10], dtype=torch.int64))
)
assert (metrics.num_decode_requests, metrics.sum_decode_kv_tokens) == (2, 14)
assert metrics.var_decode_kv_tokens == 9.0

# Degenerate batch should not reintroduce an observability-path crash.
metrics = reporter._build_scheduled_request_metrics(batch([], None))
assert (metrics.num_decode_requests, metrics.sum_decode_kv_tokens, metrics.var_decode_kv_tokens) == (0, 0, 0.0)

print("independent adversarial cases passed")
