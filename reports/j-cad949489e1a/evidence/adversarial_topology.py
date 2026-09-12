"""Independent review matrix for scheduler metric exporter selection."""

import importlib.util
from pathlib import Path


fixture_path = Path("/tmp/amdpilot-repo-j-cad949489e1a/candidate_regression.py")
spec = importlib.util.spec_from_file_location("candidate_regression", fixture_path)
fixture = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fixture)


for cp_size in (1, 2, 3, 8):
    for attn_tp_size in (1, 2, 4):
        contexts = [
            fixture._init_context(
                attn_tp_rank=attn_tp_rank,
                attn_cp_rank=attn_cp_rank,
                attn_cp_size=cp_size,
            )
            for attn_cp_rank in range(cp_size)
            for attn_tp_rank in range(attn_tp_size)
        ]
        enabled = sum(c.current_scheduler_metrics_enabled for c in contexts)
        leaders = sum(c.is_stats_logging_rank for c in contexts)
        assert (leaders, enabled) == (1, 1), (
            cp_size,
            attn_tp_size,
            leaders,
            enabled,
        )

        override_contexts = [
            fixture._init_context(
                attn_tp_rank=attn_tp_rank,
                attn_cp_rank=attn_cp_rank,
                attn_cp_size=cp_size,
                enable_metrics_for_all_schedulers=True,
            )
            for attn_cp_rank in range(cp_size)
            for attn_tp_rank in range(attn_tp_size)
        ]
        assert sum(c.is_stats_logging_rank for c in override_contexts) == 1
        assert sum(c.current_scheduler_metrics_enabled for c in override_contexts) == len(
            override_contexts
        )

print("PASS: 12 default and 12 override CP/attention-TP topology combinations")
