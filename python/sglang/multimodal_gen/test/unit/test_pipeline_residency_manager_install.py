from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)


class _Pipeline(ComposedPipelineBase):
    def create_pipeline_stages(self, server_args):
        return None


class _RecordingExecutor:
    def __init__(self):
        self.component_residency_manager = None
        self.seen_at_execute = None

    def execute_group_with_profiling(self, stages, batches, server_args):
        self.seen_at_execute = self.component_residency_manager


def test_grouped_forward_installs_residency_manager_before_execute():
    """Regression for issue 34000 Bug 2, already fixed on the tested base."""
    pipeline = object.__new__(_Pipeline)
    pipeline.executor = _RecordingExecutor()
    pipeline._stage_name_mapping = {}
    pipeline._stages = []
    pipeline.is_lora_set = lambda: False
    pipeline.is_lora_effective = lambda: False
    request = SimpleNamespace(is_warmup=False, suppress_logs=True)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            task_type=SimpleNamespace(is_action_gen=lambda: False)
        )
    )
    manager = object()

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base."
        "get_global_component_residency_manager",
        return_value=manager,
    ):
        pipeline.forward_batch([request, request], server_args)

    assert pipeline.executor.seen_at_execute is manager
