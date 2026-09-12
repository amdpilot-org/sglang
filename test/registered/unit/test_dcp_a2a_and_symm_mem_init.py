from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.model_executor.model_runner import ModelRunner


class _FakePyNccl:
    def __init__(self, disabled=True):
        self.disabled = disabled
        self.states_during_call = []

    @contextmanager
    def change_state(self, enable=None):
        old_disabled = self.disabled
        self.disabled = not enable
        try:
            yield
        finally:
            self.disabled = old_disabled

    def all_to_all_single(self, output, input_):
        self.states_during_call.append(self.disabled)


def test_all_to_all_single_temporarily_enables_pynccl():
    group = object.__new__(GroupCoordinator)
    group.pynccl_comm = _FakePyNccl(disabled=True)

    group._all_to_all_single(object(), object())

    assert group.pynccl_comm.states_during_call == [False]
    assert group.pynccl_comm.disabled is True


def test_all_to_all_single_preserves_enabled_pynccl_state():
    group = object.__new__(GroupCoordinator)
    group.pynccl_comm = _FakePyNccl(disabled=False)

    group._all_to_all_single(object(), object())

    assert group.pynccl_comm.states_during_call == [False]
    assert group.pynccl_comm.disabled is False


def test_all_to_all_single_falls_back_without_pynccl():
    group = object.__new__(GroupCoordinator)
    group.pynccl_comm = None
    group.device_group = object()
    output, input_ = object(), object()

    with patch.object(torch.distributed, "all_to_all_single") as fallback:
        group._all_to_all_single(output, input_)

    fallback.assert_called_once_with(output, input_, group=group.device_group)


def test_symmetric_pool_is_reserved_before_kv_budget_is_profiled():
    calls = []
    result = SimpleNamespace(
        max_total_num_tokens=128,
        max_running_requests=16,
        req_to_token_pool=object(),
        token_to_kv_pool=object(),
        token_to_kv_pool_allocator=object(),
        memory_pool_config=object(),
        unified_memory_pool=None,
    )
    runner = object.__new__(ModelRunner)
    runner.memory_pool_config = None
    runner.is_draft_worker = False
    runner.device = "cuda"
    runner.forward_stream = object()
    runner.pre_model_load_memory = 1.0
    runner.is_hybrid_swa = False
    runner.init_kv_cache_configurator = lambda: setattr(
        runner,
        "kv_cache_configurator",
        SimpleNamespace(configure=lambda **kwargs: calls.append("profile") or result),
    )
    runner._init_post_memory_pool_components = lambda: None

    exec_config = SimpleNamespace(comm=SimpleNamespace(enable_symm_mem=True))
    with (
        patch(
            "sglang.srt.model_executor.model_runner.get_exec",
            return_value=exec_config,
        ),
        patch(
            "sglang.srt.model_executor.model_runner.prealloc_symmetric_memory_pool",
            side_effect=lambda **kwargs: calls.append("reserve"),
        ) as reserve,
    ):
        runner.alloc_memory_pool()

    assert calls == ["reserve", "profile"]
    reserve.assert_called_once_with(
        is_draft_worker=False,
        enable_symm_mem=True,
        device="cuda",
        forward_stream=runner.forward_stream,
    )
