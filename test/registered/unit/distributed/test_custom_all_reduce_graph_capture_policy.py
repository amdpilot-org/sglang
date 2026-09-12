import sys
import types
from functools import partial
from unittest import mock

import pytest
import torch

from sglang.srt.distributed.device_communicators import custom_all_reduce
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("decode_backend", "enable_dp_attention", "expected"),
    [
        ("dsa", True, False),
        ("dsa", False, True),
        ("tilelang", True, True),
    ],
)
def test_registered_graph_input_policy(decode_backend, enable_dp_attention, expected):
    with (
        mock.patch.object(
            custom_all_reduce.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH,
            "get",
            return_value=False,
        ),
        mock.patch(
            "sglang.srt.runtime_context.attention_backends",
            return_value=("tilelang", decode_backend),
        ),
        mock.patch("sglang.srt.runtime_context.get_parallel") as get_parallel,
    ):
        get_parallel.return_value.enable_dp_attention = enable_dp_attention
        assert custom_all_reduce._enable_register_for_capturing() is expected


def test_memory_saver_always_uses_copy_in():
    with mock.patch.object(
        custom_all_reduce.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH,
        "get",
        return_value=True,
    ):
        assert not custom_all_reduce._enable_register_for_capturing()


@pytest.mark.parametrize("enable_register", [False, True])
def test_capture_uses_selected_input_path(enable_register):
    communicator = object.__new__(custom_all_reduce.CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator._IS_CAPTURING = True
    communicator.enable_register_for_capturing = enable_register
    communicator.should_custom_ar = mock.Mock(return_value=True)
    communicator._all_reduce_impl = mock.Mock(return_value=mock.sentinel.output)
    input_tensor = torch.empty(8)

    with mock.patch.object(
        torch.cuda, "is_current_stream_capturing", return_value=True
    ):
        output = communicator.custom_all_reduce(input_tensor)

    assert output is mock.sentinel.output
    communicator._all_reduce_impl.assert_called_once_with(
        input_tensor, registered=enable_register
    )


def test_policy_is_forwarded_to_sglang_custom_allreduce():
    with (
        mock.patch.object(custom_all_reduce, "_is_cuda", False),
        mock.patch.object(custom_all_reduce, "_is_musa", False),
        mock.patch.object(custom_all_reduce, "_is_hip", True),
        mock.patch.object(
            custom_all_reduce, "_use_amd_deterministic_impl", return_value=False
        ),
        mock.patch.object(
            custom_all_reduce, "_enable_register_for_capturing", return_value=False
        ),
        mock.patch.object(custom_all_reduce, "get_bool_env_var", return_value=False),
    ):
        factory = custom_all_reduce.dispatch_custom_allreduce(
            group=mock.sentinel.group,
            device=torch.device("cpu"),
        )

    assert isinstance(factory, partial)
    assert factory.func is custom_all_reduce.CustomAllreduce
    assert factory.keywords["enable_register_for_capturing"] is False


def test_policy_is_forwarded_to_deterministic_sglang_custom_allreduce():
    with (
        mock.patch.object(custom_all_reduce, "_is_cuda", False),
        mock.patch.object(custom_all_reduce, "_is_musa", False),
        mock.patch.object(custom_all_reduce, "_is_hip", True),
        mock.patch.object(
            custom_all_reduce, "_use_amd_deterministic_impl", return_value=True
        ),
        mock.patch.object(
            custom_all_reduce, "_enable_register_for_capturing", return_value=False
        ),
    ):
        factory = custom_all_reduce.dispatch_custom_allreduce(
            group=mock.sentinel.group,
            device=torch.device("cpu"),
        )

    assert isinstance(factory, partial)
    assert factory.func is custom_all_reduce.CustomAllreduce
    assert factory.keywords["enable_register_for_capturing"] is False


def test_policy_is_forwarded_to_aiter_custom_allreduce():
    fake_aiter_module = types.ModuleType(
        "aiter.dist.device_communicators.custom_all_reduce"
    )

    class FakeAiterCustomAllreduce:
        pass

    fake_aiter_module.CustomAllreduce = FakeAiterCustomAllreduce
    with (
        mock.patch.dict(
            sys.modules,
            {"aiter.dist.device_communicators.custom_all_reduce": fake_aiter_module},
        ),
        mock.patch.object(custom_all_reduce, "_is_cuda", False),
        mock.patch.object(custom_all_reduce, "_is_musa", False),
        mock.patch.object(custom_all_reduce, "_is_hip", True),
        mock.patch.object(
            custom_all_reduce, "_use_amd_deterministic_impl", return_value=False
        ),
        mock.patch.object(
            custom_all_reduce, "_enable_register_for_capturing", return_value=False
        ),
        mock.patch.object(custom_all_reduce, "get_bool_env_var", return_value=True),
    ):
        factory = custom_all_reduce.dispatch_custom_allreduce(
            group=mock.sentinel.group,
            device=torch.device("cpu"),
        )

    assert isinstance(factory, partial)
    assert factory.func is FakeAiterCustomAllreduce
    assert factory.keywords["enable_register_for_capturing"] is False
