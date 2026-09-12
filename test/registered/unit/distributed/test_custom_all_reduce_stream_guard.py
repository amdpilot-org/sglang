from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
from sglang.srt.distributed.device_communicators import (
    custom_all_reduce,
    custom_all_reduce_v2,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    SingleStreamGuard,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _guard_with_streams(*streams):
    guard = SingleStreamGuard(torch.device("cuda:0"))
    guard._raw_stream = MagicMock(
        side_effect=[stream.cuda_stream for stream in streams]
    )
    return guard


@patch("torch.cuda.is_current_stream_capturing", return_value=False)
@patch("torch.cuda.Event")
@patch("torch.cuda.current_stream")
def test_stream_change_serializes_after_previous_launch(current_stream, event_cls, _):
    first = MagicMock(cuda_stream=11)
    second = MagicMock(cuda_stream=22)
    event = event_cls.return_value
    current_stream.side_effect = [first, second]
    guard = _guard_with_streams(first, second)

    guard.maybe_serialize()
    guard.maybe_serialize()

    event_cls.assert_called_once_with(enable_timing=False)
    event.record.assert_called_once_with(first)
    second.wait_event.assert_called_once_with(event)


@patch("torch.cuda.is_current_stream_capturing", return_value=False)
@patch("torch.cuda.Event")
@patch("torch.cuda.current_stream")
def test_same_stream_is_fast_path(current_stream, event_cls, _):
    stream = MagicMock(cuda_stream=11)
    current_stream.return_value = stream
    guard = _guard_with_streams(stream, stream)

    guard.maybe_serialize()
    guard.maybe_serialize()

    current_stream.assert_called_once_with(0)
    event_cls.assert_not_called()


@patch("torch.cuda.is_current_stream_capturing", return_value=True)
@patch("torch.cuda.Event")
@patch("torch.cuda.current_stream")
def test_capture_does_not_add_external_dependency(current_stream, event_cls, _):
    stream = MagicMock(cuda_stream=11)
    guard = _guard_with_streams(stream)

    guard.maybe_serialize()

    current_stream.assert_not_called()
    event_cls.assert_not_called()
    assert guard._last_stream is None
    assert guard._last_raw_stream is None


def test_v2_communicator_guards_before_launch():
    comm = object.__new__(custom_all_reduce_v2.CustomAllReduceV2)
    comm.disabled = True
    comm.stream_guard = MagicMock()
    comm.override_algo = AllReduceAlgo.ONE_SHOT_PUSH
    comm._can_use_graph = MagicMock(return_value=False)
    comm.obj = MagicMock()
    inp = torch.ones(4)

    with patch.object(custom_all_reduce_v2, "custom_all_reduce") as launch:
        comm.custom_all_reduce(inp)

    comm.stream_guard.maybe_serialize.assert_called_once_with()
    launch.assert_called_once()


def test_legacy_communicator_guards_before_launch():
    comm = object.__new__(custom_all_reduce.CustomAllreduce)
    comm.disabled = True
    comm.stream_guard = MagicMock()
    comm.use_amd_deterministic_impl = False
    comm._ptr = MagicMock()
    comm.buffer = torch.empty(16, dtype=torch.uint8)
    inp = torch.ones(4)

    with patch.object(custom_all_reduce.ops, "all_reduce_unreg") as launch:
        comm._all_reduce_impl(inp, registered=False)

    comm.stream_guard.maybe_serialize.assert_called_once_with()
    launch.assert_called_once()
