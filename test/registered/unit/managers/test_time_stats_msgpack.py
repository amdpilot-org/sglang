import pickle
from array import array
from unittest.mock import patch

import msgspec

import sglang.srt.observability.req_time_stats as stats_module
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import io_struct
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    TokenizedGenerateReqInput,
    sock_recv,
    sock_send,
)
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    EncoderReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class LoopbackSocket:
    def send(self, data, flags=0):
        self.data = data

    def recv(self, flags=0):
        return self.data

    def send_pyobj(self, obj, flags=0, protocol=None):
        self.data = pickle.dumps(obj, protocol=protocol)

    def recv_pyobj(self, flags=0):
        return pickle.loads(self.data)


def round_trip(value, use_pickle):
    socket = LoopbackSocket()
    with patch.object(io_struct, "_USE_PICKLE_IPC", use_pickle):
        sock_send(socket, value)
        return sock_recv(socket)


def test_all_time_stats_are_msgspec_structs():
    for cls in (
        APIServerReqTimeStats,
        DPControllerReqTimeStats,
        SchedulerReqTimeStats,
        EncoderReqTimeStats,
    ):
        assert issubclass(cls, msgspec.Struct)


def test_request_time_stats_round_trip_in_both_transport_modes():
    for use_pickle in (False, True):
        req = TokenizedGenerateReqInput(
            input_text="hello",
            input_ids=array("i", [1]),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(),
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            time_stats=DPControllerReqTimeStats(
                disagg_mode=DisaggregationMode.PREFILL
            ).to_ipc(),
        )
        decoded = round_trip(req, use_pickle)
        assert isinstance(decoded.time_stats, DPControllerReqTimeStats)
        assert decoded.time_stats.disagg_mode == DisaggregationMode.PREFILL
        assert decoded.time_stats.metrics_collector is None


def test_scheduler_time_stats_calibrate_and_preserve_diagnostic_timing():
    with patch.object(stats_module, "global_diff_realtime_monotonic", 100.0):
        snapshot = SchedulerReqTimeStats(
            has_timing_data=True,
            wait_queue_entry_time=10.0,
            forward_entry_time=12.0,
            prefill_finished_time=14.0,
        ).to_ipc()
        output = BatchEmbeddingOutput(
            rids=["rid"],
            finished_reasons=[None],
            embeddings=[1.0],
            prompt_tokens=[1],
            cached_tokens=[0],
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            time_stats=[snapshot],
        )
        sockets = {}
        for use_pickle in (False, True):
            socket = LoopbackSocket()
            with patch.object(io_struct, "_USE_PICKLE_IPC", use_pickle):
                sock_send(socket, output)
            sockets[use_pickle] = socket

    for use_pickle, socket in sockets.items():
        with (
            patch.object(stats_module, "global_diff_realtime_monotonic", 90.0),
            patch.object(io_struct, "_USE_PICKLE_IPC", use_pickle),
        ):
            decoded = sock_recv(socket)
        result = decoded.time_stats[0]
        assert result.wait_queue_entry_time == 20.0
        assert result.forward_entry_time == 22.0
        assert result.prefill_finished_time == 24.0
        assert result.metrics_collector is None
