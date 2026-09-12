import asyncio
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec

from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.rust_server.server import RustServer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPiggybackLoad(unittest.TestCase):
    def test_rust_cache_receives_watch_snapshot_without_generation(self):
        snapshot = LoadSnapshot(timestamp=20.0, dp_rank=0, num_running_reqs=0)
        native = MagicMock()
        scheduler = SimpleNamespace(
            load_snapshot_writer=SimpleNamespace(
                publish_counter=0,
                publish_interval=1,
                write=MagicMock(),
            ),
            load_inquirer=SimpleNamespace(get_loads=MagicMock(return_value=snapshot)),
            rust_server=SimpleNamespace(update_load_snapshot=native.update_load_snapshot),
        )

        result = Scheduler.publish_load_snapshot(scheduler, force=True)

        self.assertIs(result, snapshot)
        native.update_load_snapshot.assert_called_once()

    def test_rust_generation_wire_carries_load_snapshot(self):
        native = MagicMock()
        native.push_decode_result_batch.return_value = True
        server = object.__new__(RustServer)
        server.server = native
        payload = SimpleNamespace(
            rids=["request-1"],
            finished_reasons=[None],
            output_ids=[array("i", [7])],
            prompt_tokens=[3],
            load_snapshot=LoadSnapshot(
                timestamp=20.0, dp_rank=1, num_running_reqs=2
            ),
            output_token_logprobs_val=None,
            input_token_logprobs_val=None,
            output_top_logprobs_val=None,
            input_top_logprobs_val=None,
            output_token_ids_logprobs_val=None,
            input_token_ids_logprobs_val=None,
            output_hidden_states=None,
        )

        server.push_generation(payload)

        header, _ = native.push_decode_result_batch.call_args.args
        columns = msgspec.msgpack.decode(header)
        self.assertEqual(columns[-1]["dp_rank"], 1)
        self.assertEqual(columns[-1]["num_running_reqs"], 2)

    def test_output_snapshot_collection_is_best_effort(self):
        streamer = object.__new__(SchedulerOutputStreamer)
        streamer.load_snapshot_provider = MagicMock(
            side_effect=RuntimeError("snapshot unavailable")
        )

        with self.assertLogs(
            "sglang.srt.managers.scheduler_components.output_streamer",
            level="WARNING",
        ):
            self.assertIsNone(streamer._get_load_snapshot())

    def test_http_worker_rejects_out_of_order_piggyback_updates(self):
        manager = object.__new__(TokenizerManager)
        manager.piggyback_load_snapshots = {}
        newest = LoadSnapshot(timestamp=20.0, dp_rank=1, num_running_reqs=2)
        stale = LoadSnapshot(timestamp=10.0, dp_rank=1, num_running_reqs=99)

        manager._record_piggyback_load(newest)
        manager._record_piggyback_load(stale)

        self.assertIs(manager.piggyback_load_snapshots[1], newest)

    def test_external_query_merges_watch_and_piggyback_by_timestamp(self):
        watched_new = LoadSnapshot(timestamp=30.0, dp_rank=0, num_running_reqs=3)
        watched_stale = LoadSnapshot(timestamp=10.0, dp_rank=1, num_running_reqs=10)
        piggyback = LoadSnapshot(timestamp=20.0, dp_rank=1, num_running_reqs=2)
        manager = SimpleNamespace(
            auto_create_handle_loop=MagicMock(),
            elastic_worker_count=2,
            load_snapshot_reader=SimpleNamespace(
                read_all=MagicMock(return_value=[watched_new, watched_stale])
            ),
            piggyback_load_snapshots={1: piggyback},
        )

        loads = asyncio.run(TokenizerControlMixin.get_loads(manager))

        self.assertEqual(loads, [watched_new, piggyback])
        manager.load_snapshot_reader.read_all.assert_called_once_with()

    def test_external_query_keeps_watch_mode_for_idle_rank(self):
        watched = LoadSnapshot(timestamp=30.0, dp_rank=0, num_running_reqs=0)
        manager = SimpleNamespace(
            auto_create_handle_loop=MagicMock(),
            elastic_worker_count=2,
            load_snapshot_reader=SimpleNamespace(
                read_all=MagicMock(return_value=[watched])
            ),
            piggyback_load_snapshots={},
        )

        loads = asyncio.run(TokenizerControlMixin.get_loads(manager, dp_rank=0))

        self.assertEqual(loads, [watched])


if __name__ == "__main__":
    unittest.main()
