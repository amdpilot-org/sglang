import concurrent.futures
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeTransferBatching(unittest.TestCase):
    @staticmethod
    def _make_manager(
        side_effect=None, enable_custom_mem_pool=False, max_batch_indices=0
    ):
        engine = MagicMock()
        if side_effect is None:
            engine.batch_transfer_sync.return_value = 0
        else:
            engine.batch_transfer_sync.side_effect = side_effect
        manager = SimpleNamespace(
            engine=engine,
            is_mla_backend=True,
            is_hybrid_mla_backend=False,
            pp_size=1,
            enable_custom_mem_pool=enable_custom_mem_pool,
            enable_deferred_decode_kv_release=False,
            max_transfer_batch_indices=max_batch_indices,
            get_mla_kv_ptrs_with_pp=MagicMock(
                return_value=([1000, 2000], [5000, 6000], 2)
            ),
        )
        manager._transfer_data = lambda session, blocks: (
            MooncakeKVManager._transfer_data(manager, session, blocks)
        )
        manager._await_transfer_futures = lambda futures: (
            MooncakeKVManager._await_transfer_futures(manager, futures)
        )
        return manager

    @staticmethod
    def _send(
        manager,
        dst_device_data_indices=None,
        dst_device_data_ptrs=None,
    ):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return MooncakeKVManager._send_kvcache_generic(
                manager,
                mooncake_session_id="session",
                src_data_ptrs=[1000, 2000],
                dst_data_ptrs=[5000, 6000],
                item_lens=[10, 20],
                prefill_data_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
                dst_data_indices=np.array([10, 11, 12, 13, 14], dtype=np.int32),
                executor=executor,
                dst_device_data_indices=dst_device_data_indices,
                dst_device_data_ptrs=dst_device_data_ptrs,
            )

    def test_slices_index_arrays_before_forming_transfer_ranges(self):
        manager = self._make_manager(max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, 0)
        self.assertEqual(
            manager.engine.batch_transfer_sync.call_args_list,
            [
                call("session", [1000, 2000], [5100, 6200], [20, 40]),
                call("session", [1020, 2040], [5120, 6240], [20, 40]),
                call("session", [1040, 2080], [5140, 6280], [10, 20]),
            ],
        )

    def test_preserves_legacy_single_batch_path_for_short_transfers(self):
        for max_batch_indices in (0, 5, 6):
            with self.subTest(max_batch_indices=max_batch_indices):
                manager = self._make_manager(max_batch_indices=max_batch_indices)
                ret = self._send(manager)

                self.assertEqual(ret, 0)
                manager.engine.batch_transfer_sync.assert_called_once_with(
                    "session",
                    [1000, 2000],
                    [5100, 6200],
                    [50, 100],
                )

    def test_stops_after_first_failed_index_batch(self):
        manager = self._make_manager(side_effect=[0, -1], max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, -1)
        self.assertEqual(manager.engine.batch_transfer_sync.call_count, 2)

    def test_uses_device_page_indices_in_batched_path(self):
        manager = self._make_manager(max_batch_indices=2)
        ret = self._send(
            manager,
            dst_device_data_indices=np.array([20, 21, 22, 23, 24], dtype=np.int32),
            dst_device_data_ptrs={6000},
        )

        self.assertEqual(ret, 0)
        self.assertEqual(
            manager.engine.batch_transfer_sync.call_args_list,
            [
                call("session", [1000, 2000], [5100, 6400], [20, 40]),
                call("session", [1020, 2040], [5120, 6440], [20, 40]),
                call("session", [1040, 2080], [5140, 6480], [10, 20]),
            ],
        )

    def test_preserves_one_transfer_per_layer_for_custom_mem_pool(self):
        manager = self._make_manager(enable_custom_mem_pool=True, max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, 0)
        self.assertEqual(manager.engine.batch_transfer_sync.call_count, 2)
        manager.engine.batch_transfer_sync.assert_has_calls(
            [
                call("session", [1000], [5100], [50]),
                call("session", [2000], [6200], [100]),
            ],
            any_order=True,
        )


class TestMooncakeFailedSessionRecovery(unittest.TestCase):
    @staticmethod
    def _make_manager(failed_sessions, send_probe):
        return SimpleNamespace(
            engine=SimpleNamespace(send_probe=MagicMock(side_effect=send_probe)),
            failed_sessions=set(failed_sessions),
            session_failures={session_id: 1 for session_id in failed_sessions},
            session_generations={session_id: 0 for session_id in failed_sessions},
            session_lock=threading.Lock(),
        )

    def test_successful_probe_unblacklists_session(self):
        manager = self._make_manager(["decode:1234"], lambda _: 0)

        MooncakeKVManager._run_one_probe_pass(manager)

        self.assertNotIn("decode:1234", manager.failed_sessions)
        self.assertNotIn("decode:1234", manager.session_failures)

    def test_registration_starts_a_new_session_generation(self):
        manager = self._make_manager(["decode:1234"], lambda _: 0)

        MooncakeKVManager._mark_session_registered(manager, "decode:1234")

        self.assertEqual(manager.session_generations["decode:1234"], 1)
        self.assertNotIn("decode:1234", manager.failed_sessions)
        self.assertNotIn("decode:1234", manager.session_failures)

    def test_probe_is_enabled_by_default(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_ENABLE_FAILED_SESSION_PROBE", None)
            self.assertTrue(envs.SGLANG_ENABLE_FAILED_SESSION_PROBE.get())

    def test_probe_can_be_disabled_explicitly(self):
        with envs.SGLANG_ENABLE_FAILED_SESSION_PROBE.override(False):
            self.assertFalse(envs.SGLANG_ENABLE_FAILED_SESSION_PROBE.get())

    def test_stale_probe_does_not_clear_newer_failure(self):
        probe_started = threading.Event()
        finish_probe = threading.Event()

        def send_probe(_):
            probe_started.set()
            self.assertTrue(finish_probe.wait(timeout=5))
            return 0

        manager = self._make_manager(["decode:1234"], send_probe)
        probe_thread = threading.Thread(
            target=MooncakeKVManager._run_one_probe_pass, args=(manager,)
        )
        probe_thread.start()
        self.assertTrue(probe_started.wait(timeout=5))
        with manager.session_lock:
            manager.session_failures["decode:1234"] += 1
            manager.failed_sessions.add("decode:1234")
        finish_probe.set()
        probe_thread.join(timeout=5)

        self.assertFalse(probe_thread.is_alive())
        self.assertIn("decode:1234", manager.failed_sessions)
        self.assertEqual(manager.session_failures["decode:1234"], 2)

    def test_stale_probe_does_not_clear_reregistered_failure_generation(self):
        probe_started = threading.Event()
        finish_probe = threading.Event()

        def send_probe(_):
            probe_started.set()
            self.assertTrue(finish_probe.wait(timeout=5))
            return 0

        manager = self._make_manager(["decode:1234"], send_probe)
        probe_thread = threading.Thread(
            target=MooncakeKVManager._run_one_probe_pass, args=(manager,)
        )
        probe_thread.start()
        self.assertTrue(probe_started.wait(timeout=5))

        MooncakeKVManager._mark_session_registered(manager, "decode:1234")
        with manager.session_lock:
            manager.session_failures["decode:1234"] = 1
            manager.failed_sessions.add("decode:1234")

        finish_probe.set()
        probe_thread.join(timeout=5)

        self.assertFalse(probe_thread.is_alive())
        self.assertIn("decode:1234", manager.failed_sessions)
        self.assertEqual(manager.session_failures["decode:1234"], 1)

    def test_failed_probe_keeps_session_blacklisted(self):
        manager = self._make_manager(["decode:1234"], lambda _: -1)

        MooncakeKVManager._run_one_probe_pass(manager)

        self.assertIn("decode:1234", manager.failed_sessions)
        self.assertEqual(manager.session_failures["decode:1234"], 1)

    def test_probe_exception_does_not_block_other_sessions(self):
        def send_probe(session_id):
            if session_id == "bad:1234":
                raise RuntimeError("injected probe failure")
            return 0

        manager = self._make_manager(["bad:1234", "good:1234"], send_probe)

        MooncakeKVManager._run_one_probe_pass(manager)

        self.assertIn("bad:1234", manager.failed_sessions)
        self.assertNotIn("good:1234", manager.failed_sessions)

    def test_transfer_engine_forwards_probe_to_mooncake(self):
        inner_engine = MagicMock()
        inner_engine.send_probe.return_value = 0
        engine = MooncakeTransferEngine.__new__(MooncakeTransferEngine)
        engine.engine = inner_engine

        self.assertEqual(engine.send_probe("decode:1234"), 0)
        inner_engine.send_probe.assert_called_once_with("decode:1234")


if __name__ == "__main__":
    unittest.main()
