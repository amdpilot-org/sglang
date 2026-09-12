import threading
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.profile_utils import _ProfilerTorch, _get_stage_from_forward_mode


class TestProfilerManagerStageStop(unittest.TestCase):
    def test_profile_v2_classifies_speculative_steps_as_decode(self):
        self.assertEqual(_get_stage_from_forward_mode(ForwardMode.TARGET_VERIFY), "decode")
        self.assertEqual(
            _get_stage_from_forward_mode(ForwardMode.DRAFT_EXTEND_V2), "decode"
        )
        self.assertEqual(_get_stage_from_forward_mode(ForwardMode.EXTEND), "prefill")
        self.assertEqual(_get_stage_from_forward_mode(ForwardMode.DECODE), "decode")

    def _stage_manager(self):
        manager = SchedulerProfilerManager.__new__(SchedulerProfilerManager)
        manager.profile_by_stage = True
        manager.profiler_prefill_ct = 0
        manager.profiler_decode_ct = 0
        manager.profiler_target_prefill_ct = 1
        manager.profiler_target_decode_ct = 1
        manager.profile_in_progress = False
        manager._start_profile = Mock(
            side_effect=lambda _stage: setattr(manager, "profile_in_progress", True)
        )
        manager._stop_profile = Mock(
            side_effect=lambda **_kwargs: setattr(manager, "profile_in_progress", False)
        )
        return manager

    @patch("sglang.srt.managers.scheduler_components.profiler_manager.envs.SGLANG_PROFILE_V2.get", return_value=False)
    def test_speculative_steps_count_as_decode(self, _profile_v2):
        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
            with self.subTest(mode=mode):
                manager = self._stage_manager()
                with patch(
                    "sglang.srt.managers.scheduler_components.profiler_manager.envs.SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS.get",
                    return_value=0,
                ):
                    manager._profile_batch_predicate(
                        SimpleNamespace(forward_mode=mode, batch_size=lambda: 1)
                    )

                self.assertEqual(manager.profiler_prefill_ct, 0)
                self.assertEqual(manager.profiler_decode_ct, 1)
                manager._start_profile.assert_called_once_with(mode)

    @patch("sglang.srt.managers.scheduler_components.profiler_manager.envs.SGLANG_PROFILE_V2.get", return_value=False)
    def test_prefill_and_decode_boundaries_are_unchanged(self, _profile_v2):
        prefill = self._stage_manager()
        prefill._profile_batch_predicate(
            SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        )
        self.assertEqual((prefill.profiler_prefill_ct, prefill.profiler_decode_ct), (1, 0))

        decode = self._stage_manager()
        with patch(
            "sglang.srt.managers.scheduler_components.profiler_manager.envs.SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS.get",
            return_value=0,
        ):
            decode._profile_batch_predicate(
                SimpleNamespace(forward_mode=ForwardMode.DECODE, batch_size=lambda: 1)
            )
        self.assertEqual((decode.profiler_prefill_ct, decode.profiler_decode_ct), (0, 1))

    @patch("sglang.srt.managers.scheduler_components.profiler_manager.envs.SGLANG_PROFILE_V2.get", return_value=False)
    def test_trace_export_does_not_block_stop(self, _profile_v2):
        export_started = threading.Event()
        allow_export = threading.Event()

        def export_trace(_path):
            export_started.set()
            self.assertTrue(allow_export.wait(timeout=5))

        profiler = Mock()
        profiler.export_chrome_trace.side_effect = export_trace
        manager = SchedulerProfilerManager.__new__(SchedulerProfilerManager)
        manager.profile_in_progress = True
        manager.torch_profiler = profiler
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        manager.torch_profiler_output_dir = Path(temp_dir.name)
        manager.profile_prefix = ""
        manager.profile_id = "test"
        manager.ps = SimpleNamespace(
            tp_rank=0,
            dp_size=1,
            dp_rank=0,
            pp_size=1,
            pp_rank=0,
            moe_ep_size=1,
            moe_ep_rank=0,
        )
        manager.dp_tp_cpu_group = object()
        manager.rpd_profiler = None
        manager.profiler_activities = []
        manager.merge_profiles = False
        manager.profiler_start_forward_ct = 1
        manager.detailed_annotations = False

        with patch("torch.distributed.barrier") as barrier:
            started = time.monotonic()
            manager._stop_profile(stage=ForwardMode.DECODE)
            elapsed = time.monotonic() - started
            self.assertTrue(export_started.wait(timeout=1))
            self.assertLess(elapsed, 0.5)
            allow_export.set()
            manager._wait_for_pending_exports()

        barrier.assert_not_called()

        profiler.stop.assert_called_once_with()
        profiler.export_chrome_trace.assert_called_once()
        self.assertFalse(manager.profile_in_progress)

    def test_profile_v2_trace_export_does_not_block_stop(self):
        export_started = threading.Event()
        allow_export = threading.Event()

        def export_trace(_path):
            export_started.set()
            self.assertTrue(allow_export.wait(timeout=5))

        profiler = Mock()
        profiler.export_chrome_trace.side_effect = export_trace
        with tempfile.TemporaryDirectory() as temp_dir:
            profile = _ProfilerTorch.__new__(_ProfilerTorch)
            profile.output_dir = temp_dir
            profile.output_prefix = ""
            profile.output_suffix = "-DECODE"
            profile.profile_id = "test-v2"
            profile.ps = SimpleNamespace(
                tp_rank=0,
                dp_size=1,
                dp_rank=0,
                pp_size=1,
                pp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
            )
            profile.torch_profiler = profiler

            with patch("torch.distributed.barrier") as barrier:
                started = time.monotonic()
                profile.stop()
                elapsed = time.monotonic() - started
                self.assertTrue(export_started.wait(timeout=1))
                self.assertLess(elapsed, 0.5)
                allow_export.set()
                profile.export_thread.join(timeout=5)

            barrier.assert_not_called()
            self.assertFalse(profile.export_thread.is_alive())
            profiler.stop.assert_called_once_with()
            profiler.export_chrome_trace.assert_called_once()


if __name__ == "__main__":
    unittest.main()
