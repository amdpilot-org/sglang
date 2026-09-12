from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

import unittest

from sglang.srt.eplb.expert_distribution import _ExpertDistributionRecorderReal
from sglang.srt.utils import Withable
from sglang.test.test_utils import CustomTestCase


class _StubGatherer:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.append("reset")

    def on_forward_pass_start(self, forward_batch):
        self.calls.append(("start", forward_batch))

    def collect(self):
        self.calls.append("collect")
        return {"counts": 1}

    def on_select_experts(self, layer_idx, topk_ids):
        self.calls.append(("select", layer_idx, topk_ids))


class _StubAccumulator:
    def __init__(self):
        self.calls = []

    def append(self, forward_pass_id, gatherer_key, single_pass_data, outputs):
        self.calls.append((forward_pass_id, gatherer_key, single_pass_data, outputs))

    def get_single_pass_gatherer_key(self, debug_name):
        return "primary"


def _make_recorder(*, capturing: bool, recording: bool = True):
    recorder = _ExpertDistributionRecorderReal.__new__(_ExpertDistributionRecorderReal)
    recorder._recording = recording
    recorder._disable_all = False
    recorder._is_current_stream_capturing = lambda: capturing
    recorder._current_layer_idx = Withable()
    recorder._current_debug_name = Withable()
    recorder._single_pass_gatherers = {"primary": _StubGatherer()}
    recorder._accumulator = _StubAccumulator()
    return recorder


class TestExpertDistributionCaptureGuard(CustomTestCase):
    def test_pass_boundary_is_skipped_during_capture(self):
        recorder = _make_recorder(capturing=True)

        recorder._on_forward_pass_start(forward_batch="batch")
        recorder._on_forward_pass_end(forward_pass_id=7, outputs={})

        self.assertEqual(recorder._single_pass_gatherers["primary"].calls, [])
        self.assertEqual(recorder._accumulator.calls, [])

    def test_pass_boundary_runs_outside_capture(self):
        recorder = _make_recorder(capturing=False)
        outputs = {}

        recorder._on_forward_pass_start(forward_batch="batch")
        recorder._on_forward_pass_end(forward_pass_id=7, outputs=outputs)

        self.assertEqual(
            recorder._single_pass_gatherers["primary"].calls,
            ["reset", ("start", "batch"), "collect"],
        )
        self.assertEqual(
            recorder._accumulator.calls,
            [(7, "primary", {"counts": 1}, outputs)],
        )

    def test_pass_boundary_is_skipped_when_not_recording(self):
        recorder = _make_recorder(capturing=False, recording=False)

        recorder._on_forward_pass_start(forward_batch="batch")
        recorder._on_forward_pass_end(forward_pass_id=7, outputs={})

        self.assertEqual(recorder._single_pass_gatherers["primary"].calls, [])
        self.assertEqual(recorder._accumulator.calls, [])

    def test_layer_hook_remains_active_during_capture(self):
        recorder = _make_recorder(capturing=True, recording=False)

        with recorder.with_current_layer(3):
            recorder.on_select_experts("topk")

        self.assertEqual(
            recorder._single_pass_gatherers["primary"].calls,
            [("select", 3, "topk")],
        )


if __name__ == "__main__":
    unittest.main()
