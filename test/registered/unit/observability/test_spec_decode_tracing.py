"""Regression coverage for speculative-decoding request tracing."""

import ast
import unittest
from pathlib import Path
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

from sglang.srt.observability import req_time_stats


class _FakeDeviceTensor:
    def __init__(self, values, host_reads=None):
        self.values = values
        self.host_reads = host_reads if host_reads is not None else []

    def __sub__(self, value):
        return _FakeDeviceTensor(
            [item - value for item in self.values], self.host_reads
        )

    def tolist(self):
        self.host_reads.append(True)
        return list(self.values)


class TestSpecVerifyBatchTiming(unittest.TestCase):
    def _reqs(self, count):
        return [mock.MagicMock() for _ in range(count)]

    def test_tracing_disabled_does_not_read_device_or_touch_requests(self):
        reqs = self._reqs(2)
        accept_lens = _FakeDeviceTensor([3, 1])

        with mock.patch.object(
            req_time_stats, "get_global_tracing_enabled", return_value=False
        ):
            req_time_stats.set_spec_verify_end_time_batch(reqs, accept_lens)

        self.assertEqual(accept_lens.host_reads, [])
        for req in reqs:
            req.time_stats.set_spec_verify_end_time.assert_not_called()

    def test_tracing_enabled_records_drafts_only_counts_with_one_timestamp(self):
        reqs = self._reqs(3)
        accept_lens = _FakeDeviceTensor([3, 1, 4])

        with (
            mock.patch.object(
                req_time_stats, "get_global_tracing_enabled", return_value=True
            ),
            mock.patch.object(req_time_stats.time, "perf_counter", return_value=12.5),
        ):
            req_time_stats.set_spec_verify_end_time_batch(reqs, accept_lens)

        self.assertEqual(len(accept_lens.host_reads), 1)
        for req, expected in zip(reqs, [2, 0, 3], strict=True):
            req.time_stats.set_spec_verify_end_time.assert_called_once_with(
                12.5, num_correct_drafts=expected
            )

    def test_empty_batch_does_not_read_device_when_tracing_enabled(self):
        accept_lens = _FakeDeviceTensor([])

        with mock.patch.object(
            req_time_stats, "get_global_tracing_enabled", return_value=True
        ):
            req_time_stats.set_spec_verify_end_time_batch([], accept_lens)

        self.assertEqual(accept_lens.host_reads, [])


class TestSpecWorkerTracingCallSites(unittest.TestCase):
    _ROOT = Path(__file__).resolve().parents[4]

    def _calls_in_function(self, relative_path, function_name):
        tree = ast.parse((self._ROOT / relative_path).read_text())
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        )
        return {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Attribute, ast.Name))
        }

    def _set_time_methods_in_function(self, relative_path, function_name):
        tree = ast.parse((self._ROOT / relative_path).read_text())
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        )
        return {
            node.args[1].value
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "set_time_batch"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
        }

    def test_eagle_workers_open_request_draft_and_verify_stages(self):
        for worker in (
            "python/sglang/srt/speculative/eagle_worker_v2.py",
            "python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py",
        ):
            methods = self._set_time_methods_in_function(
                worker, "forward_batch_generation"
            )
            self.assertIn("set_spec_draft_start_time", methods, worker)
            self.assertIn("set_spec_draft_end_time", methods, worker)
            self.assertIn("set_spec_verify_start_time", methods, worker)

    def test_eagle_shared_verify_and_ngram_close_request_verify_stage(self):
        for worker, function in (
            (
                "python/sglang/srt/speculative/eagle_worker_common.py",
                "run_eagle_verify",
            ),
            (
                "python/sglang/srt/speculative/ngram_worker.py",
                "forward_batch_generation",
            ),
        ):
            calls = self._calls_in_function(worker, function)
            self.assertIn("set_spec_verify_end_time_batch", calls, worker)

        ngram_methods = self._set_time_methods_in_function(
            "python/sglang/srt/speculative/ngram_worker.py",
            "forward_batch_generation",
        )
        self.assertIn("set_spec_verify_start_time", ngram_methods)


if __name__ == "__main__":
    unittest.main()
