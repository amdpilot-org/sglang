"""Regression coverage for request-scoped abort on fully-NaN logits."""

import unittest
from http import HTTPStatus
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.speculative.eagle_utils import _reduce_verify_nan_rows
from sglang.srt.utils.async_probe import detect_full_nan_rows, sanitize_nan_logits
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(rid, *, finished=False, retracted=False, middle_chunks=0):
    return SimpleNamespace(
        rid=rid,
        finished=lambda: finished,
        is_retracted=retracted,
        inflight_middle_chunks=middle_chunks,
        skip_radix_cache_insert=False,
        to_finish=None,
        output_ids=[],
    )


def _mark(reqs, mask, *, speculative=False):
    batch = SimpleNamespace(
        reqs=reqs,
        spec_algorithm=SimpleNamespace(is_none=lambda: not speculative),
    )
    output = LogitsProcessorOutput(next_token_logits=None, full_nan_rows=mask)
    return SchedulerBatchResultProcessor._mark_full_nan_logits_reqs(None, batch, output)


class TestAbortOnNanLogits(CustomTestCase):
    def test_sanitized_full_nan_row_is_uniform(self):
        logits = torch.full((1, 7), float("nan"))
        with envs.SGLANG_SANITIZE_NAN_LOGITS.override(True):
            sanitize_nan_logits(logits)
        probs = torch.softmax(logits, dim=-1)
        torch.testing.assert_close(probs, torch.full_like(probs, 1 / 7))

    def test_detection_is_opt_in_and_full_row_only(self):
        logits = torch.tensor(
            [[float("nan"), float("nan")], [float("nan"), 1.0], [0.0, 1.0]]
        )
        with envs.SGLANG_ABORT_ON_NAN_LOGITS.override(False):
            self.assertIsNone(detect_full_nan_rows(logits))
        with envs.SGLANG_ABORT_ON_NAN_LOGITS.override(True):
            self.assertEqual(
                detect_full_nan_rows(logits).tolist(), [True, False, False]
            )

    def test_abort_mode_sanitizes_before_sampling_without_sanitize_flag(self):
        logits = torch.full((1, 7), float("nan"))
        with (
            envs.SGLANG_ABORT_ON_NAN_LOGITS.override(True),
            envs.SGLANG_SANITIZE_NAN_LOGITS.override(False),
        ):
            self.assertEqual(detect_full_nan_rows(logits).tolist(), [True])
            sanitize_nan_logits(logits)
        self.assertTrue(torch.isfinite(logits).all())

    def test_abort_mode_sanitizes_full_fp16_row_to_finite_softmax(self):
        logits = torch.full((1, 7), float("nan"), dtype=torch.float16)
        with (
            envs.SGLANG_ABORT_ON_NAN_LOGITS.override(True),
            envs.SGLANG_SANITIZE_NAN_LOGITS.override(False),
        ):
            sanitize_nan_logits(logits)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(torch.softmax(logits, dim=-1)).all())

    def test_marks_only_affected_request_without_committing_token(self):
        reqs = [_req("healthy"), _req("nan")]
        aborted = _mark(reqs, torch.tensor([False, True]))
        self.assertEqual(aborted, {1})
        self.assertIsNone(reqs[0].to_finish)
        self.assertIsInstance(reqs[1].to_finish, FINISH_ABORT)
        self.assertEqual(reqs[1].to_finish.status_code, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(reqs[1].to_finish.err_type, "InternalServerError")
        self.assertTrue(reqs[1].skip_radix_cache_insert)
        self.assertEqual(reqs[1].output_ids, [])

    def test_finished_and_retracted_rows_are_ignored(self):
        reqs = [
            _req("done", finished=True),
            _req("retracted", retracted=True),
            _req("middle-chunk", middle_chunks=1),
        ]
        self.assertEqual(_mark(reqs, torch.tensor([True, True, True])), set())
        self.assertTrue(all(req.to_finish is None for req in reqs))

    def test_rejects_mismatched_mask(self):
        with self.assertRaisesRegex(AssertionError, "expected one row per request"):
            _mark([_req("a")], torch.tensor([True, False]))

    def test_marks_speculative_request(self):
        req = _req("spec-nan")
        self.assertEqual(_mark([req], torch.tensor([True]), speculative=True), {0})
        self.assertIsInstance(req.to_finish, FINISH_ABORT)

    def test_speculative_verify_rows_reduce_per_request(self):
        mask = torch.tensor([False, True, False, False, False, False])
        self.assertEqual(
            _reduce_verify_nan_rows(mask, batch_size=2, draft_token_num=3).tolist(),
            [True, False],
        )


if __name__ == "__main__":
    unittest.main()
