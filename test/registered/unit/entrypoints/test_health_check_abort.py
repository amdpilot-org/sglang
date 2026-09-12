"""Unit tests for scheduler-side cancellation of health probes."""

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.entrypoints.http_server import health_generate
from sglang.srt.managers.tokenizer_manager import ServerStatus
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HTTP_SERVER = "sglang.srt.entrypoints.http_server"


class FakeTokenizerManager:
    """A tokenizer manager whose probe remains pending during a scheduler stall."""

    def __init__(self, *, responsive: bool):
        self.gracefully_exit = False
        self.server_status = ServerStatus.Up
        self.is_generation = True
        self.rid_to_state = {}
        self.aborted = []
        self._responsive = responsive

    @property
    def last_receive_tstamp(self) -> float:
        return time.time() if self._responsive else 0.0

    async def generate_request(self, obj, request):
        self.rid_to_state[obj.rid] = SimpleNamespace(obj=obj)
        await asyncio.Event().wait()
        yield {}

    def abort_request(self, rid: str = "", abort_all: bool = False):
        self.aborted.append((rid, rid in self.rid_to_state))


class TestHealthGenerateProbeCancellation(CustomTestCase):
    async def _run_health_generate(self, *, responsive: bool):
        tokenizer_manager = FakeTokenizerManager(responsive=responsive)
        global_state = SimpleNamespace(tokenizer_manager=tokenizer_manager)
        request = SimpleNamespace(url=SimpleNamespace(path="/health_generate"))
        disagg = SimpleNamespace(disaggregation_mode="null")

        with (
            patch(f"{HTTP_SERVER}._global_state", global_state),
            patch(f"{HTTP_SERVER}.HEALTH_CHECK_TIMEOUT", 0.1),
            patch(f"{HTTP_SERVER}.get_disagg", lambda: disagg),
        ):
            response = await health_generate(request)

        return tokenizer_manager, response

    def test_timed_out_probe_is_aborted_before_local_state_is_removed(self):
        tokenizer_manager, response = asyncio.run(
            self._run_health_generate(responsive=False)
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(len(tokenizer_manager.aborted), 1)
        rid, still_tracked = tokenizer_manager.aborted[0]
        self.assertTrue(rid.startswith(HEALTH_CHECK_RID_PREFIX))
        self.assertTrue(still_tracked)
        self.assertNotIn(rid, tokenizer_manager.rid_to_state)

    def test_successful_probe_is_not_aborted(self):
        tokenizer_manager, response = asyncio.run(
            self._run_health_generate(responsive=True)
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(tokenizer_manager.aborted, [])


if __name__ == "__main__":
    unittest.main()
