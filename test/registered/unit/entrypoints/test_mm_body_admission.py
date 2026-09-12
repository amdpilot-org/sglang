import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.multimodal_preprocessing_admission import (
    MultimodalPreprocessingAdmission,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMultimodalBodyAdmission(CustomTestCase):
    def test_busy_requests_are_rejected_before_body_receive(self):
        async def drive():
            admission = MultimodalPreprocessingAdmission(max_inflight_items=1)
            old_state = http_server._global_state
            http_server._global_state = SimpleNamespace(
                tokenizer_manager=SimpleNamespace(mm_preprocessing_admission=admission)
            )
            first_body_entered = asyncio.Event()
            release_first = asyncio.Event()
            receive_counts = [0] * 32
            statuses = [None] * 32

            async def downstream(scope, receive, send):
                await receive()
                first_body_entered.set()
                await release_first.wait()
                await send({"type": "http.response.start", "status": 200})
                await send({"type": "http.response.body", "body": b""})

            middleware = http_server.MultimodalBodyAdmissionMiddleware(downstream)

            async def request(index):
                async def receive():
                    receive_counts[index] += 1
                    return {
                        "type": "http.request",
                        "body": b"x" * (4 * 1024 * 1024),
                        "more_body": False,
                    }

                async def send(message):
                    if message["type"] == "http.response.start":
                        statuses[index] = message["status"]

                await middleware(
                    {"type": "http", "method": "POST", "path": "/generate"},
                    receive,
                    send,
                )

            try:
                first = asyncio.create_task(request(0))
                await first_body_entered.wait()
                rejected = [asyncio.create_task(request(i)) for i in range(1, 32)]
                await asyncio.gather(*rejected)

                self.assertEqual(receive_counts[0], 1)
                self.assertEqual(sum(receive_counts[1:]), 0)
                self.assertEqual(statuses[1:], [503] * 31)
                self.assertEqual(admission.inflight_items, 1)

                release_first.set()
                await first
                self.assertEqual(statuses[0], 200)
                self.assertEqual(admission.inflight_items, 0)
            finally:
                http_server._global_state = old_state

        asyncio.run(drive())


if __name__ == "__main__":
    unittest.main(verbosity=2)
