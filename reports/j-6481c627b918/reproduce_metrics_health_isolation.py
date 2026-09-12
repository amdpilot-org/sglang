"""Deterministically exercise the /metrics event-loop starvation mechanism."""

import asyncio
import os
import tempfile
import time
from unittest.mock import patch

import httpx
from starlette.applications import Starlette

from sglang.srt.utils.common import add_prometheus_middleware


class SlowMetricsProcess:
    returncode = 0

    async def communicate(self, input=None):
        await asyncio.sleep(0.35)
        return (
            b'{"status": 200, "headers": [["Content-Type", "text/plain"]]}'
            b"\nsglang_test_metric 1\n",
            b"",
        )

    def kill(self):
        pass


async def main():
    with tempfile.TemporaryDirectory() as multiproc_dir, patch.dict(
        os.environ, {"PROMETHEUS_MULTIPROC_DIR": multiproc_dir}
    ), patch.object(
        asyncio, "create_subprocess_exec", return_value=SlowMetricsProcess()
    ):
        app = Starlette()
        add_prometheus_middleware(app)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            loop = asyncio.get_running_loop()
            timer = loop.create_future()
            expected = loop.time() + 0.02
            loop.call_later(0.02, lambda: timer.set_result(loop.time() - expected))

            first_scrape = asyncio.create_task(client.get("/metrics"))
            await asyncio.sleep(0.02)
            concurrent_scrape = await client.get("/metrics")
            event_loop_lag = await timer
            first_response = await first_scrape

            print(f"first_metrics_status={first_response.status_code}")
            print(f"concurrent_metrics_status={concurrent_scrape.status_code}")
            print(f"event_loop_lag_seconds={event_loop_lag:.3f}")
            assert first_response.status_code == 200
            assert concurrent_scrape.status_code == 503
            assert event_loop_lag < 0.1


if __name__ == "__main__":
    asyncio.run(main())
