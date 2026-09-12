"""Exercise whether synchronous metrics generation stalls unrelated async work."""

import asyncio
import os
import tempfile
import time
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from prometheus_client import asgi

from sglang.srt.utils.common import add_prometheus_middleware


async def main():
    original_bake_output = asgi._bake_output

    def slow_bake_output(*args, **kwargs):
        time.sleep(0.35)
        return original_bake_output(*args, **kwargs)

    with tempfile.TemporaryDirectory() as multiproc_dir, patch.dict(
        os.environ, {"PROMETHEUS_MULTIPROC_DIR": multiproc_dir}
    ), patch.object(
        asgi, "_bake_output", side_effect=slow_bake_output
    ) as bake_mock:
        app = FastAPI()

        @app.get("/health")
        async def health():
            return "OK"

        add_prometheus_middleware(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            loop = asyncio.get_running_loop()
            timer = loop.create_future()
            expected = loop.time() + 0.02
            loop.call_later(0.02, lambda: timer.set_result(loop.time() - expected))
            scrape_started = time.monotonic()
            scrape = asyncio.create_task(
                client.get("/metrics", headers={"Accept-Encoding": "identity"})
            )
            await asyncio.sleep(0.02)
            health_response = await client.get("/health")
            lag = await timer
            metrics_response = await scrape
            scrape_elapsed = time.monotonic() - scrape_started

    print(f"metrics_status={metrics_response.status_code}")
    print(f"health_status={health_response.status_code}")
    print(f"event_loop_lag_seconds={lag:.3f}")
    print(f"metrics_elapsed_seconds={scrape_elapsed:.3f}")
    print(f"bake_output_calls={bake_mock.call_count}")
    if lag >= 0.25:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
