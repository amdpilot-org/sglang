import asyncio
import os
import subprocess
import sys
import tempfile

import httpx
from fastapi import FastAPI
from prometheus_client import CollectorRegistry, Gauge, make_asgi_app

from sglang.srt.utils.common import add_prometheus_middleware


async def request(app, headers):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.get("/metrics", headers=headers)


async def main():
    registry = CollectorRegistry()
    Gauge("duplicate_header_metric", "probe", registry=registry).set(1)
    reference = make_asgi_app(registry=registry)

    with tempfile.TemporaryDirectory() as multiproc_dir:
        env = os.environ.copy()
        env["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from prometheus_client import Gauge;"
                "Gauge('duplicate_header_metric','probe').set(1)",
            ],
            env=env,
            check=True,
        )
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
        candidate = FastAPI()
        add_prometheus_middleware(candidate)

        accept_headers = [
            ("accept", "application/json"),
            ("accept", "application/openmetrics-text"),
            ("accept-encoding", "identity"),
        ]
        encoding_headers = [
            ("accept", "text/plain"),
            ("accept-encoding", "identity;q=0"),
            ("accept-encoding", "gzip"),
        ]
        reference_accept = await request(reference, accept_headers)
        candidate_accept = await request(candidate, accept_headers)
        reference_encoding = await request(reference, encoding_headers)
        candidate_encoding = await request(candidate, encoding_headers)

    observations = {
        "reference_accept_type": reference_accept.headers.get("content-type"),
        "candidate_accept_type": candidate_accept.headers.get("content-type"),
        "reference_content_encoding": reference_encoding.headers.get(
            "content-encoding"
        ),
        "candidate_content_encoding": candidate_encoding.headers.get(
            "content-encoding"
        ),
    }
    for key, value in observations.items():
        print(f"{key}={value}")

    assert candidate_accept.headers.get(
        "content-type"
    ) == reference_accept.headers.get("content-type")
    assert candidate_encoding.headers.get(
        "content-encoding"
    ) == reference_encoding.headers.get("content-encoding")


if __name__ == "__main__":
    asyncio.run(main())
