import asyncio
import os
import subprocess
import sys
import tempfile

import httpx
from fastapi import FastAPI
from prometheus_client import CollectorRegistry, Gauge, make_asgi_app

from sglang.srt.utils.common import add_prometheus_middleware


async def get(app, headers):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.get("/metrics", headers=headers)


async def main():
    registry = CollectorRegistry()
    Gauge("review_metric", "review", registry=registry).set(1)
    reference = make_asgi_app(registry=registry)
    with tempfile.TemporaryDirectory() as multiproc_dir:
        env = os.environ.copy()
        env["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
        subprocess.run(
            [sys.executable, "-c", "from prometheus_client import Gauge; Gauge('review_metric','review').set(1)"],
            env=env,
            check=True,
        )
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
        candidate = FastAPI()
        add_prometheus_middleware(candidate)

        duplicate_accept = [
            ("accept", "application/json"),
            ("accept", "application/openmetrics-text"),
            ("accept-encoding", "identity"),
        ]
        duplicate_encoding = [
            ("accept", "text/plain"),
            ("accept-encoding", "identity;q=0"),
            ("accept-encoding", "gzip"),
        ]
        ref_accept = await get(reference, duplicate_accept)
        cand_accept = await get(candidate, duplicate_accept)
        ref_encoding = await get(reference, duplicate_encoding)
        cand_encoding = await get(candidate, duplicate_encoding)

    observations = {
        "reference_duplicate_accept_status": ref_accept.status_code,
        "candidate_duplicate_accept_status": cand_accept.status_code,
        "reference_duplicate_accept_type": ref_accept.headers.get("content-type"),
        "candidate_duplicate_accept_type": cand_accept.headers.get("content-type"),
        "reference_duplicate_encoding": ref_encoding.headers.get("content-encoding"),
        "candidate_duplicate_encoding": cand_encoding.headers.get("content-encoding"),
    }
    for key, value in observations.items():
        print(f"{key}={value}")
    assert cand_accept.status_code == ref_accept.status_code
    assert cand_accept.headers.get("content-type") == ref_accept.headers.get("content-type")
    assert cand_encoding.headers.get("content-encoding") == ref_encoding.headers.get("content-encoding")


if __name__ == "__main__":
    asyncio.run(main())
