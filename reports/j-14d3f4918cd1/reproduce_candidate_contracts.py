import asyncio
import os
import subprocess
import sys
import tempfile

import httpx
from fastapi import FastAPI
from prometheus_client import CollectorRegistry, Gauge, make_asgi_app

from sglang.srt.utils.common import add_prometheus_middleware


async def request(app, path, headers=None):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.get(path, headers=headers)


async def main():
    registry = CollectorRegistry()
    Gauge("alpha_total", "alpha", registry=registry).set(1)
    Gauge("beta_total", "beta", registry=registry).set(2)
    reference = make_asgi_app(registry=registry)

    reference_filtered = await request(reference, "/?name[]=alpha_total")
    reference_gzip = await request(reference, "/", {"Accept-Encoding": "gzip"})
    reference_openmetrics = await request(
        reference, "/", {"Accept": "application/openmetrics-text"}
    )

    with tempfile.TemporaryDirectory() as multiproc_dir:
        env = os.environ.copy()
        env["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from prometheus_client import Gauge;"
                "Gauge('alpha_total','alpha').set(1);"
                "Gauge('beta_total','beta').set(2)",
            ],
            env=env,
            check=True,
        )
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
        candidate = FastAPI()
        add_prometheus_middleware(candidate)
        candidate_filtered = await request(candidate, "/metrics?name[]=alpha_total")
        candidate_gzip = await request(
            candidate, "/metrics", {"Accept-Encoding": "gzip"}
        )
        candidate_openmetrics = await request(
            candidate, "/metrics", {"Accept": "application/openmetrics-text"}
        )

    checks = {
        "reference_name_filter_excludes_beta": b"beta_total" not in reference_filtered.content,
        "candidate_name_filter_excludes_beta": b"beta_total" not in candidate_filtered.content,
        "reference_gzip": reference_gzip.headers.get("content-encoding") == "gzip",
        "candidate_gzip": candidate_gzip.headers.get("content-encoding") == "gzip",
        "reference_gzip_body_is_readable": b"alpha_total" in reference_gzip.content,
        "reference_openmetrics": reference_openmetrics.headers.get("content-type", "").startswith(
            "application/openmetrics-text"
        ),
        "candidate_openmetrics": candidate_openmetrics.headers.get("content-type", "").startswith(
            "application/openmetrics-text"
        ),
    }
    for name, value in checks.items():
        print(f"{name}={value}")
    print(f"candidate_gzip_content_encoding={candidate_gzip.headers.get('content-encoding')}")
    print(f"candidate_openmetrics_content_type={candidate_openmetrics.headers.get('content-type')}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
