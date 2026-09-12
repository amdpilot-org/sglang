import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from array import array
from unittest.mock import patch

import httpx
import torch
from fastapi import FastAPI

from sglang.srt.utils import common
from sglang.srt.utils.common import (
    flatten_arrays_to_int64_tensor,
    get_device_sm_nvidia_smi,
    get_nvidia_driver_version_str,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")


class _FakeMetricsProcess:
    def __init__(self, communicate, returncode=0):
        self.communicate = communicate
        self.returncode = returncode
        self.killed = False
        self.input = None

    def kill(self):
        self.killed = True


class TestPrometheusMetricsExporter(unittest.IsolatedAsyncioTestCase):
    async def test_http_negotiation_and_name_filtering(self):
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
            with patch.dict(
                os.environ, {"PROMETHEUS_MULTIPROC_DIR": multiproc_dir}
            ):
                app = FastAPI()
                common.add_prometheus_middleware(app)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    filtered = await client.get(
                        "/metrics?name[]=alpha_total",
                        headers={"Accept-Encoding": "identity"},
                    )
                    compressed = await client.get(
                        "/metrics", headers={"Accept-Encoding": "gzip"}
                    )
                    openmetrics = await client.get(
                        "/metrics",
                        headers={
                            "Accept": "application/openmetrics-text",
                            "Accept-Encoding": "identity",
                        },
                    )

        self.assertNotIn("beta_total", filtered.text)
        self.assertEqual(compressed.headers["content-encoding"], "gzip")
        self.assertIn("alpha_total", compressed.text)
        self.assertTrue(
            openmetrics.headers["content-type"].startswith(
                "application/openmetrics-text"
            )
        )

    async def test_slow_scrape_does_not_block_loop_and_concurrent_scrape_fails_fast(
        self,
    ):
        started = asyncio.Event()
        release = asyncio.Event()

        async def communicate(input=None):
            process.input = input
            started.set()
            await release.wait()
            return (
                b'{"status": 200, "headers": [["Content-Type", "text/plain"]]}'
                b"\nsglang_test_metric 1\n",
                b"",
            )

        process = _FakeMetricsProcess(communicate)
        exporter = common._PrometheusMetricsExporter("/tmp/prom", 8)
        with patch.object(
            asyncio, "create_subprocess_exec", return_value=process
        ) as create_process:
            first_scrape = asyncio.create_task(
                exporter.generate(
                    query_string="name[]=sglang_test_metric",
                    accept="application/openmetrics-text",
                    accept_encoding="gzip",
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)

            # This yield represents unrelated work such as a /health handler.
            await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
            self.assertEqual(
                await exporter.generate(),
                (
                    503,
                    common._PROMETHEUS_ERROR_HEADERS,
                    b"Prometheus metrics collection already in progress\n",
                ),
            )
            release.set()
            self.assertEqual(
                await first_scrape,
                (200, [["Content-Type", "text/plain"]], b"sglang_test_metric 1\n"),
            )
            create_process.assert_called_once()
            self.assertEqual(
                common.json.loads(process.input),
                {
                    "query_string": "name[]=sglang_test_metric",
                    "accept": "application/openmetrics-text",
                    "accept_encoding": "gzip",
                },
            )

    async def test_collection_timeout_kills_child(self):
        async def communicate(input=None):
            if process.killed:
                return b"", b""
            await asyncio.Event().wait()

        process = _FakeMetricsProcess(communicate)
        exporter = common._PrometheusMetricsExporter("/tmp/prom", 0.01)
        with patch.object(asyncio, "create_subprocess_exec", return_value=process):
            self.assertEqual(
                await exporter.generate(),
                (
                    504,
                    common._PROMETHEUS_ERROR_HEADERS,
                    b"Prometheus metrics collection timed out\n",
                ),
            )
        self.assertTrue(process.killed)

    async def test_collection_failure_does_not_expose_child_stderr(self):
        async def communicate(input=None):
            return b"partial", b"private child failure"

        process = _FakeMetricsProcess(communicate, returncode=1)
        exporter = common._PrometheusMetricsExporter("/tmp/prom", 8)
        with patch.object(asyncio, "create_subprocess_exec", return_value=process):
            self.assertEqual(
                await exporter.generate(),
                (
                    500,
                    common._PROMETHEUS_ERROR_HEADERS,
                    b"Prometheus metrics collection failed\n",
                ),
            )

    async def test_cancelled_scrape_kills_child(self):
        started = asyncio.Event()

        async def communicate(input=None):
            if process.killed:
                return b"", b""
            started.set()
            await asyncio.Event().wait()

        process = _FakeMetricsProcess(communicate)
        exporter = common._PrometheusMetricsExporter("/tmp/prom", 8)
        with patch.object(asyncio, "create_subprocess_exec", return_value=process):
            scrape = asyncio.create_task(exporter.generate())
            await asyncio.wait_for(started.wait(), timeout=1)
            scrape.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await scrape
        self.assertTrue(process.killed)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFlattenArraysToInt64Tensor(CustomTestCase):
    """`flatten_arrays_to_int64_tensor` is invoked by `prepare_for_extend`
    to build the per-batch input_ids tensor (pinned, async H2D) from a
    list of array.array('q') per-req get_fill_ids() slices. Tests the
    full matrix of (device, pin) the production code paths through.
    """

    DEVICES = ("cpu", "cuda")
    PIN_OPTIONS = (False, True)

    def _check(self, parts: list, expected: list[int]) -> None:
        for device in self.DEVICES:
            for pin in self.PIN_OPTIONS:
                with self.subTest(device=device, pin=pin):
                    out = flatten_arrays_to_int64_tensor(parts, device, pin)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    self.assertEqual(out.dtype, torch.int64)
                    self.assertEqual(out.device.type, device)
                    self.assertEqual(out.shape, (len(expected),))
                    self.assertEqual(out.cpu().tolist(), expected)

    def test_single_part(self):
        parts = [array("q", [1, 2, 3, 4, 5])]
        self._check(parts, [1, 2, 3, 4, 5])

    def test_multiple_parts(self):
        parts = [
            array("q", [10, 20, 30]),
            array("q", [100, 200]),
            array("q", [1000]),
        ]
        self._check(parts, [10, 20, 30, 100, 200, 1000])


class TestNvidiaDriverVersionStr(CustomTestCase):
    """`get_nvidia_driver_version_str` is typed as `str | None`: it returns
    `None` when nvidia-smi is missing, fails, or emits an empty string. These
    tests exercise both the success and the None-return paths by monkey-
    patching `subprocess.run`, so they don't require a GPU. The function is
    `@lru_cache`d, so the cache is cleared around each test to make the patch
    observable.
    """

    def setUp(self):
        get_nvidia_driver_version_str.cache_clear()

    def tearDown(self):
        get_nvidia_driver_version_str.cache_clear()

    def test_returns_version_string(self):
        import subprocess

        class _R:
            stdout = "595.58.03\n"

        original = subprocess.run
        subprocess.run = lambda *a, **k: _R()
        try:
            self.assertEqual(get_nvidia_driver_version_str(), "595.58.03")
        finally:
            subprocess.run = original

    def test_returns_none_on_empty_output(self):
        import subprocess

        class _R:
            stdout = "\n"

        original = subprocess.run
        subprocess.run = lambda *a, **k: _R()
        try:
            self.assertIsNone(get_nvidia_driver_version_str())
        finally:
            subprocess.run = original

    def test_returns_none_on_called_process_error(self):
        import subprocess

        original = subprocess.run

        def boom(*a, **k):
            raise subprocess.CalledProcessError(1, "nvidia-smi")

        subprocess.run = boom
        try:
            self.assertIsNone(get_nvidia_driver_version_str())
        finally:
            subprocess.run = original

    def test_returns_none_on_file_not_found(self):
        import subprocess

        original = subprocess.run

        def boom(*a, **k):
            raise FileNotFoundError("nvidia-smi")

        subprocess.run = boom
        try:
            self.assertIsNone(get_nvidia_driver_version_str())
        finally:
            subprocess.run = original


class TestGetDeviceSmNvidiaSmi(CustomTestCase):
    """`get_device_sm_nvidia_smi` parses nvidia-smi output into a (major,
    minor) tuple and falls back to (0, 0) -- logging via `logger.error` --
    when nvidia-smi fails. The success path needs a GPU; the fallback path is
    covered here by forcing a failure and asserting the (0, 0) return. The
    fallback path needs no GPU, so this test runs on CPU.
    """

    def test_fallback_on_failure_returns_zero_zero(self):
        import subprocess

        original = subprocess.run

        def boom(*a, **k):
            raise subprocess.CalledProcessError(1, "nvidia-smi")

        subprocess.run = boom
        try:
            self.assertEqual(get_device_sm_nvidia_smi(), (0, 0))
        finally:
            subprocess.run = original


if __name__ == "__main__":
    unittest.main()
