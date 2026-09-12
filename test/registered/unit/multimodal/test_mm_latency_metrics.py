import asyncio
import concurrent.futures
import io
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.moss_vl import MossVLImageProcessor


class _RecordingCollector:
    def __init__(self):
        self.media = []
        self.load_data = []
        self.processor = []

    def observe_mm_media_load(self, **kwargs):
        self.media.append(kwargs)

    def observe_mm_load_data(self, seconds):
        self.load_data.append(seconds)

    def observe_mm_processor(self, seconds):
        self.processor.append(seconds)


class _StubProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class _ImageHandler(BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self):
        time.sleep(0.02)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, format, *args):
        pass


def _png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (2, 2), (12, 34, 56)).save(output, format="PNG")
    return output.getvalue()


def test_media_timings_preserve_absent_download_and_error_paths():
    collector = _RecordingCollector()
    payload = _png_bytes()
    _ImageHandler.payload = payload
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ImageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/fixture.png"
        loaded = _StubProcessor._load_single_item(
            url, Modality.IMAGE, metrics_collector=collector
        )
        assert loaded.size == (2, 2)
        assert len(collector.media) == 1
        remote = collector.media[0]
        assert remote["modality"] == "image"
        assert remote["load_seconds"] >= remote["download_seconds"] > 0
        assert remote["download_bytes"] == len(payload)

        local = _StubProcessor._load_single_item(
            Image.new("RGB", (1, 1)),
            Modality.IMAGE,
            metrics_collector=collector,
        )
        assert local.size == (1, 1)
        assert collector.media[1]["download_seconds"] is None
        assert collector.media[1]["download_bytes"] is None

        try:
            _StubProcessor._load_single_item(
                b"not an image", Modality.IMAGE, metrics_collector=collector
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid media must fail")
        assert len(collector.media) == 3
        assert collector.media[2]["load_seconds"] > 0
    finally:
        server.shutdown()
        server.server_close()


def test_request_stage_timings_are_isolated():
    collector = _RecordingCollector()
    processor = _StubProcessor.__new__(_StubProcessor)
    processor.metrics_collector = collector
    processor.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    processor._tokenizer_auto_adds_specials = False
    processor.image_config = None
    processor.video_config = None
    processor.audio_config = None
    processor.disable_fast_image_processor = True
    processor.video_preprocessing_device = None
    processor.mm_feature_transport = "cpu"
    processor.precompute_hash_before_cpu_transfer = False
    processor.FEATURE_NAMES = []

    class _Processor:
        def __call__(self, **kwargs):
            time.sleep(kwargs.pop("delay"))
            return {"input_ids": [[1]]}

    processor._resolve_processor = lambda value: (_Processor(), object())

    async def run_load(delay):
        await processor._observe_mm_load_data(asyncio.sleep(delay))

    async def exercise():
        await asyncio.gather(run_load(0.02), run_load(0.04))

    try:
        asyncio.run(exercise())
        processor._call_process_mm_data(input_text="a", delay=0.02)
        processor._call_process_mm_data(input_text="b", delay=0.04)
    finally:
        processor.io_executor.shutdown()

    assert len(collector.load_data) == 2
    assert sorted(collector.load_data)[0] >= 0.015
    assert sorted(collector.load_data)[1] >= 0.035
    assert len(collector.processor) == 2
    assert collector.processor[0] >= 0.015
    assert collector.processor[1] >= 0.035


def test_processor_override_is_timed_at_shared_dispatch():
    collector = _RecordingCollector()

    class _OverrideProcessor(_StubProcessor):
        def process_mm_data(self, **kwargs):
            time.sleep(0.02)
            return {"input_ids": [[1]]}

    processor = _OverrideProcessor.__new__(_OverrideProcessor)
    processor.metrics_collector = collector

    processor._call_process_mm_data(input_text="override")

    assert len(collector.processor) == 1
    assert collector.processor[0] >= 0.015


def test_specialized_video_download_observes_item_metrics(monkeypatch):
    collector = _RecordingCollector()
    processor = MossVLImageProcessor.__new__(MossVLImageProcessor)
    processor.metrics_collector = collector
    payload = b"bounded-video-fixture"

    def fake_download(url, timeout):
        from sglang.srt.utils.common import download_remote_media

        # Exercise the real capture hook while keeping the fixture local.
        monkeypatch.setattr(
            "sglang.srt.utils.common._download_remote_media",
            lambda requested_url, requested_timeout: payload,
        )
        return download_remote_media(url, timeout)

    monkeypatch.setattr(
        "sglang.srt.multimodal.processors.moss_vl.download_remote_media",
        fake_download,
    )
    path = None
    try:
        path, cleanup_paths = processor._normalize_single_video_input(
            "https://fixture.invalid/sample.mp4"
        )
        assert cleanup_paths == [path]
        assert len(collector.media) == 1
        assert collector.media[0]["modality"] == "video"
        assert collector.media[0]["download_seconds"] is not None
        assert collector.media[0]["download_bytes"] == len(payload)
    finally:
        if path is not None:
            os.remove(path)
