import asyncio
import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.transformers_auto import (
    TransformersAutoMultimodalProcessor,
)


class RecordingCollector:
    def __init__(self):
        self.media = []
        self.processor = []

    def observe_mm_media_load(self, **kwargs):
        self.media.append(kwargs)

    def observe_mm_processor(self, seconds):
        self.processor.append(seconds)


class ImageHandler(BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *args):
        pass


def test_transformers_auto_remote_image_emits_per_item_metrics():
    output = io.BytesIO()
    Image.new("RGB", (2, 2)).save(output, format="PNG")
    ImageHandler.payload = output.getvalue()
    server = ThreadingHTTPServer(("127.0.0.1", 0), ImageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        processor = TransformersAutoMultimodalProcessor.__new__(
            TransformersAutoMultimodalProcessor
        )
        processor.metrics_collector = RecordingCollector()
        processor._load_images(
            [f"http://127.0.0.1:{server.server_port}/fixture.png"]
        )
        assert len(processor.metrics_collector.media) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_async_override_direct_processor_call_is_observed():
    class DirectAsyncProcessor(BaseMultimodalProcessor):
        async def process_mm_data_async(self, *args, **kwargs):
            return self.process_mm_data()

        def process_mm_data(self, **kwargs):
            time.sleep(0.01)
            return {"input_ids": [[1]]}

    processor = DirectAsyncProcessor.__new__(DirectAsyncProcessor)
    processor.metrics_collector = RecordingCollector()
    asyncio.run(processor.process_mm_data_async())
    assert len(processor.metrics_collector.processor) == 1
