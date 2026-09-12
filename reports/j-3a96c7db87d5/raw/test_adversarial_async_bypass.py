import asyncio
import time
from types import SimpleNamespace

import torch

from sglang.srt.multimodal.processors.transformers_auto import (
    TransformersAutoMultimodalProcessor,
)


class RecordingCollector:
    def __init__(self):
        self.processor = []

    def observe_mm_processor(self, seconds):
        self.processor.append(seconds)


def test_transformers_auto_direct_hf_processor_path_is_observed():
    collector = RecordingCollector()
    processor = TransformersAutoMultimodalProcessor.__new__(
        TransformersAutoMultimodalProcessor
    )
    processor.metrics_collector = collector
    processor.mm_tokens = SimpleNamespace(
        image_token_id=None, video_token_id=None, audio_token_id=None
    )
    processor.hf_config = SimpleNamespace()
    processor._is_mrope = False
    processor._load_images = lambda image_data: []

    def apply_hf_processor(**kwargs):
        time.sleep(0.02)
        return {"input_ids": torch.tensor([[1]])}

    processor._apply_hf_processor = apply_hf_processor
    processor._build_mm_items = lambda output, input_ids: []

    result = asyncio.run(
        processor.process_mm_data_async(
            image_data=[],
            audio_data=[],
            input_text="fixture",
            request_obj=SimpleNamespace(video_data=None),
        )
    )

    assert result.input_ids == [1]
    assert len(collector.processor) == 1
    assert collector.processor[0] >= 0.015
