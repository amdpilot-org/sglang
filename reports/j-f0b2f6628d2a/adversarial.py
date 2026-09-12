import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.multimodal.cache import MultimodalPreprocessCache
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor


CONTENT_ID = "sha256:" + "a" * 64


def make_processor():
    processor = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
    processor.mm_preprocess_cache = MultimodalPreprocessCache(1024 * 1024)
    processor.trust_mm_content_hashes = True
    processor.processor_fingerprint = "qwen2-vl:test"
    return processor


def make_request(video=None, audio=None):
    return SimpleNamespace(
        mm_content_hashes=[CONTENT_ID], video_data=video, audio_data=audio
    )


async def main():
    processor = make_processor()
    calls = 0

    async def compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return MultimodalProcessorOutput(mm_items=[], input_ids=[calls])

    processor._process_mm_data_uncached = AsyncMock(side_effect=compute)
    first_key = processor._request_preprocess_cache_key(
        ["saved-image"], "prompt A", make_request()
    )
    second_key = processor._request_preprocess_cache_key(
        ["saved-image"], "prompt B", make_request()
    )
    await processor.process_mm_data_async(
        ["saved-image"], "prompt A", make_request()
    )
    await processor.process_mm_data_async(
        ["saved-image"], "prompt B", make_request()
    )
    print("same_content_id_prompt_keys_equal", first_key == second_key)
    print("preprocess_calls_across_prompt_change", calls)

    fresh_processor = make_processor()
    fresh_processor._process_mm_data_uncached = AsyncMock(
        return_value=MultimodalProcessorOutput(mm_items=[], input_ids=[3])
    )
    await fresh_processor.process_mm_data_async(
        ["saved-image"], "prompt A", make_request()
    )
    print(
        "fresh_processor_preprocess_calls",
        fresh_processor._process_mm_data_uncached.await_count,
    )
    print(
        "video_cache_key",
        processor._request_preprocess_cache_key(
            ["saved-image"], "prompt A", make_request(video=["clip"])
        ),
    )
    print(
        "audio_cache_key",
        processor._request_preprocess_cache_key(
            ["saved-image"], "prompt A", make_request(audio=["sound"])
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
