import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.multimodal.cache import MultimodalPreprocessCache
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64


def _processor():
    processor = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
    processor.mm_preprocess_cache = MultimodalPreprocessCache(1024 * 1024)
    processor.trust_mm_content_hashes = True
    processor.processor_fingerprint = "qwen2-vl:test-processor"
    return processor


def _request(hashes, *, video_data=None, audio_data=None):
    return SimpleNamespace(
        mm_content_hashes=hashes,
        video_data=video_data,
        audio_data=audio_data,
    )


class TestQwenPreprocessRequestCache(unittest.IsolatedAsyncioTestCase):
    async def test_trusted_content_id_loads_without_opening_media(self):
        processor = _processor()
        request = _request([_HASH_A])
        key = processor._request_preprocess_cache_key(
            ["https://store.invalid/original.png"], "describe", request
        )
        retained = MultimodalProcessorOutput(mm_items=[], input_ids=[1, 2, 3])
        self.assertTrue(processor.mm_preprocess_cache.put(key, retained))

        # The URL deliberately differs and is unreachable. A content-ID hit
        # must return before load_mm_data attempts to open it.
        result = await processor.process_mm_data_async(
            ["https://must-not-be-opened.invalid/missing.png"],
            "describe",
            request,
        )

        self.assertEqual(result.input_ids, [1, 2, 3])
        self.assertIsNot(result, retained)
        result.input_ids.append(4)
        self.assertEqual(retained.input_ids, [1, 2, 3])

    async def test_identical_cold_requests_share_preprocessing(self):
        processor = _processor()
        request = _request([_HASH_A])
        started = asyncio.Event()
        release = asyncio.Event()
        retained = MultimodalProcessorOutput(mm_items=[], input_ids=[1, 2, 3])

        async def process_once(*args, **kwargs):
            started.set()
            await release.wait()
            return retained

        processor._process_mm_data_uncached = AsyncMock(side_effect=process_once)
        first = asyncio.create_task(
            processor.process_mm_data_async(["first-url"], "describe", request)
        )
        await started.wait()
        second = asyncio.create_task(
            processor.process_mm_data_async(["second-url"], "describe", request)
        )
        await asyncio.sleep(0)

        self.assertEqual(processor._process_mm_data_uncached.await_count, 1)
        release.set()
        first_result, second_result = await asyncio.gather(first, second)
        self.assertEqual(processor.mm_preprocess_cache.stats()["singleflight_joins"], 1)
        self.assertIsNot(first_result, second_result)
        first_result.input_ids.append(4)
        self.assertEqual(second_result.input_ids, [1, 2, 3])

    def test_identity_is_ordered_prompt_and_processor_scoped(self):
        processor = _processor()
        request = _request([_HASH_A, _HASH_B])
        base = processor._request_preprocess_cache_key(
            ["first", "second"], "prompt", request
        )

        self.assertEqual(
            base,
            processor._request_preprocess_cache_key(
                ["different-url", "another-url"], "prompt", request
            ),
        )
        self.assertNotEqual(
            base,
            processor._request_preprocess_cache_key(
                ["first", "second"], "changed prompt", request
            ),
        )
        self.assertNotEqual(
            base,
            processor._request_preprocess_cache_key(
                ["first", "second"], "prompt", _request([_HASH_B, _HASH_A])
            ),
        )
        processor.processor_fingerprint = "qwen2-vl:changed-config"
        self.assertNotEqual(
            base,
            processor._request_preprocess_cache_key(
                ["first", "second"], "prompt", request
            ),
        )

    def test_bypass_requires_complete_trusted_image_only_identity(self):
        processor = _processor()
        self.assertIsNone(
            processor._request_preprocess_cache_key(
                ["image"], "prompt", _request([None])
            )
        )
        self.assertIsNone(
            processor._request_preprocess_cache_key(
                ["image", "image2"], "prompt", _request([_HASH_A])
            )
        )
        self.assertIsNone(
            processor._request_preprocess_cache_key(
                ["image"], "prompt", _request([_HASH_A], video_data=["clip"])
            )
        )
        processor.trust_mm_content_hashes = False
        self.assertIsNone(
            processor._request_preprocess_cache_key(
                ["image"], "prompt", _request([_HASH_A])
            )
        )
        processor.trust_mm_content_hashes = True
        processor.mm_feature_transport = "cuda_ipc"
        self.assertIsNone(
            processor._request_preprocess_cache_key(
                ["image"], "prompt", _request([_HASH_A])
            )
        )


if __name__ == "__main__":
    unittest.main()
