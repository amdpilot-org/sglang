"""Regression tests for per-processor image count limits."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.srt.arg_groups.serving_hook import handle_other_validations
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.internvl import InternVLProcessor
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


class _Processor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class _HigherLimitProcessor(BaseMultimodalProcessor):
    IMAGE_NUM_LIMITATION = 12

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


def _processor(cls=_Processor):
    return cls.__new__(cls)


def _images(count):
    return [f"image-{index}" for index in range(count)]


class TestImageNumLimitation(CustomTestCase):
    def test_default_limit_and_boundary(self):
        processor = _processor()
        with patch(
            "sglang.srt.multimodal.processors.base_processor.get_mm",
            return_value=SimpleNamespace(limit_mm_data_per_request=None),
        ):
            processor.validate_image_num_limitation(_images(5))
            with self.assertRaisesRegex(ValueError, "Image count 6 exceeds limit 5"):
                processor.validate_image_num_limitation(_images(6))

    def test_processor_override(self):
        processor = _processor(_HigherLimitProcessor)
        with patch(
            "sglang.srt.multimodal.processors.base_processor.get_mm",
            return_value=SimpleNamespace(limit_mm_data_per_request=None),
        ):
            processor.validate_image_num_limitation(_images(12))
            with self.assertRaisesRegex(ValueError, "exceeds limit 12"):
                processor.validate_image_num_limitation(_images(13))

    def test_internvl_declares_image_count_not_patch_limit(self):
        self.assertEqual(InternVLProcessor.IMAGE_NUM_LIMITATION, 12)
        self.assertEqual(InternVLProcessor.IMAGE_MAX_NUM, 12)

    def test_server_override_can_lower_raise_or_disable_default(self):
        processor = _processor()
        for limit, accepted, rejected in ((2, 2, 3), (8, 8, 9), (0, 0, 1)):
            with (
                self.subTest(limit=limit),
                patch(
                    "sglang.srt.multimodal.processors.base_processor.get_mm",
                    return_value=SimpleNamespace(
                        limit_mm_data_per_request={"image": limit}
                    ),
                ),
            ):
                processor.validate_image_num_limitation(_images(accepted))
                with self.assertRaisesRegex(ValueError, f"exceeds limit {limit}"):
                    processor.validate_image_num_limitation(_images(rejected))

    def test_other_modality_override_preserves_default_image_limit(self):
        processor = _processor()
        with patch(
            "sglang.srt.multimodal.processors.base_processor.get_mm",
            return_value=SimpleNamespace(limit_mm_data_per_request={"video": 1}),
        ):
            with self.assertRaisesRegex(ValueError, "exceeds limit 5"):
                processor.validate_image_num_limitation(_images(6))

    def test_preprocessed_input_is_exempt(self):
        processor = _processor()
        with patch(
            "sglang.srt.multimodal.processors.base_processor.get_mm",
            return_value=SimpleNamespace(limit_mm_data_per_request=None),
        ):
            processor.validate_image_num_limitation(
                [{"format": "precomputed_embedding", "embedding": object()}]
            )

    def test_malformed_runtime_limit_fails_closed(self):
        processor = _processor()
        for limit in (True, -1, 1.5, "5"):
            with (
                self.subTest(limit=limit),
                patch(
                    "sglang.srt.multimodal.processors.base_processor.get_mm",
                    return_value=SimpleNamespace(
                        limit_mm_data_per_request={"image": limit}
                    ),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    processor.validate_image_num_limitation(_images(1))

    def test_server_rejects_malformed_limits_at_startup(self):
        for limit in (True, -1, 1.5, "5"):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "non-negative integers"),
            ):
                handle_other_validations(
                    ServerArgs(
                        model_path="dummy",
                        limit_mm_data_per_request={"image": limit},
                    )
                )

    def test_load_mm_data_rejects_before_decode(self):
        processor = _processor()
        processor.fast_load_mm_data = AsyncMock(
            side_effect=AssertionError("image decoding must not start")
        )
        processor.legacy_load_mm_data = AsyncMock(
            side_effect=AssertionError("image decoding must not start")
        )
        with patch(
            "sglang.srt.multimodal.processors.base_processor.get_mm",
            return_value=SimpleNamespace(limit_mm_data_per_request=None),
        ):
            with self.assertRaisesRegex(ValueError, "exceeds limit 5"):
                asyncio.run(
                    processor.load_mm_data(
                        prompt="irrelevant",
                        multimodal_tokens=object(),
                        image_data=_images(6),
                    )
                )
        processor.fast_load_mm_data.assert_not_awaited()
        processor.legacy_load_mm_data.assert_not_awaited()

    def test_common_serving_validation_covers_processor_bypasses(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.mm_processor = _processor()
        request = SimpleNamespace(image_data=_images(6))
        with (
            patch(
                "sglang.srt.managers.tokenizer_manager.get_mm",
                return_value=SimpleNamespace(limit_mm_data_per_request=None),
            ),
            patch(
                "sglang.srt.multimodal.processors.base_processor.get_mm",
                return_value=SimpleNamespace(limit_mm_data_per_request=None),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "exceeds limit 5"):
                manager._validate_mm_limits(request)


if __name__ == "__main__":
    unittest.main()
