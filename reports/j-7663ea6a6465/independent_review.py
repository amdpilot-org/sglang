import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor


class Processor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


def proc():
    return Processor.__new__(Processor)


def main():
    p = proc()
    assert hasattr(p, "IMAGE_NUM_LIMITATION"), "missing processor default"
    with patch(
        "sglang.srt.multimodal.processors.base_processor.get_mm",
        return_value=SimpleNamespace(limit_mm_data_per_request=None),
    ):
        p.validate_image_num_limitation([object()] * 5)
        try:
            p.validate_image_num_limitation([object()] * 6)
        except ValueError as exc:
            assert "Image count 6 exceeds limit 5" in str(exc)
        else:
            raise AssertionError("six images were accepted")

    # Independent early-rejection proof: neither decode path may be entered.
    p.fast_load_mm_data = AsyncMock(side_effect=AssertionError("decoded"))
    p.legacy_load_mm_data = AsyncMock(side_effect=AssertionError("decoded"))
    with patch(
        "sglang.srt.multimodal.processors.base_processor.get_mm",
        return_value=SimpleNamespace(limit_mm_data_per_request={"video": 1}),
    ):
        try:
            asyncio.run(p.load_mm_data("x", object(), image_data=[b"x"] * 6))
        except ValueError as exc:
            assert "exceeds limit 5" in str(exc)
        else:
            raise AssertionError("other-modality override disabled image default")
    p.fast_load_mm_data.assert_not_awaited()
    p.legacy_load_mm_data.assert_not_awaited()

    # A configured image limit must override, including zero.
    for limit, count, rejected in [(8, 8, 9), (0, 0, 1)]:
        with patch(
            "sglang.srt.multimodal.processors.base_processor.get_mm",
            return_value=SimpleNamespace(limit_mm_data_per_request={"image": limit}),
        ):
            p.validate_image_num_limitation([object()] * count)
            try:
                p.validate_image_num_limitation([object()] * rejected)
            except ValueError:
                pass
            else:
                raise AssertionError((limit, rejected))

    # Common serving validation protects processors that bypass the base loader.
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.mm_processor = p
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
        try:
            manager._validate_mm_limits(SimpleNamespace(image_data=[b"x"] * 6))
        except ValueError:
            pass
        else:
            raise AssertionError("serving bypass accepted six images")

    print("independent candidate contract checks passed")


if __name__ == "__main__":
    main()
