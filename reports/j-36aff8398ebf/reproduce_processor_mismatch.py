"""Deterministic reproduction of the GLM-V placeholder/data mismatch.

This exercises BaseMultimodalProcessor.load_mm_data with GLM-V's actual token
strings.  The media loader is stubbed because the defect occurs after prompt
splitting but before image decoding and does not require model weights.
"""

import asyncio
import concurrent.futures

from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class ReproductionProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _load_single_item(data, modality, frame_count_limit, *args, **kwargs):
        return data


async def main():
    processor = object.__new__(ReproductionProcessor)
    processor.skip_tokenizer_init = False
    processor.support_dynamic_frame_expansion = False
    processor.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    processor.mm_tokens = MultimodalSpecialTokens(
        image_token="<|image|>", video_token="<|video|>"
    )
    processor.mm_tokens.parse_regex()
    try:
        prompt = (
            "<|begin_of_image|><|image|><|end_of_image|>"
            "The template inserts a <|image|> token per image."
        )
        await processor.load_mm_data(
            prompt=prompt,
            multimodal_tokens=processor.mm_tokens,
            image_data=["one-real-image"],
        )
    finally:
        processor.io_executor.shutdown(wait=True)


if __name__ == "__main__":
    asyncio.run(main())
