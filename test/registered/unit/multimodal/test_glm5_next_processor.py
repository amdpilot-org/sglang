import asyncio
import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.configs.glm5_next import Glm5NextConfig
from sglang.srt.configs.glm5_next_processing import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    smart_resize,
)
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.multimodal.customized_mm_processor_utils import (
    _CUSTOMIZED_MM_PROCESSOR,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.glm4v import (
    Glm4vImageProcessor,
    _collapse_glm5_next_image_tokens,
)
from sglang.srt.utils import load_image


def _jpeg_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_glm5_next_uses_vision_processor():
    assert _CUSTOMIZED_MM_PROCESSOR[Glm5NextConfig.model_type] is Glm5NextProcessor


def test_glm5_next_jpeg_data_url_reaches_image_processor():
    source = Image.new("RGB", (64, 48), color=(20, 90, 170))
    decoded, _ = load_image(_jpeg_data_url(source), gpu_image_decode=False)

    output = Glm5NextImageProcessor()(images=decoded, return_tensors="pt")

    assert decoded.size == (64, 48)
    assert output.image_grid_thw.tolist() == [[1, 8, 10]]
    assert output.pixel_values.shape == (80, 1176)
    assert output.pixel_values.isfinite().all()


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (28, 28, (112, 112)),
        (48, 64, (112, 140)),
        (4096, 4096, (2492, 2492)),
    ],
)
def test_glm5_next_smart_resize_boundaries(height, width, expected):
    resized = smart_resize(num_frames=2, height=height, width=width)
    assert resized == expected
    assert resized[0] % 28 == 0
    assert resized[1] % 28 == 0


def test_glm5_next_smart_resize_rejects_impossible_budget():
    with pytest.raises(ValueError, match="too small"):
        smart_resize(
            num_frames=2,
            height=64,
            width=64,
            temporal_factor=2,
            factor=28,
            max_pixels=0,
        )


def test_collapse_glm5_next_image_tokens_preserves_distinct_spans():
    assert _collapse_glm5_next_image_tokens(
        [10, 99, 99, 11, 10, 99, 99, 99, 11], 99
    ) == [10, 99, 11, 10, 99, 11]
    assert _collapse_glm5_next_image_tokens([99, 99, 1, 99, 99], 99) == [
        99,
        1,
        99,
    ]
    assert _collapse_glm5_next_image_tokens([], 99) == []


def test_collapse_glm5_next_image_tokens_respects_image_count():
    assert _collapse_glm5_next_image_tokens([10, 99, 99, 11], 99, 2) == [
        10,
        99,
        99,
        11,
    ]
    assert _collapse_glm5_next_image_tokens([10, 99, 99, 99, 11], 99, 2) == [
        10,
        99,
        99,
        11,
    ]
    assert _collapse_glm5_next_image_tokens([10, 99, 11], 99, 2) == [10, 99, 11]


@pytest.mark.parametrize(
    ("model_type", "expected_prompt"),
    [
        ("glm5_next", [1, 10, 99, 11, 2]),
        ("glm4v", [1, 10, 99, 99, 99, 99, 11, 2]),
    ],
)
def test_processor_expanded_image_span_is_collapsed_only_for_glm5_next(
    monkeypatch, model_type, expected_prompt
):
    expanded = [1, 10, 99, 99, 99, 99, 11, 2]
    processor = object.__new__(Glm4vImageProcessor)
    processor.hf_config = SimpleNamespace(model_type=model_type)
    processor.IM_TOKEN_ID = 99
    processor.mm_tokens = MultimodalSpecialTokens(image_token_id=99, video_token_id=98)
    processor.video_config = {}
    processor._processor = SimpleNamespace(video_processor=None)
    processor.load_mm_data = AsyncMock(
        return_value=BaseMultiModalProcessorOutput(
            input_text="unused", input_ids=expected_prompt
        )
    )
    processor.process_and_combine_mm_data_async = AsyncMock(
        return_value=([], torch.tensor(expanded), SimpleNamespace())
    )
    monkeypatch.setattr(
        MRotaryEmbedding,
        "get_rope_index_glm4v",
        MagicMock(return_value=(torch.zeros((3, 1, len(expanded))), torch.zeros(1))),
    )

    output = asyncio.run(
        processor.process_mm_data_async(
            image_data=[],
            input_text=expanded,
            request_obj=SimpleNamespace(video_data=None),
        )
    )

    assert processor.load_mm_data.await_args.kwargs["prompt"] == expected_prompt
    assert output.input_ids == expanded


def test_adjacent_pretokenized_image_placeholders_preserve_media_cardinality(
    monkeypatch,
):
    adjacent = [1, 10, 99, 99, 11, 2]
    processor = object.__new__(Glm4vImageProcessor)
    processor.hf_config = SimpleNamespace(model_type="glm5_next")
    processor.IM_TOKEN_ID = 99
    processor.mm_tokens = MultimodalSpecialTokens(image_token_id=99, video_token_id=98)
    processor.video_config = {}
    processor._processor = SimpleNamespace(video_processor=None)
    processor.load_mm_data = AsyncMock(
        return_value=BaseMultiModalProcessorOutput(
            input_text="unused", input_ids=adjacent
        )
    )
    processor.process_and_combine_mm_data_async = AsyncMock(
        return_value=([], torch.tensor(adjacent), SimpleNamespace())
    )
    monkeypatch.setattr(
        MRotaryEmbedding,
        "get_rope_index_glm4v",
        MagicMock(return_value=(torch.zeros((3, 1, len(adjacent))), torch.zeros(1))),
    )

    asyncio.run(
        processor.process_mm_data_async(
            image_data=["first", "second"],
            input_text=adjacent,
            request_obj=SimpleNamespace(video_data=None),
        )
    )

    assert processor.load_mm_data.await_args.kwargs["prompt"] == adjacent
