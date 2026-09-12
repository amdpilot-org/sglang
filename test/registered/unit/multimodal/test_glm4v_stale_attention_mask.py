import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor


def _run_processor(final_input_ids, processor_mask):
    processor = object.__new__(Glm4vImageProcessor)
    processor.hf_config = SimpleNamespace(model_type="glm4v")
    processor.IM_TOKEN_ID = 99
    processor.mm_tokens = MultimodalSpecialTokens(image_token_id=99, video_token_id=98)
    processor.video_config = {}
    processor._processor = SimpleNamespace(video_processor=None)
    processor.load_mm_data = AsyncMock(
        return_value=BaseMultiModalProcessorOutput(
            input_text="unused", input_ids=final_input_ids, images=[object()]
        )
    )
    processor.process_and_combine_mm_data_async = AsyncMock(
        return_value=(
            [],
            torch.tensor(final_input_ids),
            SimpleNamespace(
                image_grid_thw=torch.tensor([[1, 2, 2]]),
                video_grid_thw=None,
                attention_mask=torch.tensor([processor_mask]),
            ),
        )
    )

    return asyncio.run(
        processor.process_mm_data_async(
            image_data=["image"],
            input_text=final_input_ids,
            request_obj=SimpleNamespace(video_data=None),
        )
    )


@pytest.mark.parametrize(
    ("final_input_ids", "processor_mask"),
    [
        pytest.param([10, 11, 12, 13], [1, 1, 1], id="retokenization-shrank"),
        pytest.param([10, 11, 12], [1, 1, 1], id="canonical-same-length"),
    ],
)
def test_glm4v_mrope_mask_follows_final_input_ids(
    monkeypatch, final_input_ids, processor_mask
):
    observed_masks = []

    def fake_get_rope_index_glm4v(*, input_ids, attention_mask, **kwargs):
        observed_masks.append(attention_mask)
        effective_mask = (
            torch.ones_like(input_ids) if attention_mask is None else attention_mask
        )
        selected_ids = input_ids[0][effective_mask[0] == 1]
        positions = torch.arange(selected_ids.numel()).view(1, 1, -1).expand(3, 1, -1)
        return positions, torch.zeros(1)

    monkeypatch.setattr(
        MRotaryEmbedding,
        "get_rope_index_glm4v",
        MagicMock(side_effect=fake_get_rope_index_glm4v),
    )

    output = _run_processor(final_input_ids, processor_mask)

    assert observed_masks == [None]
    assert output.input_ids == final_input_ids
    assert output.mrope_positions.shape[-1] == len(final_input_ids)
