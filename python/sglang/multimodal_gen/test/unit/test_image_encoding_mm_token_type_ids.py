import torch
from transformers.feature_extraction_utils import BatchFeature

from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
)


def _processor_output(*, include_mm_token_type_ids: bool) -> BatchFeature:
    data = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "pixel_values": torch.ones((1, 3, 2, 2)),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    if include_mm_token_type_ids:
        data["mm_token_type_ids"] = torch.tensor([[0, 1, 0]], dtype=torch.int32)
    return BatchFeature(data=data)


def test_text_encoder_inputs_forwards_mm_token_type_ids():
    image_inputs = _processor_output(include_mm_token_type_ids=True)

    encoder_inputs = ImageEncodingStage._text_encoder_inputs(image_inputs)

    assert encoder_inputs["mm_token_type_ids"] is image_inputs.mm_token_type_ids


def test_text_encoder_inputs_keeps_legacy_processor_compatible():
    image_inputs = _processor_output(include_mm_token_type_ids=False)

    encoder_inputs = ImageEncodingStage._text_encoder_inputs(image_inputs)

    assert set(encoder_inputs) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    }


def test_text_encoder_inputs_satisfies_qwen3_vl_multimodal_contract():
    image_inputs = _processor_output(include_mm_token_type_ids=True)

    def qwen3_vl_contract(*, image_grid_thw=None, mm_token_type_ids=None, **kwargs):
        del kwargs
        if image_grid_thw is not None and mm_token_type_ids is None:
            raise ValueError("mm_token_type_ids is missing")
        return mm_token_type_ids

    result = qwen3_vl_contract(
        **ImageEncodingStage._text_encoder_inputs(image_inputs)
    )

    assert result is image_inputs.mm_token_type_ids


def test_positive_and_negative_processor_outputs_are_built_independently():
    positive = _processor_output(include_mm_token_type_ids=True)
    negative = _processor_output(include_mm_token_type_ids=True)
    negative["mm_token_type_ids"] = torch.tensor([[1, 0, 1]], dtype=torch.int32)

    positive_inputs = ImageEncodingStage._text_encoder_inputs(positive)
    negative_inputs = ImageEncodingStage._text_encoder_inputs(negative)

    assert torch.equal(
        positive_inputs["mm_token_type_ids"],
        torch.tensor([[0, 1, 0]], dtype=torch.int32),
    )
    assert torch.equal(
        negative_inputs["mm_token_type_ids"],
        torch.tensor([[1, 0, 1]], dtype=torch.int32),
    )
