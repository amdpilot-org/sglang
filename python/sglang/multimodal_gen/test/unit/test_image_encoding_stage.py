# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import BatchFeature

from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
)


class _ImageProcessor:
    def __init__(self, include_mm_token_type_ids):
        self.include_mm_token_type_ids = include_mm_token_type_ids
        self.called = False

    def __call__(self, images, return_tensors, text=None, padding=None):
        del images, return_tensors, padding
        self.called = True
        inputs = BatchFeature(
            data={
                "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
                "attention_mask": torch.ones((1, 7), dtype=torch.long),
                "pixel_values": torch.ones((1, 3, 2, 2)),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        )
        if self.include_mm_token_type_ids:
            marker = 3 if text == ["negative"] else 1
            inputs["mm_token_type_ids"] = torch.tensor(
                [[0, marker, marker, marker, marker, 0, 0]],
                dtype=torch.long,
            )
        return inputs


class _CapturingTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        output_hidden_states,
        use_cache,
        mm_token_type_ids=None,
    ):
        assert output_hidden_states is True
        assert use_cache is False
        self.calls.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "mm_token_type_ids": mm_token_type_ids,
            }
        )
        hidden_states = torch.zeros((*input_ids.shape, 4))
        return SimpleNamespace(hidden_states=[hidden_states])


class _StrictTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.called = False

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        output_hidden_states,
        use_cache,
    ):
        del attention_mask, pixel_values, image_grid_thw
        assert output_hidden_states is True
        assert use_cache is False
        self.called = True
        hidden_states = torch.zeros((*input_ids.shape, 4))
        return SimpleNamespace(hidden_states=[hidden_states])


def _make_server_args():
    def prepare_image_processor_kwargs(batch, neg=False):
        del batch
        text = "negative" if neg else "positive"
        return {
            "padding": True,
            "per_prompt_images": [[object()]],
            "text": [text],
        }

    pipeline_config = SimpleNamespace(
        image_encoder_extra_args={},
        postprocess_text_funcs=(lambda outputs, _inputs: outputs.hidden_states[-1],),
        prepare_image_processor_kwargs=prepare_image_processor_kwargs,
    )
    return SimpleNamespace(
        component_precisions={},
        pipeline_config=pipeline_config,
    )


def _make_batch(do_classifier_free_guidance):
    return SimpleNamespace(
        condition_image=object(),
        do_classifier_free_guidance=do_classifier_free_guidance,
        image_embeds=[],
        prompt_embeds=[],
        negative_prompt_embeds=[],
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        prompt_seq_lens=None,
        negative_prompt_seq_lens=None,
    )


def _assert_image_alignment(call):
    token_type_ids = call["mm_token_type_ids"]
    assert token_type_ids is not None
    assert token_type_ids.dtype == torch.long
    assert token_type_ids.device == torch.device("cpu")
    assert token_type_ids.shape == call["input_ids"].shape
    assert call["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert int((token_type_ids != 0).sum()) == int(call["image_grid_thw"].prod())


@pytest.mark.parametrize("do_classifier_free_guidance", [False, True])
def test_forwards_mm_token_type_ids_to_image_edit_text_encoder(
    do_classifier_free_guidance,
):
    text_encoder = _CapturingTextEncoder()
    stage = ImageEncodingStage(
        image_processor=_ImageProcessor(include_mm_token_type_ids=True),
        text_encoder=text_encoder,
    )
    server_args = _make_server_args()
    stage.server_args = server_args

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding.get_local_torch_device",
        return_value=torch.device("cpu"),
    ):
        stage.forward(
            _make_batch(do_classifier_free_guidance),
            server_args,
        )

    _assert_image_alignment(text_encoder.calls[0])
    assert torch.equal(
        text_encoder.calls[0]["mm_token_type_ids"],
        torch.tensor([[0, 1, 1, 1, 1, 0, 0]], dtype=torch.long),
    )
    if do_classifier_free_guidance:
        _assert_image_alignment(text_encoder.calls[1])
        assert torch.equal(
            text_encoder.calls[1]["mm_token_type_ids"],
            torch.tensor([[0, 3, 3, 3, 3, 0, 0]], dtype=torch.long),
        )


def test_omits_mm_token_type_ids_when_processor_does_not_return_them():
    text_encoder = _StrictTextEncoder()
    stage = ImageEncodingStage(
        image_processor=_ImageProcessor(include_mm_token_type_ids=False),
        text_encoder=text_encoder,
    )
    server_args = _make_server_args()
    stage.server_args = server_args

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding.get_local_torch_device",
        return_value=torch.device("cpu"),
    ):
        stage.forward(_make_batch(False), server_args)

    assert text_encoder.called


def test_text_only_request_returns_without_encoding():
    image_processor = _ImageProcessor(include_mm_token_type_ids=True)
    text_encoder = _CapturingTextEncoder()
    stage = ImageEncodingStage(
        image_processor=image_processor,
        text_encoder=text_encoder,
    )
    batch = _make_batch(False)
    batch.condition_image = None

    returned = stage.forward(batch, _make_server_args())

    assert returned is batch
    assert image_processor.called is False
    assert text_encoder.calls == []
