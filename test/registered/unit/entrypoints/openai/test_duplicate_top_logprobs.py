from sglang.srt.entrypoints.openai.protocol import LogProbs
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_responses import (
    _build_output_text_logprobs,
)
from sglang.srt.entrypoints.openai.utils import to_openai_style_logprobs


def _duplicate_logprobs():
    return {
        "output_token_logprobs": [(-0.1, 1, "A")],
        "output_top_logprobs": [
            [(-0.1, 1, "A"), (-0.5, 77, "�"), (-2.5, 88, "�")]
        ],
    }


def test_chat_top_logprobs_preserve_duplicate_decoded_text():
    logprobs = to_openai_style_logprobs(**_duplicate_logprobs())

    content = OpenAIServingChat._process_logprobs_tokens(
        None, logprobs, use_token_index=True
    )

    assert [(item.token, item.logprob) for item in content[0].top_logprobs] == [
        ("A", -0.1),
        ("�", -0.5),
        ("�", -2.5),
    ]
    # The legacy text-keyed projection cannot contain both entries, but must
    # retain the better-ranked candidate instead of overwriting it.
    assert logprobs.top_logprobs == [{"A": -0.1, "�": -0.5}]


def test_responses_top_logprobs_preserve_duplicate_decoded_text():
    content = _build_output_text_logprobs(_duplicate_logprobs())

    assert [(item.token, item.logprob) for item in content[0].top_logprobs] == [
        ("A", -0.1),
        ("�", -0.5),
        ("�", -2.5),
    ]


def test_top_logprobs_preserve_unique_and_missing_positions():
    logprobs = to_openai_style_logprobs(
        output_token_logprobs=[(-0.1, 1, "A"), (-0.2, 2, "B")],
        output_top_logprobs=[[(-0.1, 1, "A"), (-0.3, 3, "C")], None],
    )

    assert logprobs.top_logprobs == [{"A": -0.1, "C": -0.3}, None]
    assert logprobs.top_logprobs_raw == [
        [(-0.1, 1, "A"), (-0.3, 3, "C")],
        None,
    ]
    assert "top_logprobs_raw" not in logprobs.model_dump()

    content = OpenAIServingChat._process_logprobs_tokens(
        None, logprobs, use_token_index=True
    )
    assert [(item.token, item.logprob) for item in content[0].top_logprobs] == [
        ("A", -0.1),
        ("C", -0.3),
    ]
    assert content[1].top_logprobs == []


def test_chat_accepts_legacy_logprobs_without_raw_candidates():
    logprobs = LogProbs(
        tokens=["A"], token_logprobs=[-0.1], top_logprobs=[{"A": -0.1}]
    )

    content = OpenAIServingChat._process_logprobs_tokens(
        None, logprobs, use_token_index=True
    )

    assert [(item.token, item.logprob) for item in content[0].top_logprobs] == [
        ("A", -0.1)
    ]
