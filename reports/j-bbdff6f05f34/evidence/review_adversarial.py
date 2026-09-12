import json
import sys

sys.path.insert(0, "/job/repo/test/registered/unit/entrypoints/openai")

from test_serving_completions import ServingCompletionTestCase

from sglang.srt.entrypoints.openai.protocol import CompletionRequest


def _ret(idx, ttft):
    return {
        "text": f"answer-{idx}",
        "meta_info": {
            "id": f"cmpl-{idx}",
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "cached_tokens": 0,
            "finish_reason": {"type": "stop"},
            "weight_version": "default",
            "time_to_first_token": ttft,
            "generation_time": 0.5,
            "e2e_latency": ttft + 0.5,
            "mean_inter_token_latency": 0.5,
            "output_token_throughput": 2.0,
        },
    }


def _serving():
    case = ServingCompletionTestCase(methodName="runTest")
    case.setUp()
    return case.sc


def test_batched_prompts_with_n_one_keep_metrics_for_every_output():
    """Two prompts and n=1 still produce two output choices."""
    request = CompletionRequest(
        model="x",
        prompt=["first", "second"],
        max_tokens=2,
        n=1,
        return_request_metrics=True,
    )
    response = _serving()._build_completion_response(
        request, [_ret(0, 0.1), _ret(1, 0.2)], 123
    )
    metrics = response.model_dump()["sglext"]["request_metrics"]
    assert isinstance(metrics, list), metrics
    assert [m["time_to_first_token"] for m in metrics] == [0.1, 0.2]


def test_opt_out_response_schema_remains_unchanged():
    request = CompletionRequest(model="x", prompt="first", max_tokens=2)
    response = _serving()._build_completion_response(request, [_ret(0, 0.1)], 123)
    dumped = response.model_dump()
    assert "sglext" not in dumped
    assert "request_metrics" not in json.dumps(dumped)
