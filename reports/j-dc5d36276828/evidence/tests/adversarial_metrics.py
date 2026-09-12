import json

from sglang.benchmark.serving import (
    RequestFuncOutput,
    calculate_metrics,
    format_metric_scope_note,
)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


# Ten output tokens: first token at 4.0 s, nine decode tokens over 0.9 s.
# Only eight stream-event intervals are present, leaving a 0.1 s terminal tail.
output = RequestFuncOutput(
    success=True,
    latency=4.9,
    ttft=4.0,
    output_len=10,
    generated_text=" ".join(["x"] * 10),
    itl=[0.1] * 8,
    start_time=100.0,
)
metrics, _ = calculate_metrics(None, [output], 4.9, Tokenizer(), "sglang")
note = format_metric_scope_note()
observed = {
    "note": note,
    "mean_tpot_ms": metrics.mean_tpot_ms,
    "whole_request_per_output_token_ms": output.latency / output.output_len * 1000,
    "decode_phase_per_nonfirst_token_ms": (
        (output.latency - output.ttft) / (output.output_len - 1) * 1000
    ),
    "itl_sample_count": len(output.itl),
    "nonfirst_output_token_count": output.output_len - 1,
    "itl_sum_s": sum(output.itl),
    "decode_phase_s": output.latency - output.ttft,
    "whole_run_output_throughput_tok_s": metrics.output_throughput,
    "synthetic_recent_window_throughput_tok_s": 10 / 0.5,
}
print(json.dumps(observed, indent=2))

assert abs(metrics.mean_tpot_ms - 100.0) < 1e-9
assert abs(observed["whole_request_per_output_token_ms"] - 490.0) < 1e-9
assert observed["itl_sample_count"] == 8
assert observed["nonfirst_output_token_count"] == 9
assert abs(observed["itl_sum_s"] - 0.8) < 1e-12
assert abs(observed["decode_phase_s"] - 0.9) < 1e-12
assert observed["whole_run_output_throughput_tok_s"] != observed[
    "synthetic_recent_window_throughput_tok_s"
]
assert "after the first token" in note
assert "stream events" in note
assert "not expected to match exactly" in note
