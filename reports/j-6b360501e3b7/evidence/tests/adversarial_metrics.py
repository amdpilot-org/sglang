import json
from sglang.benchmark.serving import RequestFuncOutput, calculate_metrics, format_metric_scope_note

class Tok:
    def encode(self, text, add_special_tokens=False):
        return text.split()

# 10-token response, 4s TTFT, then 9 tokens over 0.9s. Whole request is 4.9s,
# while TPOT intentionally measures only post-first-token decode time.
out = RequestFuncOutput(success=True, latency=4.9, ttft=4.0, output_len=10,
                        generated_text=' '.join(['x']*10), itl=[0.1]*8,
                        start_time=100.0)
metrics, _ = calculate_metrics(None, [out], 4.9, Tok(), 'sglang')
note = format_metric_scope_note()
result = {
  'note': note,
  'mean_tpot_ms': metrics.mean_tpot_ms,
  'whole_request_per_output_token_ms': out.latency / out.output_len * 1000,
  'decode_phase_per_nonfirst_token_ms': (out.latency-out.ttft)/(out.output_len-1)*1000,
  'itl_sum_s': sum(out.itl),
  'decode_phase_s': out.latency-out.ttft,
  'output_throughput_tok_s': metrics.output_throughput,
  'tpot_inverse_times_configured_concurrency_tok_s': 4/(metrics.mean_tpot_ms/1000),
}
print(json.dumps(result, indent=2))
assert abs(metrics.mean_tpot_ms - 100.0) < 1e-9
assert abs(result['whole_request_per_output_token_ms'] - 490.0) < 1e-9
assert abs(result['itl_sum_s'] - 0.8) < 1e-9 and abs(result['decode_phase_s'] - 0.9) < 1e-9
assert 'whole request lifetime' in note
