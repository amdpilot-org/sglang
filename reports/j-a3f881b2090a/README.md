# Abort/grammar overlap cancellation investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/34111

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1677

The tokenizer manager recorded cancellation in `ReqState.abort_sent`, but its
stream consumer did not use that state as a visibility boundary. A scheduler
output already queued when `/abort_request` arrived could therefore be yielded
before the later abort acknowledgement. The acknowledgement could also expose
the tokenizer's accumulated token IDs (and text when the tokenizer can decode
the sampled token).

The fix drops nonterminal stream output after cancellation has been recorded.
When the terminal scheduler response is an abort, it preserves the abort
metadata but clears text and output token IDs so buffered content does not
become a post-cancellation chunk. Non-streaming requests, uncancelled streams,
and a completion that wins the scheduler race are unchanged.

## Evidence

The focused regression failed before the source change because the queued
`"invalid"` chunk completed `__anext__()` immediately after cancellation. It
passes after the change, along with independent uncancelled and incremental
coalescing boundaries. The complete tokenizer-manager test file passes: 29
tests and 3 subtests.

The one-GPU probe used the deterministic fixture from amdpilot-org/sglang PR
649 at commit `f1d603677ca76a9ea21124a544e405c5b0cbd315`, generated privately at
`/tmp/amdpilot-repo-j-a3f881b2090a/tiny-random-llama`. Both runs used the real
source checkout, ROCm 7.2, Triton attention, and the assigned AMD Instinct
MI350X (`gfx950`). In `evidence/before/events.json`, the cancelled grammar
request emits an abort event after cancellation with `output_ids: [0]`. In
`evidence/fixed/events.json`, the corresponding event has `output_ids: []` and
empty text; the overlap peer continues to its length finish in both runs.

## Limitation

The qualified fixture is a random two-layer Llama with a 128-token WordLevel
vocabulary that has no JSON punctuation. XGrammar consequently logged
`Accepted tokens: []` for the requested JSON schema, and token 0 decodes to an
empty string. This GPU probe validates the transport/engine cancellation race
and the leaked token ID, but it cannot validate Llama-3.2-1B-Instruct grammar
semantics or reproduce the source report's visible invalid text. Those weights
and the reported NVIDIA configuration were unavailable.

## Reproduction

```bash
PY=/tmp/amdpilot-repo-j-a3f881b2090a/venv/bin/python
HIP_VISIBLE_DEVICES=0 $PY reports/j-a3f881b2090a/run_tiny_abort_probe.py \
  --fixture /tmp/amdpilot-repo-j-a3f881b2090a/tiny-random-llama \
  --output /tmp/j-a3f881b2090a-probe

$PY -m pytest -q \
  test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py
```
