# Independent review evidence

Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/1732 at
`084b94f67bf37874a070e7e36199857bde662f25`

Upstream issue: https://github.com/sgl-project/sglang/issues/34740

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1768

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Result

Request changes. The candidate fixes Defect A and bounds the retained decode
window in token space, including when one event contains 4,096 invalid-byte
tokens. It does not fully implement the original byte-completeness contract.

The forced recovery commits the entire suffix once 64 stalled tokens have
accumulated and moves `surr_offset` to the end. If token 64 is the first byte of
a valid UTF-8 sequence, the following event can no longer complete it. With the
deterministic byte tokenizer:

```text
event 1: 63 * 0x80, then 0xE4
event 2: 0xB8, 0xAD
expected suffix: '���中'
candidate suffix: '����'
text equal: false
```

This is a direct boundary case for the original requirement to distinguish an
actual incomplete byte suffix from a complete U+FFFD character. The candidate
achieves a strict performance bound by finalizing a possibly incomplete suffix,
which can corrupt otherwise valid output spanning the recovery boundary.

## Reproduction and regression

On the recorded base, 256 four-token invalid-byte events left a 1,024-token
live decode window, and 32 events of 4,096 tokens left a 131,072-token window;
neither emitted nor committed text. This independently reproduces the original
unbounded-window behavior. Split bytes `E4 B8 AD` still decoded to `中`, showing
the reason an incomplete suffix must normally be retained.

At the exact candidate, its focused regression plus existing stop-trimming
coverage passed (`15 passed`). The same invalid streams emitted replacement
text and ended with a zero-token live window, confirming that the performance
bound is effective. The independent boundary case above nevertheless loses the
valid `中` character.

On one assigned AMD Instinct MI350X (`gfx950`) with Torch
`2.11.0+rocm7.2`, the candidate imported
`/job/repo/python/sglang/srt/speculative/spec_utils.py`, resolved fixture token
ID 64 (`a` rather than U+FFFD), and the synchronized device tensor path produced
`predict=[64, 64, 64, 64]`. This verifies the fixed-token control/tensor path,
not model serving.

Raw logs and the independent fixtures are preserved outside the checkout at
`/job/review-evidence-j-94b50c1d11a1/`.

## Limitations

DeepSeek-V4-Pro weights and tokenizer were unavailable. The reported eight-GPU
TP/DP/EAGLE workload, semantic output, throughput, TTFT, and distributed routing
were not reproduced. The available single gfx950 GPU only qualified the focused
tensor operation. No native source changed, `repository-environment.json`
declares `native: null`, and no native rebuild was applicable.
