# Independent review of PR 1833

Reviewed candidate: `878c921ae87b843d07349a04635933e92840a998`

Upstream issue: https://github.com/sgl-project/sglang/issues/33603

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1870

## Verdict

Recommendation: **accept**.

The recorded base reproduces the backend defect: its mask helper has no
`causal` argument and produces a causal-only mask when compared with the
issue's bidirectional contract. In a deterministic seven-token attention
fixture, using that mask differs from an independent bidirectional
masked-softmax result by a maximum absolute error of `1.6200188398361206`.

At the exact candidate commit, the runtime imported
`/job/repo/python/sglang/srt/layers/attention/torch_native_backend.py`. The
candidate adds the non-causal mask branch, passes `causal` through both SDPA
helpers, and no longer discards a sliding window for encoder-only
`forward_extend`. The candidate's four focused regressions passed.

Independent cases covered radius one, radius two, a non-square query with a
query offset, zero width, an oversized window, and preservation of causal
offset behavior. All matched a separately constructed positional reference.
CPU SDPA matched an explicit masked-softmax oracle within
`1.1920928955078125e-07`. On the assigned single gfx950 GPU (reported by Torch
as AMD Instinct MI350X), a 257-token/window-128 case matched within
`3.5762786865234375e-07`.

No native files changed. The prepared environment declares no native rebuild
recipe, and the changed runtime path is Python source, so `native_rebuilt` is
false and no rebuild was applicable.

## Scope and limitations

The candidate fully repairs the isolated backend contract and no remaining
backend counterexample was found. However, `fully_resolves_original` is
reported as false because the selected repository does not contain the
ModernBERT implementation referenced by the report and the
`jhu-clsp/ettin-encoder-17m` weights were not prepared. Therefore the original
HTTP serving and HuggingFace embedding-parity reproduction could not be run.
The GPU fixture validates attention numerics only, not ModernBERT semantic
accuracy, HTTP transport, or a distributed workload. The tiny Llama fixture
was intentionally not substituted because it cannot qualify encoder-only
bidirectional attention.
