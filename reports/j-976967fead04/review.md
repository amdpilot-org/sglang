# Independent review of PR 809

Upstream issue: https://github.com/sgl-project/sglang/issues/38450

Mirror issue: https://github.com/amdpilot-org/sglang/issues/854

Candidate: https://github.com/amdpilot-org/sglang/pull/809 at
`c7e1be173b74bcaeaf03845ccc0c10378fb13bde`.

## Recommendation

Accept as test-only hardening. The candidate adds a focused regression and does
not change production or native code. The production correction was already in
the recorded base, `358c163250ad3b1f62939b01ce1314a0a31a0365`.

This review does not mark the candidate itself as a full original-issue fix.
The candidate adds no fix, and the reported 4x H200 DeepSeek-V4-Flash-Vision
serving configuration and weights were unavailable. The deterministic encoder
root cause is reproduced at the reporter's commit and corrected on both the
prepared base and candidate tree; the full stochastic model behavior remains
unverified.

## Evidence

- At reporter image commit `40b3e15ddbd9a1067e181283d9900dd3f4d76ed7`,
  dict-normalized `{"filePath":"/etc/hosts"}` encoded as a DSML parameter
  named `arguments`, while the equivalent OpenAI JSON string encoded a
  `filePath` parameter. This reproduces the deterministic failure mechanism.
- At the prepared base, JSON-string arguments normalized by `serving_chat` and
  direct dict arguments encoded identically as a flat `filePath` parameter.
- The exact candidate test passed: 3 tests and 2 subtests.
- Four adjacent serving/DSV encoder tests passed: 4 tests and 6 subtests.
- Independent adversarial checks passed for nested objects, arrays, booleans,
  null, zero, empty strings, invalid top-level argument shapes, and a legitimate
  tool schema field literally named `arguments`.
- Imports resolved to the checkout sources at
  `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` and
  `python/sglang/srt/entrypoints/openai/serving_chat.py`.
- Candidate diff contains only one test and two report files. No native source
  changed, so a native rebuild was neither required nor performed.

Raw command output captured during revision switching is summarized in
`review_evidence.txt`; the original raw files remain outside the checkout under
`/tmp/amdpilot-repo-j-976967fead04/review-evidence/`.

## Environment limitations

The prepared interpreter used Python 3.12.3, Torch 2.11.0+rocm7.2, HIP 7.2,
and one AMD Instinct MI355X/gfx950. The issue reported four NVIDIA H200 GPUs,
CUDA 13, tensor parallelism 4, DSPARK, and DeepSeek-V4-Flash-Vision preview
weights. Those weights and that architecture were unavailable. No GPU execution
was needed for the deterministic prompt encoder checks, and no claim is made
about full-model semantic accuracy, stochastic tool-call frequency, tensor
parallelism, or the NVIDIA-only serving configuration.
