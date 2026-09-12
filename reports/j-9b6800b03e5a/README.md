# Independent review of PR 1244

Upstream issue: https://github.com/sgl-project/sglang/issues/36352

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1282

Candidate: https://github.com/amdpilot-org/sglang/pull/1244 at exact commit
`a98f1aec8abd104076f7d995fbf6d3f576414a20`.

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Recommendation

Accept. The candidate fully resolves the original issue's stated contract: an
explicit speculative algorithm combined with an effective `torch_native`
target attention backend is rejected in the real attention compatibility hook,
before model loading and warmup. This is a source fix with regression coverage,
not test-only hardening.

On the recorded base, an independent harness showed that the actual hook
accepted the reported NGRAM combination, EAGLE, and split prefill/decode
`torch_native` variants. At the exact candidate commit, all unsupported cases
raised the actionable `ValueError`. Controls remained valid: plain Torch Native
was accepted and disabled both CUDA-graph phases, NGRAM with Triton was
accepted, and a generic Torch Native value fully overridden by supported split
backends was accepted.

The candidate's focused regression passed (4 tests plus 2 split-backend
subtests). The broader server-argument file produced 228 passes and 46 passing
subtests. Its two failures are unrelated existing ROCm context-parallel
expectations: this HIP platform rejects deprecated prefill CP earlier than the
tests expect.

## Source and build verification

Fresh interpreter processes loaded `sglang` and `attention_hook.py` from
`/job/repo/python`, not an installed SGLang wheel. The candidate changes only
Python source, Python tests, and reports. It changes no native/FlyDSL source, so
no native rebuild was applicable. The prepared environment reports PyTorch
`2.11.0+rocm7.2`, HIP 7.2, and an assigned AMD Instinct MI355X (`gfx950`).

## Limitations

No production model weights were available, and the available architecture is
AMD gfx950 rather than the report's NVIDIA CUDA system. Therefore this review
does not claim a Qwen/NVIDIA full-server warmup reproduction or the later
`extend_prefix_lens` GPU traceback. No GPU kernel or model inference was run.
The deterministic tiny-Llama fixture would validate transport and engine
execution, but it would not add evidence to this pre-model-load validation
contract or qualify Qwen/NVIDIA behavior.

Complete command output was preserved outside revision switches under
`/job/review-evidence-j-9b6800b03e5a/`.
