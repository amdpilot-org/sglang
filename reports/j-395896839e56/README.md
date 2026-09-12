# Independent review of PR 3493

Candidate reviewed: `5334091df7b584723e28e1e114bff123d4c31d88`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Upstream issue: https://github.com/sgl-project/sglang/issues/33627

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3490

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3496

## Verdict

Recommendation: **accept**, specifically as test-only hardening of an opt-in
implementation already present in the recorded base. The candidate changes no
runtime or native source. It does not by itself fully resolve the original open
feature question.

The candidate's new regressions are relevant and pass. They verify that a
direct FP32-output dense GEMM preserves a constructed top-1 distinction lost by
BF16 output rounding, that a mocked standard TP gather receives FP32, and that
FP32 bypasses the BF16-only multimem wrapper to a mocked generic gather without
casting. Source inspection confirms that attention-TP gather, TP all-to-all,
and DP scatter allocate from the input/logits dtype.

The same candidate test file passes against the recorded base source (9 tests),
which confirms this is not a failing-before/passing-after runtime fix. The base
already had six focused tests and the opt-in `torch.mm(...,
out_dtype=torch.float32)` implementation. The original default BF16 path still
reproduces the reported rounding mechanism by design; the opt-in path avoids
it.

## Independent GPU evidence

On the assigned AMD Instinct MI350X with Torch `2.11.0+rocm7.2` and HIP
`7.2.26015`, a constructed BF16-input case produced reference logits
`[64.0, 64.0078125]`. The default BF16-output/post-cast path produced
`[64.0, 64.0]` and chose token 0, while direct FP32 output exactly matched the
reference and chose token 1.

For a separate random `M=17, K=4096, N=8192` case, concatenating four
single-device shard GEMMs had zero top-1 mismatches versus one full GEMM, but
was not bitwise equal (maximum delta `0.0001010894775390625`). This is only
arithmetic evidence and is not a live TP4 run.

Rerunning the candidate GPU script on this MI350X reproduced its qualitative
results: direct FP32-output maximum absolute error versus explicit FP32-input
reference was `6.87e-5` to `7.93e-4`, versus `0.499` to `0.980` after BF16
output rounding. At `M=128`, the local direct path was slightly slower than the
post-cast path in this run, so these single-GPU timings do not establish the
distributed performance tradeoff.

## Paths and rebuild status

- SGLang import: `/job/repo/python/sglang/__init__.py`
- Logits processor import: `/job/repo/python/sglang/srt/layers/logits_processor.py`
- Torch import: `/opt/venv/lib/python3.12/site-packages/torch/__init__.py`
- Interpreter: `/tmp/amdpilot-repo-j-395896839e56/venv/bin/python`
- Preserved raw review evidence: `/job/review-evidence-j-395896839e56/`
- Candidate production/native delta: none
- Native rebuild: not applicable and not performed

## Remaining limitations

- One assigned GPU; no live TP2/TP4 RCCL/NCCL collective.
- No actual multimem execution. MI350X/ROCm does not qualify the reported
  B300/CUDA multimem environment.
- No GLM-5.2, DeepSeek V4, or other representative model weights, semantic
  accuracy run, or end-to-end benchmark.
- No distributed gather performance measurement and no compiler/ISA claim.
- Quantized and architecture-specific LM-head implementations were not
  qualified by the dense-GEMM regressions.
- The default remains disabled. The candidate reasonably preserves it, but the
  original issue's default-policy choice remains open pending representative
  distributed benchmarks.
