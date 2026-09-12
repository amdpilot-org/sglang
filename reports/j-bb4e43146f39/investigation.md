# Investigation of sglang#37052

Upstream issue: https://github.com/sgl-project/sglang/issues/37052

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1029

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Outcome

`candidate_rejected`: no source correction is justified by the available evidence.

The issue's proposed `full` decode CUDA-graph row-sizing/aliasing analogy to PR
#36749 does not match the current implementation:

- `DecodeCudaGraphRunner` computes target-verify capture rows as
  `num_tokens = batch_size * captured_req_width`.
- `FullCudaGraphBackend` retains the output produced inside each captured graph. It
  does not use the breakable backend's request-count clamp.
- The only shared output buffer in `FullCudaGraphBackend` is enabled by the prefill
  backend construction, not the decode backend.

The upstream reporter also added later evidence to #37052 that the same
invalid-probability failure recurred with CUDA graphs and overlap scheduling both
disabled. Consequently, the original graph-only causal inference is no longer
supported. The remaining suspected area spans long re-prefill/speculative decode,
ReplaySSM, recurrent GDN/Mamba state, QSA, and ModelOpt NVFP4; the first corrupting
operation remains unknown.

## Reproduction and boundaries

The prepared host provides one AMD Instinct MI350X (`gfx950`) with ROCm 7.2. It does
not provide the reported two NVIDIA GB10 (`sm_121`) nodes, CUDA 13, the
Qwen3.8-Flash-Next-NVFP4 weights, ModelOpt FP4 kernels, RoCE TP2 topology, or the
production request stream. Therefore the original serving failure was not
reproduced, and this report does not claim model, architecture, semantic, or
distributed validation.

A deterministic real-GPU boundary fixture checked the narrow row-integrity
hypothesis using HIP graphs and an eager numerical reference. For draft width 4 and
batch sizes 1, 2, and 8, graph output shapes were respectively `[4, 7]`, `[8, 7]`,
and `[32, 7]`; all values were finite and nonnegative. The graph/eager maximum
absolute error was `0.0` in every case. Probability row-sum maximum absolute error
was `0.0`, `0.0`, and `1.1920928955078125e-07`.

This fixture validates graph transport of token-shaped rows on gfx950 only. It is
not a substitute for the missing Qwen/GB10 workload.

## Commands

```bash
/tmp/amdpilot-repo-j-bb4e43146f39/venv/bin/python -m pytest -q \
  test/registered/unit/model_executor/runner_backend/test_full_cuda_graph_backend.py

HIP_VISIBLE_DEVICES=0 \
  /tmp/amdpilot-repo-j-bb4e43146f39/venv/bin/python \
  reports/j-bb4e43146f39/check_full_graph_rows.py
```

Raw measured output is retained in `pytest_full_backend.txt` and
`full_graph_rows_output.json`.
