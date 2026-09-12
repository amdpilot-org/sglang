# Independent review of amdpilot-org/sglang#530

Reviewed candidate commit `f5bffd783f1b1a9d1b2006fe5b4be4c1db960ae2` against base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the contract in:

- Upstream issue: https://github.com/sgl-project/sglang/issues/39092
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/663

## Recommendation

Accept. The candidate fully resolves the trace cursor/misattribution defect at the tested graph/profiler boundary. It binds each profiler flush to the exact immutable `ShapeKey` supplied to `FullCudaGraphBackend.capture_one`, and includes batch size, PDMUX stream index, LoRA variant, and attention/DSA variant in the filename when present.

## Independent evidence

On the untouched base, a genuine ROCm `torch.cuda.CUDAGraph` plus scheduled `torch.profiler` run failed on its third capture with the reported `IndexError`. Two files existed before failure; the second was named `bs_7` even though capture order showed it was the second `bs=11` variant. See `raw/base-gpu-repro.log`.

At the exact candidate commit, imports resolved to `/job/repo/python/sglang/...`, not a stale installed wheel. The candidate's focused tests passed (18 tests). A real GPU run then captured and replayed 16 graphs spanning two batch sizes, two stream IDs, LoRA/no-LoRA, and dense/sparse attention. It emitted 16 unique traces with zero replay error. An independent parser matched every filename identity to its expected `ProfilerStep#(3j+2)` and confirmed kernel events were present, rather than trusting filenames or startup success.

The source change is Python-only (`decode_cuda_graph_runner.py` and `full_cuda_graph_backend.py`); no C++, FlyDSL, or other native source changed, so no native rebuild was applicable.

## Limitations

- Hardware was one AMD Instinct MI350X (`gfx950`) with ROCm 7.2 and Torch `2.11.0+rocm7.2`, not the reporter's MI355X.
- The full `amd/GLM-5.2-MXFP4` TP=8 server was not launched. The test exercised the same runner/backend profiler callback and real GPU graph capture with synthetic numerical forwards.
- PDMUX is NVIDIA-only in production. Two separate profiler context entries and two stream identities were exercised on ROCm, proving collision-free naming and queue continuity, but the production NVIDIA PDMUX stack was not available.
- No remaining counterexample was found within the approved contract. The queue assumes the current synchronous profiler callback behavior; the real profiler run verified that behavior in the prepared Torch build.

## Commands

```text
/tmp/amdpilot-repo-j-9a024001c2c2/venv/bin/python /job/review-evidence-j-9a024001c2c2/candidate_gpu_repro.py
/tmp/amdpilot-repo-j-9a024001c2c2/venv/bin/python -m pytest -q test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py test/registered/unit/model_executor/runner_backend/test_full_cuda_graph_backend.py
/tmp/amdpilot-repo-j-9a024001c2c2/venv/bin/python /job/review-evidence-j-9a024001c2c2/verify_trace_attribution.py
```

Raw command output is retained in `raw/`; the reviewed source remains available at the exact candidate commit above.
