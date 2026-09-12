# Investigation of sglang issue 29960

Upstream issue: https://github.com/sgl-project/sglang/issues/29960

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2405

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The affected production setup cannot be reproduced on the assigned hardware. The report is specifically about NVIDIA `cudaGraphLaunch` host time on eight H200 GPUs, CUDA 13.0, NCCL TP=8, and a GLM-5.1 FP8 model with EAGLE/MTP. The prepared environment exposes one AMD Instinct MI355X (`gfx950`) through PyTorch ROCm 7.2, and the model weights are absent.

Current main does still have the implementation shape described by the report:

- `FullCudaGraphBackend.capture_one` creates one `torch.cuda.CUDAGraph` and stores it in `_graphs[shape_key]`.
- `FullCudaGraphBackend.replay` invokes `_graphs[shape_key].replay()` exactly once.
- `DecodeCudaGraphRunner` calls `backend.capture_one(...)` once for a shape and `backend.replay(...)` once during execution.
- `cuda_graph_config.py` permits `backend`, `max_bs`, `bs`, and `tc_compiler` for decode. It has no `split_nlayers` setting and defines no layer-split full backend.

Thus, the source does not already contain the proposed solution. GitHub inspection found the upstream issue still open. Searches for a PR or current source containing `split_nlayers` returned no result. The only PR returned by a broad issue-number search was unrelated PR 30027, whose body happened to reference the issue number.

## Why no speculative patch was made

The proposed change is not a local replay-loop tweak. A correct implementation must split the model forward at a layer boundary and preserve model-specific bridge state, including residuals and DSA top-k indices, while ensuring attention metadata initializes only in the first graph. It must then demonstrate that two host launches overlap useful GPU work and reduce exposed scheduler latency without changing GLM-5.1/MTP output.

The available AMD single-GPU fixture cannot supply that evidence. Adding a configuration field or replaying an arbitrary graph twice would only test a neighboring mechanism and could introduce an unqualified model-wide execution path. Accordingly, the outcome is `unsupported_architecture`, not fixed or reproduced.

## Evidence retained

- `source_evidence.txt`: relevant source locations and current GitHub search results.
- `pytest.txt`: focused existing unit-test result (5 passed).
- `gpu_graph_check.py` and `gpu_graph_check.txt`: deterministic single-gfx950 graph execution with an independent numerical reference and a mutated-input boundary case.

The GPU fixture validates that graph capture/replay is functional in this ROCm environment only. It is not evidence about CUDA Graph launch cost, H200 behavior, GLM semantics, EAGLE/MTP target verification, or TP=8/NCCL execution.
