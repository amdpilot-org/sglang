# Investigation of sgl-project/sglang#30197

Upstream issue: https://github.com/sgl-project/sglang/issues/30197

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2389

## Result

The original failure was not reproducible in the prepared environment. The report describes GLM-5.2-w4afp8 on 32 NVIDIA H20 GPUs in a 2-prefill/2-decode NIXL deployment, but this job has one AMD Instinct MI350X (gfx950) and no GLM-5.2 weights. The upstream issue also contains no traceback or comments: its only failure text is Python resource-tracker cleanup warnings after the workers have already exited. Those warnings do not identify the initiating exception.

No source correction is justified from the available evidence.

## Current-source behavior

The reported prefill command uses the former options:

```
--enable-nsa-prefill-context-parallel
--nsa-prefill-cp-mode round-robin-split
```

At base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`, these options are deliberately not part of the public CLI. Existing regression `TestContextParallelServerArgs.test_generic_v1_cp_options_are_not_public_cli` verifies rejection of both options and four adjacent former CP spellings. It passed with all six subtests. Current GLM-5.2 CUDA documentation instead uses:

```
--attn-cp-size 8
--enable-prefill-cp
--cp-strategy interleave
```

and explicitly requires `--dp 1`. This is evidence that the reported 0.5.14 command is obsolete in the current implementation, but it is not proof that the original 32-H20 deployment bug is fixed: that needs the reported architecture, weights, topology, and the missing initiating traceback.

On the assigned HIP platform, current source explicitly rejects canonical prefill CP with `Prefill CP on HIP/NPU/MUSA is deprecated; CP support will be refactored soon.` Therefore an AMD substitute cannot qualify the H20 CUDA path.

## Evidence

- `raw/legacy_cli_rejection.log`: issue-specific CLI boundary regression; exit 0, one test and six subtests passed.
- `raw/context_parallel_unit.log`: broader class; two issue-relevant boundary tests passed, while two generic tests failed because the prepared HIP platform reaches the intentional HIP rejection before their CUDA-oriented assertions. This was not treated as reproduction of the upstream issue.
- `raw/gpu_environment.log`: PyTorch 2.11.0+rocm7.2 saw one AMD Instinct MI350X. A seeded 128x96 by 96x80 FP32 GPU matmul agreed with the CPU result (`max_abs_error=9.5367431640625e-06`, `allclose` at rtol/atol 1e-4). This confirms assigned-GPU execution only; it does not validate GLM-5.2, W4AFP8, NIXL, speculative decoding, or distributed behavior.

## Remaining limitations

- No NVIDIA H20 GPU or 32-GPU / multi-node topology.
- No GLM-5.2-w4afp8 or EAGLE draft weights.
- No NIXL prefill/decode deployment reproduction.
- The upstream report omits the exception preceding worker shutdown, so the resource-tracker warnings alone cannot localize a defect.
- The deterministic tiny Llama fixture cannot qualify GLM-5.2 architecture, W4AFP8 kernels, or a distributed NIXL workload, so it was not substituted for the requested reproduction.
