# TP8 long-generation A/B for issue 35122

## Status

**Semantic A/B: FAILED.** The isolated PR60 FP8 software-conversion patch did not turn the failed long-generation control into coherent, correct, repeatable, EOS-completed generation. See `A_B_RESULT.md` for the full measured result and limits.

## Identities

- Control: detached `amdpilot-org/sglang` at `484c2286c993d36e862343c390a77439a003d244`, clean.
- Candidate: detached worktree at the same commit plus only `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp8_utils.cuh`.
- Baseline header SHA256: `6b686494e4ee7ac6f12f972e1033837ff020d1629e833e06f57f304ed0c5f022`.
- Patched header SHA256: `16ee2f59330081e0015390d4f895cbd7aa093b55ace86723a668699b130ff375`.
- Delivery branch: `amdpilot/j-2dd26df9bdab`, cut from `main`.

## Isolation

Each arm gets a separate initially empty `SGLANG_JIT_CACHE_DIR`, `TRITON_CACHE_DIR`, and native build cache. Arms run sequentially in fresh Python/server processes with the same explicit production environment and numerical settings. The control runs first and must fully exit before the candidate starts.

## Primary acceptance

For each arm, run the four fixed cases twice through `/v1/chat/completions` and twice through `/generate`, using that arm's official DSV4 encoder. Sampling is temperature `0`, top-p `1`, seed `12345`, max tokens `256`, and `ignore_eos=false`. A response is complete only with genuine EOS and the required coherent, factually correct output. Repeated requests must match exactly. A `finish_reason=length` result is incomplete.

The primary sample is deliberately small and declared. It is not a 200-question or upstream-score result.

## Preserved evidence

Raw requests, responses, token IDs, logs, process identity, imported module paths, private cache contents, compiler commands, loaded native `.so` paths and hashes, and per-request flushed evidence are collected under `/job`. A top-level recovery patch is updated at each milestone.
