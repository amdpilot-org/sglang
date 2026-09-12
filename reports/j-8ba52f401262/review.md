# Independent review of PR 3016

Candidate: https://github.com/amdpilot-org/sglang/pull/3016

Exact candidate commit: `4c7768d36ecfa2fb078ad51cfed49932677dfda0`

Upstream issue: https://github.com/sgl-project/sglang/issues/32124

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2960

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3053

## Recommendation

Request changes. The candidate is a partial fix. Its tests demonstrate marker creation, exact payload matching, conservative fallback, successful same-path reuse, and coverage of the previously reported sibling-source/native-artifact identity counterexample. However, it does not fulfill the original cache-transfer contract when a compatible cache is restored at a different absolute path.

## Blocking counterexample

`_warmup_marker_payload()` includes every `DG_JIT_*` environment variable in the compatibility settings. That includes `DG_JIT_CACHE_DIR`, which is the storage location rather than a property of the generated kernels. Consequently, moving or copying an otherwise identical cache changes the payload and marker digest.

The independent adversarial reproduction generated a completed marker in `baked-image-cache`, copied the entire cache to `runtime-restored-cache`, cleared process-local initialization state, and invoked the same kernel warmup under the same mocked implementation identity and shapes. The candidate performed the exhaustive warmup a second time (`warmup_calls_after_relocated_restore=2`) and created a second marker. This conflicts with the issue's explicit cache-transfer use case.

The correction should exclude location-only values such as `DG_JIT_CACHE_DIR` from compatibility identity while retaining settings that affect generated code. It should add a regression that copies the cache to a different directory and proves the marker is reused there.

## Evidence

- Recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`: two simulated processes execute warmup twice and no completion marker exists. This reproduces the redundant-warmup behavior at the actual call boundary.
- Exact candidate: its complete committed regression file passes, 12 tests.
- Exact candidate: the independent relocated-cache adversarial test fails because warmup executes twice.
- Candidate source imports came from `/job/repo/python/sglang`; Torch imported from `/opt/venv/lib/python3.12/site-packages/torch`.

## Architecture and runtime limitations

The assigned GPU is an AMD Instinct MI355X with Torch `2.11.0+rocm7.2`; `torch.version.cuda` is `None`, and `deep_gemm` is not installed. Therefore NVIDIA DeepGEMM JIT compilation, real cache restoration, GPU numerical parity, CUDA graph parity, DeepSeek-V4-Pro TP=8 behavior, and startup performance remain unverified. No native source changed in the candidate, so no native rebuild was applicable. No GPU kernel was executed as review evidence.
