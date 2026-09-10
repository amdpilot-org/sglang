# MI300X target-token sampling investigation

## Scope and upstream context

- This investigation covers seeded target-token sampling, not speculative acceptance ratios.
- Upstream issue 30344 is the DSpark roadmap tracker. Its current description and comments do not describe this sampler defect.
- Merged upstream PR 33423 already fixes the `u == 1` Gumbel endpoint and is present in both tested sources. It was not duplicated.
- Open upstream PR 35697 changes rank-space versus vocab-ID noise after filtering. It is related but does not address Inductor inlining of the Triton hash launcher, so it was not used as the fix.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`
- Python: `/opt/venv/bin/python` (3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Installed source: `/sgl-workspace/sglang/python`, commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed sampler: `/sgl-workspace/sglang/python/sglang/srt/layers/sampler.py`
- Installed Triton hash: `/sgl-workspace/sglang/python/sglang/kernels/ops/sampling/murmur_hash.py`
- Installed native package: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`, version `0.4.6.post1`
- Persistent checkout: `/job/sglang`, base commit `0084030179bfba86bfeb6d43f7997d4076329d2c`

## Installed-source baseline

Command:

```bash
/opt/venv/bin/python /tmp/sglang_baseline_30344.py
```

Timing used `time.perf_counter` around the first sampler call and ended after `torch.cuda.synchronize()`. First synchronized GPU execution took `2.9679394010454416` seconds, including first-use compilation. The installed API accepts fixed seed and position values but not caller-supplied uniforms, so fixed uniforms were not used.

The independent CPU oracle sorted positive probabilities, applied top-k and top-p prefix mass, and normalized with `math.fsum`. Boundary cases for zero probabilities, near-unit mass, trailing padding, exact top-p, next-after top-p, and exact top-k all sampled inside the oracle support.

The bounded 512-trial supplementary check exposed the defect:

- Oracle support: tokens `[3, 2, 1]`
- Oracle expected counts: `[0, 113.77777777777777, 170.66666666666666, 227.55555555555554]`
- Installed GPU counts: `[0, 0, 0, 512]`
- Chi-square: `640.0`

The complete raw baseline is retained at `/job/baseline-first.json`. It is an installed-source baseline and is not proof for later checkout changes.

## Root cause and fix

The eager implementation and `torch.compile(..., backend="eager")` matched the independent oracle. Default Inductor compilation returned one rank for every row. Compiling the same Gumbel math with a precomputed hash also matched, isolating the failure to Inductor inlining the custom Triton hash launcher.

The fix adds `@torch.compiler.disable` to `murmur_hash32`. This creates a graph boundary around the unregistered Triton launcher while retaining Inductor compilation for the downstream Gumbel operations.

## Validation

Commands:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/sampling/test_deterministic_sampling_boundaries.py \
  test/registered/sampling/test_deterministic_gumbel_u1.py

PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/sampling/test_sampling_mask.py::TestSamplingMaskCapture
```

Results:

- Focused deterministic tests: `4 passed`, including six boundary subtests.
- Existing Gumbel endpoint regression: passed.
- Adjacent sampling-mask capture tests: `2 passed, 2 skipped` (FlashInfer cases skip on ROCm).
- Post-fix 512-trial distribution: counts `[0, 147, 152, 213]`, expected `[0, 146.28571428571428, 146.28571428571428, 219.42857142857142]`, chi-square `0.4150390625`.

The new boundary test uses binary-exact probability masses for exact cutoff checks. A decimal, non-representable top-p boundary can still differ from an exact-decimal oracle because the production threshold comparison is float32; that behavior is unchanged here.

Only the assigned MI300X was used. No model weights were downloaded, no node-wide state was modified, and no upstream issue, PR, or comment was posted or changed.
