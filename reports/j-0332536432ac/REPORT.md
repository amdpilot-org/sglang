# MI300X structured speculative verifier and sampler inputs

## Scope

This is a bounded, structured-input robustness follow-up to the prior MI300X speculative verifier backend comparison. It does not repeat that timing study and does not simulate a full model speedup.

The new case set uses six structured input distributions with identical shapes:

- zeros
- tiny finite values
- mixed magnitudes
- cancellation
- skewed probabilities
- state cases with varying accept lengths

Each case uses the same batch size, draft-tree width, and vocabulary width. The test compares complete-block outputs from the installed native verifier and the Triton chain sampler against independent CPU references.

## Environment

- Source commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Image identity: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Native extension: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Triton source: `/job/sglang/python/sglang/kernels/ops/speculative/reject_sampling.py`
- Triton cache: `/tmp/sglang-cache-j-0332536432ac/triton`

## Installed-source baseline

Before cloning or editing, the existing native verifier test was run from the installed source:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/python/sglang/kernels/aot/tests/speculative/test_eagle_utils.py
```

It passed `1` test in `0.42 s`, with `7.251396 s` total first-execution elapsed time. The installed source revision was `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`. This baseline is labeled as installed-source evidence only and is not proof for later checkout changes. Full details are in `/job/baseline-first.json`.

## Method

- Used one synthetic float32 target-probability tensor of shape `[8, 5, 64]` per case.
- Used one synthetic float32 draft-probability tensor of shape `[8, 4, 64]` per case.
- Used a fixed five-node branching draft tree for the native verifier and a fixed five-node chain draft tree for the Triton chain sampler.
- Used fixed uniform samples of `0.5` and final samples of `0.0`.
- Compared complete-block `predicts`, `accept_index`, and `accept_token_num` outputs against independent CPU references with exact equality gates.
- Preserved the documented `float32` input and `int32` output dtypes.
- Used exactly six workload cases and no checkpoint download.

## Results

All six cases passed the independent CPU reference and complete-block exact equality gates for both primitives.

| Case | Native accept counts | Chain accept counts |
|---|---|---|
| zeros | `[0, 0, 0, 0, 0, 0, 0, 0]` | `[0, 0, 0, 0, 0, 0, 0, 0]` |
| tiny | `[3, 3, 3, 3, 3, 3, 3, 3]` | `[4, 4, 4, 4, 4, 4, 4, 4]` |
| mixed magnitudes | `[3, 3, 3, 3, 3, 3, 3, 3]` | `[4, 4, 4, 4, 4, 4, 4, 4]` |
| cancellation | `[3, 3, 3, 3, 3, 3, 3, 3]` | `[4, 4, 4, 4, 4, 4, 4, 4]` |
| skewed | `[3, 3, 3, 3, 3, 3, 3, 3]` | `[4, 4, 4, 4, 4, 4, 4, 4]` |
| state | `[0, 1, 2, 3, 3, 0, 1, 2]` | `[0, 1, 2, 3, 4, 0, 1, 2]` |

The native verifier uses a branching tree with a maximum path length of three accepted drafts. The chain sampler uses a chain tree with a maximum path length of four accepted drafts. The different maximum depths explain the different maximum accept counts.

## Unsupported native stochastic sampling boundary

The installed native stochastic sampling op is unavailable on this stack:

```text
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute
'tree_speculative_sampling_target_only'
```

This boundary is recorded without fabricating an engine result. The structured-input test therefore uses the supported native verifier and the supported Triton chain sampler.

## Reproduction

```bash
PYTHONPATH=/job/sglang/python \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-0332536432ac/triton \
timeout 900s /opt/venv/bin/python -m pytest -q -p no:cacheprovider \
  /job/sglang/test/registered/kernels/ops/speculative/test_structured_speculative_inputs.py
```

Raw complete-block outputs are in `/job/sglang/reports/j-0332536432ac/results.json`.

## Context and limitations

- Read-only upstream context came from `sgl-project/sglang` issue `30344`, its comments, and related changes. No upstream issue, PR, or comment was posted or changed.
- Prior mirror PR `494` covered the native-versus-Triton verifier timing comparison. This report is distinct: it adds structured-input correctness coverage and does not repeat that timing study.
- The test is intentionally small and deterministic. It does not attempt to measure end-to-end model speedup.
