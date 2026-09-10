# MI300X speculative verifier backend comparison

## Scope

This is a bounded, report-only comparison of two genuinely supported installed AMD execution backends on one assigned MI300X (`gfx942`):

- Native HIP: `sgl_kernel.verify_tree_greedy`
- Triton: `sglang.kernels.ops.speculative.spec_tree.verify_tree_greedy_kernel_triton`

It uses synthetic float32 target probabilities, locally generated synthetic weights, and branching draft trees. It does not simulate a full model speedup, use a checkpoint download, or relabel a fallback as a requested backend.

## Environment

- Source commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Image identity: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Native extension: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Triton source: `/job/sglang/python/sglang/kernels/ops/speculative/spec_tree.py`
- Triton cache: `/tmp/sglang-cache-j-dfaba28606cb/triton`

## Installed-source baseline

Before cloning or editing, the existing native verifier test was run from the installed source:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/python/sglang/kernels/aot/tests/speculative/test_eagle_utils.py
```

It passed `1` test in `1.84 s`, with `9.427426 s` total first-execution elapsed time. The installed source revision was `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`. This baseline is labeled as installed-source evidence only and is not proof for later checkout changes. Full details are in `/job/baseline-first.json`.

## Method

- Generated one synthetic float32 weight matrix of shape `[256, 4096]` (`4,194,304` bytes), below the `4 GiB` limit.
- Generated per-case float32 features, logits, target probabilities, and integer target predictions from that same weight matrix.
- Constructed valid branching draft trees with batch sizes `1`, `8`, `16`, `32`, `64`, and `128`; branching factors `2` and `3`; depths `2`, `3`, and `4`.
- Used exactly six workload cases, each with fresh output allocation, backend dispatch, kernel execution, and synchronization inside the timed block.
- Used `torch.cuda.Event(enable_timing=True)` with `3` warmup runs and `20` measured runs per backend per case.
- Compared native HIP and Triton outputs against an independent Python CPU tree traversal with exact equality gates for `predicts`, `accept_index`, and `accept_token_num`.
- Recorded complete-block median latency, standard deviation, minimum, maximum, and output differences.
- Did not use graph replay, a full model, a checkpoint, or an unbounded loop.

## Results

All six cases passed the independent CPU reference and native-versus-Triton exact equality gates. Every output difference was `0`.

| Case | Batch | Branching | Depth | Nodes | Native median (ms) | Triton median (ms) | Native / Triton speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `small_tree` | 1 | 2 | 2 | 7 | 0.041034 | 0.077879 | 1.897891x |
| `batch_tree` | 8 | 2 | 3 | 15 | 0.040052 | 0.078521 | 1.960452x |
| `wide_tree` | 16 | 3 | 2 | 13 | 0.039612 | 0.077659 | 1.960504x |
| `deep_tree` | 32 | 2 | 4 | 31 | 0.039491 | 0.078300 | 1.982743x |
| `large_tree` | 64 | 2 | 3 | 15 | 0.039010 | 0.074632 | 1.913150x |
| `max_batch_tree` | 128 | 2 | 3 | 15 | 0.039010 | 0.074111 | 1.899795x |

Peak allocated memory across cases was at most `165,676,032` bytes, below the `48 GiB` live-allocation limit. The full benchmark completed in `14.806216 s`.

## Unsupported native stochastic sampling boundary

The installed native stochastic sampling op is unavailable on this stack:

```text
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute
'tree_speculative_sampling_target_only'
```

This boundary is recorded without fabricating an engine result. The comparison therefore uses the two genuinely supported verifier backends above.

## Reproduction

```bash
PYTHONPATH=/job/sglang/python \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-dfaba28606cb/triton \
timeout 900s /opt/venv/bin/python \
  /job/sglang/reports/j-dfaba28606cb/backend_comparison.py \
  --output /job/sglang/reports/j-dfaba28606cb/results.json \
  --warmup-runs 3 --measured-runs 20
```

Raw measurements are in `/job/sglang/reports/j-dfaba28606cb/results.json`.

## Context and limitations

- Read-only upstream context came from `sgl-project/sglang` issue `30344`, its comments, and related changes. No upstream issue, PR, or comment was posted or changed.
- Prior mirror PR `472` covered Triton chain-speculative verifier eager-versus-graph replay. This report is distinct: it compares native HIP and Triton greedy verifier backends on identical branching-tree data and weights, without graph replay.
- This is a reduced-block study on one MI300X, not a full-model or end-to-end serving benchmark.
- No checkpoint or full model weights were downloaded.
- No environment replacement, GPU burn, infinite repetition, or sleep loop was used.
