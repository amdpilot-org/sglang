# Qwen3.8 Flash Next PLE weight-scale investigation

## Result

The reported assertion is reproducible with an 884-byte synthetic checkpoint on one AMD Instinct MI300X. The failure is not caused by a scale that should be intentionally skipped: the checkpoint's non-unit PLE `weight_scale` is semantically required, but its checkpoint layer name does not match the model's registered buffer name.

For a model configured with `ple_layer_ids=[1]`, SGLang registers the PLE buffer as:

```text
model.layers.0.ple.ple_embedding.ngram_embedding.weight_scale
```

The checkpoint supplies:

```text
model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale
```

The dedicated PLE buffer loader therefore finds no exact buffer. The name later reaches the generic unknown-`_scale` fallback, which correctly rejects a non-unit value:

```text
Expected 1.0, got 0.00019931793212890625 in skipped model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale
```

No assertion or numerical gate was weakened. No production mapping change is included in this report PR.

## Scale contract

| Scale | Value | Consumed | Intentionally skipped | Result |
|---|---:|---:|---:|---|
| Checkpoint `weight_scale` | `0.00019931793212890625` | No | No | Falls through to the generic non-unit assertion. |
| Model-registered `weight_scale` | `0.00019931793212890625` | Yes | No | Dedicated PLE buffer loader copies the exact BF16 value. |
| Unrelated unit `_scale` | `1.0` | No | Yes | Generic fallback permits this intentional skip. |

The mapped control loads the same directly referenced FP8 shard and BF16 scale through the real `Qwen4ExpForConditionalGeneration.load_weights` path. It then runs the actual `VocabParallelEmbedding` lookup on GPU and multiplies by the loaded `weight_scale`. The BF16 result matches the direct `FP8 -> BF16`, then `* scale` reference exactly. The maximum float32 residual after BF16 rounding is `1.7881393432617188e-07`.

## Tested candidates

| Candidate | Commit | Result |
|---|---|---|
| `amdpilot-org/sglang` `main` base | `0084030179bfba86bfeb6d43f7997d4076329d2c` | Reproduced; mapped control passed. |
| Upstream PR 36601 head | `3003ddf1574ef5004e21a10e36aaabc364766921` | Reproduced; mapped control passed. |
| Closest source branch, issue-time commit | `24fd11b7fe09993ed35da3cc1d4b33369ba2c8e8` | Reproduced; mapped control passed. |
| Closest source branch head | `599d740312f97cfdba91a818dfec43037ab42fe8` | Reproduced; mapped control passed. |

Upstream issue 36616 remains open. Its one comment only says that it would be checked; no fix or related change is linked. Upstream PR 36919 explicitly says its image reproduces this assertion and points to PR 36601 for the engine implementation. PR 36601 was tested directly and still reproduces the failure.

The issue does not name its source branch, and upstream has no branch named `qwen-next`. The closest available source candidate is `sglang-miles-qwen38next`; its issue-time and head commits were tested. The harness accepts `--require-commit` and fails clearly if the requested source root or commit is unavailable or mismatched.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942` / capability `9.4`, 304 compute units
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Triton: `3.7.0+amd.rocm7.2.0.git89002410`
- Delivery source: `/job/sglang`
- Preinstalled source used only for environment context: `/sgl-workspace/sglang`
- Torch path: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch HIP library: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- Triton path: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Triton library: `/opt/venv/lib/python3.10/site-packages/triton/_C/libtriton.so`

The image ID is the operator-provided local image identity. No Docker, Podman, containerd, CRI, or Skopeo runtime/socket was available inside the job for an independent image inspection.

## Reproduction

The harness generates the tiny fixture in a job-private cache; it does not download Qwen weights. It contains one FP8 PLE shard, the non-unit BF16 scale from the issue, and one unrelated unit scale.

The generated fixture is 884 bytes with SHA-256 `fdffc09a0e4005181e4fe3be1057962272afbfe09497968c674af8f3d12c04de`.

```bash
CACHE=/tmp/sglang-cache-j-ce23dab7ebc1
mkdir -p "$CACHE"

PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-ce23dab7ebc1/reproduce_ple_weight_scale.py \
  --fixture "$CACHE/qwen4-ple-tiny.safetensors" \
  --output "$CACHE/amdpilot-main-result.json" \
  --source-root /job/sglang \
  --source-label amdpilot-main \
  --require-commit "$(git -C /job/sglang rev-parse HEAD)"
```

Candidate worktrees were created outside the delivery worktree and tested with the same command shape:

```bash
git -C /job/sglang worktree add --detach \
  "$CACHE/sglang-pr-36601" 3003ddf1574ef5004e21a10e36aaabc364766921

PYTHONPATH="$CACHE/sglang-pr-36601/python" /opt/venv/bin/python \
  /job/sglang/reports/j-ce23dab7ebc1/reproduce_ple_weight_scale.py \
  --fixture "$CACHE/qwen4-ple-tiny.safetensors" \
  --output "$CACHE/pr-36601-result.json" \
  --source-root "$CACHE/sglang-pr-36601" \
  --source-label pr-36601 \
  --require-commit 3003ddf1574ef5004e21a10e36aaabc364766921
```

The command exits successfully only when all three checks pass:

1. The unchanged checkpoint-named, non-unit scale reaches the exact assertion.
2. The mapped model-named scale and FP8 shard load through the actual loader.
3. The GPU embedding/dequantization control matches the directly loaded reference.

Raw JSON outputs are retained in `reports/j-ce23dab7ebc1/results/`.

## Limits and unfinished work

- The exact issue image source branch is not identified by the issue and could not be proven; only the closest available source candidate was tested.
- This PR reports and proves the contract but does not change the production name mapping.
- No full Qwen weights were downloaded, and no upstream issue, PR, or comment was posted or modified.
