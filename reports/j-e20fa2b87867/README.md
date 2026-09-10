# MI300X shared-prefix and short-tail LSE investigation

## Result

No kernel or backend change is included. The current checkout's supported
ROCm merge path preserves shared-prefix plus empty/short-tail degeneracy and
LSE semantics for the bounded real-GPU cases tested here. Because no mismatch was
demonstrated, this report-only change does not modify the operation contract.

This is a separate follow-up to amdpilot-org/sglang issue 228. The read-only
upstream context is sgl-project/sglang issue 1715 and the closed candidate
sgl-project/sglang pull request 5206. The candidate's final listed commit is
`1659640209beb809a8b30f1e8616df8fb15d09f9`; it was reviewed but not applied.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local
  image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- GPU: one AMD Instinct MI300X, compute capability `9.4` (`gfx942`), 206 GB HBM.
- Interpreter: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; ROCm `7.2.26015-fc0010cf6a`.
- Torch path: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Installed SGLang source: `/sgl-workspace/sglang`, imported from
  `/sgl-workspace/sglang/python/sglang`.
- Installed `sgl_kernel` path: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`.
- Mirror checkout: `/job/sglang`, base commit
  `0084030179bfba86bfeb6d43f7997d4076329d2c`.

The container does not expose its image ID through a runtime metadata file. The
qualified image and local image ID above are therefore recorded from the task
assignment rather than read from inside the container.

## Installed-source baseline

`baseline-first.json` records the first GPU execution. It is an
installed-source baseline only and is not proof for later checkout changes.

The installed FA3 wrapper import failed with:

```text
ImportError: Can not import FA3 in sgl_kernel. Please check your installation.
```

The supported neighboring control was
`torch.nn.functional.scaled_dot_product_attention` on BF16 tensors of shape
`[2, 2, 8, 128]` for queries and `[2, 2, 16, 128]` for keys and values. It was
compared with an independent float32 einsum softmax reference:

- Maximum output absolute difference: `0.0013458728790283203`.
- Mean output absolute difference: `0.0003359642287250608`.
- All independent reference LSE values were finite.
- Timing used CUDA events with three warmups and the mean of 20 calls:
  `0.022523950040340423` milliseconds per call.
- First GPU execution occurred `0.7408393509685993` seconds after process start.

## Current-checkout property control

`cascade-property-results.json` records the real-GPU cases run against the
mirror checkout. The tested operation was
`sglang.kernels.ops.attention.merge_state.merge_state_triton`. On this ROCm
system, `sglang.srt.layers.attention.merge_state._is_cuda` is false, so the
in-tree Triton merge is the meaningful supported control. The installed native
`sgl_kernel.merge_state_v2` op was unavailable:

```text
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'merge_state_v2'
```

Inputs used a finite deterministic adversarial sign/magnitude matrix with no
NaN or Inf values. Each case used 8 query tokens, 4 query heads, head dimension
128, and a 64-key shared prefix. Tail lengths were 0, 1, 3, and 7 keys. The
independent reference computed float32 einsum softmax and logsumexp over the
concatenated prefix and tail keys.

The unchanged numerical gates were:

- Float32 output maximum absolute difference: at most `1e-6`.
- Float32 LSE maximum absolute difference: at most `1e-5`.
- BF16 output maximum absolute difference: at most `0.02`.
- BF16 LSE maximum absolute difference: at most `0.01`.

All cases passed. Maximum observed differences were:

| dtype | Tail keys | Output max abs diff | LSE max abs diff |
|---|---:|---:|---:|
| float32 | 0 | `0.0` | `0.0` |
| float32 | 1 | `1.862645149230957e-08` | `4.76837158203125e-07` |
| float32 | 3 | `2.2351741790771484e-08` | `4.76837158203125e-07` |
| float32 | 7 | `2.2351741790771484e-08` | `4.76837158203125e-07` |
| bfloat16 | 0 | `0.00012004747986793518` | `0.0` |
| bfloat16 | 1 | `0.00020081177353858948` | `4.76837158203125e-07` |
| bfloat16 | 3 | `0.00023667141795158386` | `4.76837158203125e-07` |
| bfloat16 | 7 | `0.00020556896924972534` | `4.76837158203125e-07` |

All merged outputs and LSE values were finite. Timing used CUDA events with
three warmups and the mean of 20 calls; per-case means are in
`cascade-property-results.json`.

## Unsupported candidate boundary

Current main does not contain the cascade decode path from upstream pull request
5206. That closed candidate imports `MultiLevelCascadeAttentionWrapper` from
`flashinfer.cascade` and adds a decode path gated by common-prefix length and
batch statistics.

The qualified environment has no importable `flashinfer` distribution, and the
installed `sgl_kernel` FA3 wrapper is unavailable. Testing the stale candidate
would require adding or replacing the attention framework stack, which is outside
this task's operation contract. The candidate was therefore not applied or
claimed as a working fix. No upstream issue, pull request, or comment was posted
or changed.

## Reproduction

From the repository root:

```bash
/opt/venv/bin/python reports/j-e20fa2b87867/reproduce.py
```

The script writes `/job/cascade-property-results.json`. It uses only the
assigned `cuda:0` device, a bounded 64-key prefix, tail lengths 0/1/3/7, three
warmups, and 20 timed calls per case. It does not download model weights or run
unbounded GPU work.

The installed-source baseline command was:

```bash
/opt/venv/bin/python /job/baseline_first.py
```

## Raw evidence

- `baseline-first.json`: installed-source baseline and first-GPU timing.
- `cascade-property-results.json`: current-checkout raw numerical and timing results.
- `reproduce.py`: bounded reproduction script used for the current-checkout cases.
