# Reduced DSpark block replay study

## Result

This is a bounded, one-GPU reduced-block study. It is not a GLM checkpoint result, a distributed claim, or a new graph framework.

The reduced pipeline uses existing SGLang primitives for:

- target hidden-state handoff through `TargetHiddenKvInjector`;
- target-hidden projection through a synthetic model adapter;
- commit-side KV projection through `CommitKvProj`;
- draft hidden-state projection through `project_through_lm_head`;
- Markov draft proposal through `run_markov_block`;
- greedy accept through `AcceptGreedy`;
- commit finalization through `FinalizeAcceptLens`.

Each handoff is compared with an independently coded reference. Eager and captured-replay runs use fresh input values and independently cloned outputs. The captured path uses one `torch.cuda.CUDAGraph` per case over preallocated static input tensors; fresh values are copied into those tensors before each replay, and captured outputs are cloned outside the timed region.

## Context

- Upstream context read: `sgl-project/sglang` issue `30734`.
- Coordination tracker read: `amdpilot-org/amdpilotv2` issue `402`.
- Upstream PR `31457` is still open and was not duplicated.
- Mirror `main` already contains later DSpark replay work, including commit `861d40f3e` (`Fix DSpark CUDA graph replay with MegaMoE TP attention (#34919)`).
- Delivery branch base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 304 CUs, 196592 MiB reported by Torch.
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Torch path: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Triton path: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Checkout SGLang path: `/job/sglang/python/sglang/__init__.py`
- Native import observed during startup: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`

## First installed-source GPU baseline

The first GPU execution objective was completed before cloning or editing.

Command:

```text
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/kernels/ops/speculative/test_spec_topk1.py \
  -k test_draft_topk1_postprocess_matches_argmax_and_position_add
```

Installed-source revision: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.

The test passed with four subtests. It compares `draft_topk1_postprocess` against independently constructed argmax indices, probabilities, and incremented positions with `rtol=0, atol=0`. Complete subprocess wall time was `35.757189 s`; pytest reported `32.67 s`.

The first timing wrapper attempt used `/usr/bin/time`, which is absent in the container, and failed before launching the GPU test. The successful baseline uses Python `time.monotonic` around the complete pytest subprocess. The required artifact is `/job/baseline-first.json`; this installed-source result is not evidence for the later mirror checkout.

## Workload cases

There are three cases, below the six-case limit.

| Case | Batch | Draft steps | Full block tokens | Hidden | Vocab | Synthetic weights | Peak allocated |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 1 | 2 | 3 | 1024 | 512 | 3,539,072 B | 243,288,576 B |
| medium | 4 | 4 | 20 | 2048 | 1024 | 13,369,472 B | 341,062,144 B |
| large | 8 | 4 | 40 | 4096 | 2048 | 51,904,640 B | 487,493,120 B |

All cases use one KV head with head dimension 64. Hidden states, logits, K, and V use `bfloat16`. Positions and cache locations use `int64`; prefix and accept lengths use `int32`; token IDs use `int64`.

## Accuracy gates

- Integer outputs: `rtol=0, atol=0`.
- `bfloat16` outputs: `rtol=1e-2, atol=1e-2`.
- Validation uses two fresh seeded input sets per case for both eager and replay.
- Compared handoffs include K/V buffers, draft base logits, sampled tokens, corrected logits, accept length, bonus, trim length, commit length, new sequence length, and commit trim length.
- All eager and replay comparisons passed.

## Complete-block latency

Each mode uses three real warmup forwards, one capture forward for the graph path, and 20 measured real forwards per case. Input generation and static-buffer copies are outside the timed region. Timing uses CUDA events around the complete reduced block.

| Case | Eager median | Replay median | Median speedup |
|---|---:|---:|---:|
| small | 0.541249 ms | 0.083754 ms | 6.462405x |
| medium | 0.651543 ms | 0.129339 ms | 5.037506x |
| large | 0.660685 ms | 0.180216 ms | 3.666073x |

Raw timing arrays and complete summaries are in `reduced-replay-results.json`.

## Unsupported capture boundaries

- No model-specific DSpark constructor is instantiated. `DSparkDraftMixin` is a mixin; concrete model classes require full HF configuration and checkpoint state. A synthetic adapter supplies the target-hidden projection boundary.
- The full `DecodeCudaGraphRunner` is not captured because it requires `ModelRunner`, attention-backend, KV-pool, and distributed runtime state. This study uses one direct `torch.cuda.CUDAGraph` per reduced case.
- The reduced fake KV pool does not expose the MLA `set_swa_key_buffer_radix_fused_norm_rope` path, so `TargetHiddenKvInjector` uses the generic `set_kv_buffer` path.
- No GLM weights are downloaded, and no distributed or model-level throughput claim is made.

## Commands

Run the reduced study:

```text
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  /job/sglang/reports/j-0f672a2e3c66/reduced_replay.py \
  --output /job/sglang/reports/j-0f672a2e3c66/reduced-replay-results.json \
  --measured-runs 20 --warmup-runs 3 --validation-sets 2
```

Run the existing checkout parity control:

```text
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  /job/sglang/test/registered/spec/dspark/test_dspark_kernel_parity.py
```

The existing control passed with one test and 22 subtests in `21.35 s`.

## Limits and caveats

- One assigned MI300X (`gfx942`) GPU.
- Three workload cases, below the six-case limit.
- Synthetic weights are all below 4 GiB; the largest case is about 51.9 MiB.
- Peak allocated memory is below 48 GiB; the largest observed value is about 487.5 MiB.
- Wall limit is 7200 seconds. The first baseline began at `2026-09-10T09:05:08Z`; GPU work and validation completed within 469 seconds.
- The harness rejects more than six cases, more than 100 measured runs, more than 10 warmups, or more than five validation sets.
- No synthetic burn, sleep loop, unbounded loop, full model weights, or node-wide state change is used.

## Development errors observed

The initial harness run hit these concrete boundaries and was corrected:

- `project_through_lm_head` is exported by `sglang.srt.models.dspark`, not `sglang.srt.layers.logits_processor`.
- `AcceptGreedy` requires candidates and target predictions to share the full block width, including the current token.
- Independent reference `sum` promotes int32 to int64; reference accept lengths were normalized to the primitive's int32 output dtype.
- This Torch build exposes `torch.cuda.memory_reserved()`, not `torch.cuda.reserved_memory()`.

All corrected runs passed. These are harness-contract findings, not claims about unrelated production behavior.
