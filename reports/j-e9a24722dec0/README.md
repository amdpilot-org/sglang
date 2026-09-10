# MI300X synthetic Llama memory-scaling report

## Scope

This is the bounded decode-sized reduced-block study from mirror issue 386.  I
read read-only upstream issue 35003 (the 2026 Q3 AMD roadmap; no comments), and
found no prior related mirror issue or PR for this distinct workload.  No
upstream issue, PR, or comment was posted or changed.

## Installed-source baseline

The first GPU execution used the preinstalled interpreter and installed source,
before modifying the persistent checkout:

```bash
cd /sgl-workspace/sglang/test/registered/ops
/opt/venv/bin/python -m unittest -v \
  test_aiter_greedy_sample_amd.TestAiterGreedySample.test_single_request
```

- Installed source: `/sgl-workspace/sglang`, commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`
- Test: `aiter.greedy_sample` versus independent `torch.argmax`
- Case: batch 1, vocabulary 32,000, bfloat16
- Timing: `time.perf_counter` around one subprocess invocation
- First GPU execution elapsed: `11.824009` seconds
- Result: passed (`Ran 1 test ... OK`)

The complete installed-source record is `/job/baseline-first.json`.  This
baseline is evidence for the preinstalled stack only and is not proof for later
checkout changes.

## Environment

- Requested image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID:
  `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, gfx942/CAP `9.4`
- Delivery checkout: `/job/sglang`, base commit
  `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Checkout Python path: `/job/sglang/python/sglang/__init__.py`
- Native kernel path:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Job-private caches: `/tmp/sglang-cache-j-e9a24722dec0`

## Reproduction

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-e9a24722dec0/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-e9a24722dec0/inductor
export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang-cache-j-e9a24722dec0/profile

/opt/venv/bin/python -m sglang.benchmark.synthetic_memory_scaling \
  --output /job/sglang/reports/j-e9a24722dec0/synthetic-memory-scaling.json
```

The benchmark generates a local tiny Llama configuration and uses SGLang's
`DummyModelLoader`; it does not download a tokenizer, checkpoint, or model
weights.

## Workload

- Architecture: 4 layers, hidden size 128, 4 attention heads, 2 KV heads,
  intermediate size 256, vocabulary 512, bfloat16
- Synthetic SGLang weights: `1,444,096` bytes
- Independent Torch control weights: `1,444,096` bytes
- KV workspace: `4,195,328` bytes
- Request workspace: `69,904` bytes
- Cases: batch `{1, 2, 4}` × context `{128, 512}`, eight output tokens each
- Repetitions: one warmup plus three measured repetitions per case
- Real paths: `ModelRunner.extend` for prefill and `ModelRunner.decode` for
  each decode step, with CUDA graphs disabled
- Attention backend: Triton

## Gates and results

The independent Torch control is a separate Llama implementation loaded from
the same synthetic state dict.  Every prefill and decode step checks:

1. Maximum absolute logit difference is within `atol + rtol * max_abs_logit`.
2. If greedy tokens differ, the control's top-two margin is a near tie.
3. KV token availability and request-slot availability match the expected
   allocator state after every forward.

Final raw results are in
`reports/j-e9a24722dec0/synthetic-memory-scaling.json`.

- Workload cases: 6
- Forward/control checks: 144
- Allocator state checks passed: 144
- Token equality: 0 direct matches, 144 near-ties
- Maximum absolute logit difference: `2.7865171432495117e-06`
- Maximum control top-two margin: `0.0`
- Median decode latency: `0.0061541334725916386` seconds
- P90 decode latency: `0.007615393027663231` seconds
- Maximum measured live allocation: `145,966,080` bytes
- Maximum measured peak allocation: `150,725,632` bytes

The activation predictor is an explicit upper bound and is not a leak detector.
Across all checks, predicted activation bytes divided by measured peak-allocation
delta ranged from `0.02158018867924528` to `178.8421052631579`.  The large
overprediction is expected for this tiny fused-path model; the raw JSON records
both predicted and measured values for every step.

## Boundaries and limitations

- This is a reduced synthetic-model study, not a production-model accuracy or
  throughput benchmark.
- Dummy random weights make greedy tokens tie-prone; the independent control
  therefore gates logits and near-tie margin rather than requiring identical
  argmax tie-breaks.
- The benchmark uses one MI300X, one Triton attention backend, and no CUDA
  graph capture.
- No full model weights, tokenizer, or checkpoint were downloaded.
- No node-wide state was modified.
