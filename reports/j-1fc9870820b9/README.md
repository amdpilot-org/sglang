# SGLang ROCm environment qualification: j-1fc9870820b9

Outcome: **qualification_passed**

This qualifies execution and HTTP protocol behavior of the prepared SGLang
environment on its assigned ROCm GPU. The model is intentionally random and
synthetic, so this is not a semantic-quality result and does not claim that an
upstream model-quality issue is solved.

Platform context: https://github.com/amdpilot-org/amdpilotv2/pull/435

## Environment and model

- Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- GPU: AMD Instinct MI355X, `gfx950`, 294896 MiB reported by Torch
- Torch: `2.11.0+rocm7.2`; HIP: `7.2.26015`
- SGLang: `0.5.19.dev20260908+g554f817948`
- Attention backend: SGLang Triton (ROCm supported)
- Model: locally generated two-layer random Llama, FP16, hidden size 64,
  four attention heads, two KV heads, 128-token vocabulary and 256 positions
- Seed: `20260912`
- Weight SHA256: `6632fab7c351a0bd85518d80a89ec673f68139587f5ad3fa8d5ab38fad87e75f`

The full model configuration, exact tokenizer vocabulary, dependency versions,
seed and weight digest are in `evidence/fixture/fixture-manifest.json`. No model
was downloaded. `create_tiny_llama.py` recreates it in the job-private cache.

## Results

Two real source-checkout servers were launched on unused loopback ports with a
128-token context, 256-token pool, at most four running requests and 1% static
memory fraction. Both became ready and passed:

| Probe | Eager run | Graph run | Output evidence |
|---|---:|---:|---|
| ordinary `/generate` | HTTP 200 | HTTP 200 | 8 completion tokens |
| OpenAI `/v1/completions` | HTTP 200 | HTTP 200 | 8 completion tokens |
| two-prompt `/generate` batch | HTTP 200 | HTTP 200 | 6 + 6 completion tokens |
| streaming OpenAI completion | HTTP 200 | HTTP 200 | 8 SSE data events |

The eager server became ready in 32.062 seconds. Its log records real Triton
compilation and GPU prefill/decode work. The graph server became ready in 25.036
seconds and captured full decode CUDA graphs for batch sizes 1 and 2 in 2.51
seconds before executing the same probes. The generated text has no semantic
meaning because the weights are random.

No dependency, Torch, ROCm, host, or system-service changes were made. The
server warned that host NUMA balancing is enabled; it did not prevent startup
or execution and was not changed.

## Cleanup and limitation

Both owned servers were stopped and GPU memory returned to the pre-test idle
level (308121600 bytes reported used, 0% utilization). During the first run,
the initial launcher stopped the process group but two already-dead worker
grandchildren became zombies under container PID 1, which did not reap them.
They hold no GPU resources and cannot be reaped by an unrelated process. The
checked-in runner now installs itself as a Linux child subreaper; the graph run
then reaped both descendants and confirmed that its process group was gone.
This container/PID-1 cleanup behavior is the only remaining platform limitation.

## Reproduce

From the prepared checkout, using the required interpreter:

```bash
PY=/tmp/amdpilot-repo-j-1fc9870820b9/venv/bin/python
$PY reports/j-1fc9870820b9/create_tiny_llama.py
$PY reports/j-1fc9870820b9/run_server_probe.py \
  --fixture /job/cache/j-1fc9870820b9/tiny-random-llama \
  --output /job/qualification-j-1fc9870820b9/eager
$PY reports/j-1fc9870820b9/run_server_probe.py \
  --fixture /job/cache/j-1fc9870820b9/tiny-random-llama \
  --output /job/qualification-j-1fc9870820b9/graph --graph
```

`evidence/eager` and `evidence/graph` contain the complete raw server logs,
launch/exit metadata, requests, raw responses, parsed response schemas, token
counts and streaming events from the qualification runs.
