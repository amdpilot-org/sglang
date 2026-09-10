# Open task report: j-9e6ab99e71b6

## Scope

- Campaign: `repo-e2e-20260909`
- Coordination tracker: `amdpilot-org/amdpilotv2` issue `402`
- Upstream context: `sgl-project/sglang` issue `31545`
- Delivery base: `amdpilot-org/sglang` `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Branch: `amdpilot/j-9e6ab99e71b6`

## Environment

- GPU: one assigned AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, serial `692440003982`, GUID `47961`, node `6`
- Image: operator-provided local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`; container hostname was not used as image identity
- Interpreter: `/opt/venv/bin/python` (Python 3.10.12)
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`
- Native modules observed through `libtorch_hip.so`: `/opt/rocm/lib/libamdhip64.so.7`, `/opt/rocm/lib/libroctracer64.so.4`, and `/opt/rocm/lib/libhsa-runtime64.so.1`
- Installed-source context: `/sgl-workspace/sglang/python/sglang/__init__.py`, version `0.5.18.dev20260826+g937af8538b`, commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Delivery checkout: `/job/sglang/python/sglang/__init__.py`
- No full model weights, framework stack replacement, node-wide state change, or synthetic GPU occupancy workload were used

## First GPU baseline

The first execution objective was completed with the installed source before cloning or editing the delivery checkout. It is labeled as an installed-source baseline and is not proof for later checkout changes.

- Command: `/opt/venv/bin/python /job/baseline-first.py`
- Kernel chain: finite `addmm` plus `Gelu`, five steps in each mode
- Reference: eager GPU output compared with graph-replay output after CPU transfer
- Numerical result: eager and graph replay both passed `torch.allclose`; observed maximum absolute difference was `0.0`
- Timing method: CUDA/HIP events around each step, with the end event synchronized before the next step
- First GPU execution elapsed time from process start through warmup synchronization: `4.047891540918499` seconds
- Raw eager trace: `/job/baseline-first-traces/eager.json`, SHA-256 `d122f978ec8ab95060a7c60892a899201fa55e264d2cd49b9cc9158b8f943529`
- Raw graph trace: `/job/baseline-first-traces/graph-replay.json`, SHA-256 `4e442b9ea1136dbd50845c9f6c9f58978f653b1cdda1ac559a787cd16711ab25`
- Baseline record: `/job/baseline-first.json`

The synchronized event sequence establishes actual GPU completion order as step 0 through step 4. In this small, synchronized control, both eager and graph replay recorded ten chain kernels and placed two under each of the five CPU markers. This does not contradict the larger production-graph deferred-attribution report; it only shows that a two-kernel graph is too small to reproduce the partial event loss or deferred burst.

## Upstream candidate

The relevant open upstream candidate is `sgl-project/sglang` PR `35390`, head commit `b4761baafb304d72c2c7c9aa0825e85d05ad36cd`. Its image-runtime fix was not duplicated because it targets ROCm 7.2.4 images and this job must preserve the qualified ROCm 7.2.0 stack.

The exact candidate commit was tested in a detached worktree at `/tmp/sglang-pr35390-b4761ba`:

```bash
timeout 620 /opt/venv/bin/python scripts/ci/amd/check_hip_graph_profiling.py --timeout 300
```

Result on MI300X (`gfx942`), ROCm 7.2.0:

- Eager: `512/512` kernel events
- Graph replay: `448/512` kernel events
- Verdict: `FAIL`; 64 graph-replay dispatch events did not reach the trace
- Elapsed command time: 9 seconds
- Raw log: `/job/pr35390-b4761ba-gfx942.log`, SHA-256 `e40cb83c0b86410ff6a8a82326500181453083c8481c2a627b3d2a7f5d6946f9`

This confirms the candidate's reported partial roctracer loss on this architecture and runtime. It does not test the PR's intended ROCm 7.2.4 image repair, and no image or shared-library state was modified.

## Delivery test

The delivery checkout adds a finite, one-GPU profiling test that preserves raw traces and separates synchronized execution order from marker attribution.

Command:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest \
  /job/sglang/test/registered/profiling/test_hip_graph_replay_attribution.py \
  -q -s --basetemp=/job/delivery-traces-final
```

Result:

- One test passed on the assigned MI300X
- Eager and graph replay both matched the independent eager reference with maximum absolute difference `0.0`
- Each mode recorded five CPU markers, five GPU markers, and ten chain kernel events
- Each step used a CUDA/HIP event pair and synchronized the end event before the next step
- Final raw eager trace: `/job/delivery-traces-final/test_hip_graph_replay_attribut0/eager.json`, SHA-256 `802d38ce6766e46f48c16081ea6572a63897d29c78fb8cd5ba897ea8d20c254f`
- Final raw graph trace: `/job/delivery-traces-final/test_hip_graph_replay_attribut0/graph-replay.json`, SHA-256 `d405341b7eeae7a2e985948cf0329d2944092327a15f4398f025039f47e1d425`
- Final test log: `/job/delivery-test-gfx942-final.log`

## Architecture-specific limitations

- ROCm 7.2.0 can partially drop HIP graph-replay dispatch events. The exact PR 35390 probe recorded 448 of 512 expected graph-replay kernel events on this MI300X.
- A two-kernel synchronized graph is a useful numerical and ordering control, but it is not a valid probe for larger-graph event loss or deferred attribution.
- CPU profiler markers do not prove GPU execution order. Graph-replay kernels may be visible but deferred into a burst after the last CPU marker.
- Use synchronized CUDA/HIP events for actual bounded step timing. Event elapsed time measures the enqueued interval and does not recover per-kernel attribution.
- `--disable-cuda-graph` restores per-step attribution, but eager traces do not preserve graph-mode wall time or overlapping graph streams.
- PyTorch in-trace graph annotations are CUDA-only and are unavailable on ROCm.
- Dispatch-level ROCm profiling is required for graph-node attribution; the current SGLang PyTorch profiler path does not provide it.
- Issue 31545's MI355X production evidence was not reproduced here because this job was assigned one MI300X. No claim is made about MI355X behavior from this run.

## Changes

- Added `test/registered/profiling/test_hip_graph_replay_attribution.py`
- Added the ROCm HIP graph attribution and partial-event-loss limitations to `3rdparty/amd/profiling/PROFILING.md`

No runtime kernel, scheduler, server, or numerical path was changed.
