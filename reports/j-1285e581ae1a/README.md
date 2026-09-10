# gfx942 investigation for sgl-project/sglang issue 37936

## Result

This is a bounded, truthful MI300X investigation, not a claim that the full V100/Qwen EAGLE failure is fixed.

- The installed-source gather test is unsupported on ROCm because its JIT device matcher accepts only `cuda`; the raw error is preserved in `baseline-first-installed-jit.txt`.
- A native `torch.gather` control on the assigned MI300X matched its CPU reference exactly in 10 independent-output trials. The median CUDA-event time was 0.22620099782943726 ms.
- Upstream pull request 37988, commit `106abde0b90076404143cc0ceb8db7ebc3c885e4`, moves the single-layer EAGLE draft-extend shared-read event after graph replay. Its focused tests passed from this checkout: 4 passed in 16.07 seconds.
- A two-stream valid boundary-index gather control used indices `0` and `4095`, independent outputs, and exact CPU references. All five bounded trials passed on both streams.
- A one-trial out-of-range control used indices `-1` and `4096`. ROCm aborted the process with `HSA_STATUS_ERROR_EXCEPTION` and `hipErrorLaunchFailure` rather than raising the CUDA-style catchable assertion from the issue. The raw evidence is in `two-stream-oob.txt`.
- A post-abort valid gather passed, and `rocm-smi --showrasinfo all` reported zero correctable and uncorrectable errors for the monitored blocks.

## Environment

- Campaign: `repo-e2e-20260909`
- Job: `j-1285e581ae1a`
- GPU: one AMD Instinct MI300X, `gfx942`, serial `692440003970`, node ID 4.
- Interpreter: `/opt/venv/bin/python`, Python 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- Imported installed source: `/sgl-workspace/sglang/python/sglang`, commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Imported native kernel package: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`, with `common_ops.cpython-310-x86_64-linux-gnu.so`.
- Torch HIP native library: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`.
- Requested image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`. No container image API was available for independent in-container verification.

## Commands

The installed-source baseline and its supported neighboring control are recorded in `baseline-first.json`.

The exact candidate commit was fetched from the PR author's fork and checked out detached:

```bash
git fetch --depth 2 https://github.com/Hao-tian-Zheng/sglang.git 106abde0b90076404143cc0ceb8db7ebc3c885e4
git switch --detach 106abde0b90076404143cc0ceb8db7ebc3c885e4
```

The first candidate test command imported the preinstalled source and failed with a method-signature mismatch. That raw evidence is preserved in `candidate-tests-installed-import.txt`. The checkout-correct command was:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/unit/spec/test_eagle_draft_extend_shared_read_event.py \
  test/registered/unit/spec/test_eagle_draft_cuda_graph_runner.py -s
```

It passed 4 tests in 16.07 seconds. The imported path was verified as `/job/sglang/python/sglang`.

The valid two-stream control ran five separate processes, each bounded by `timeout 30s`:

```bash
timeout 30s env PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  repro-two-stream-gather.py "$trial"
```

The negative boundary control ran once under `timeout 30s`:

```bash
timeout 30s env PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  repro-two-stream-oob.py
```

## Upstream context

- Issue 37936 reports a V100 EAGLE/MTP crash in `vectorized_gather_kernel` while a long decode overlaps a constrained empty-object tool request.
- The only issue comment at investigation time was from Hao-tian-Zheng on 2026-09-04, requesting focused logs and stating that an isolated A100 reproduction was being minimized.
- Open upstream PR 37988, commit `106abde0b90076404143cc0ceb8db7ebc3c885e4`, parent `01e66a62db889b6ed921eaf08360119ea4e2ba75`, changes the event publication order and adds a CPU ordering regression test.
- Upstream PR 38041 later merged a revert of the separate final multi-layer EAGLE shared-read event change from PR 36752. That revert does not show that the single-layer PR 37988 ordering is fixed or invalid, but it prevents generalizing the event-fence approach without focused validation.

## Limitations

- This was one MI300X (`gfx942`), not the reported 4x V100 PCIe topology.
- No model weights were downloaded and no full SGLang server, EAGLE model, grammar request, CUDA graph replay, NCCL collective, or TP4 workload was run.
- The candidate's new test is a CPU ordering regression; passing it does not prove the full runtime race is fixed.
- The two-stream gather controls exercise native gather and stream support, not SGLang's scheduler, speculative batch state, or graph replay.
- ROCm reports out-of-range gather as a fatal `HSA_STATUS_ERROR_EXCEPTION` / `hipErrorLaunchFailure`, so the issue's CUDA assertion text is not expected to be byte-identical on this architecture.
- PR 37988 remains open upstream. This report deliberately does not duplicate its production change.
