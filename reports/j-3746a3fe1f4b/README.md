# gfx942 flow-Euler step fidelity

## Scope

This report covers the local numerical fidelity of
`FlowMatchEulerDiscreteScheduler.step` on one AMD Instinct MI300X (gfx942). It
does not claim full-generation quality, normalization PTX correctness, or H3
platform admission.

The checked-in GPU test covers:

- deterministic and stochastic flow updates;
- first and last timestep endpoints from `[0.8, 0.4, 0.0]`;
- float32/bfloat16/float16 sample and model-output dtype combinations;
- an independent CPU float64 reference formula;
- caller input preservation with `torch.equal`.

## Commands

Run the focused test:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_flow_match_euler_step_gpu.py
```

Regenerate the raw evidence:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-3746a3fe1f4b/run_step_fidelity.py \
  reports/j-3746a3fe1f4b/results.json
```

## Results

The installed-source baseline is recorded outside the repository at
`/job/baseline-first.json`. It used the preinstalled scheduler at
`/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_flow_match_euler_discrete.py`
and is not evidence for later checkout changes.

The mirror-checkout evidence is in `results.json`. All 24 cases passed the
unchanged tolerance gate of four output-dtype epsilons and preserved both caller
inputs. The largest observed difference from the float64 reference was
`0.015625` for a bfloat16 output, which is within the gate.

Timing used CUDA events after three warmups, with ten measured calls per case.
Median step times ranged from `0.026982` to `0.040614` ms for deterministic
updates and from `0.067396` to `0.086560` ms for stochastic updates. The
evidence also includes one CUDA profiler pass for each mode and records the
native Torch kernel names and self-CUDA times.

## Upstream context

Read-only context was sgl-project/sglang issue 23494, the 2026 Q2 AMD roadmap.
Its diffusion-related item is kernel fusion; its comments discuss unrelated
KV-cache topics. Nearby open PRs 37940 and 33552 were inspected: they add
trajectory hooks and fix a different MiniMax H3 sigma schedule, respectively.
Neither changes this local Euler update, so this work does not duplicate them.
