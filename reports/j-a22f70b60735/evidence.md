# EP scatter investigation evidence

The base implementation at `358c163250ad3b1f62939b01ce1314a0a31a0365`
contained this cross-lane communication through global memory:

```python
tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)
cur_expert_start = tl.load(expert_start_loc + cur_expert)
```

On the assigned AMD Instinct MI350X (`gfx950:sramecc+:xnack-`), Triton 3.7.0
preserved the dependency in generated ISA without inserting a barrier:

```text
; ep_moe_kernels.py:1036:5
buffer_store_dword v0, v4, s[16:19], 0 offen
; ep_moe_kernels.py:1039:35
v_mov_b32_e32 v4, 0
global_load_dword v0, v4, s[0:1]
```

The baseline numerical stress used poisoned prefix storage and an oversized,
guarded output allocation. All 200 launches happened to produce correct results
on gfx950, so this investigation does not claim reproduction of the reporter's
intermittent H20 illegal access. The compiler output nevertheless demonstrates
the reported unsynchronized store/reload pattern in the actual checked-out
kernel.

The fix computes `cur_expert_start` by reducing `tokens_per_expert`, which is
already resident in registers. Post-fix TTIR has no
`tt.load %cur_expert_start`, and 200 guarded launches again matched an
independent PyTorch reference. The committed tests include the invariant as a
failing-before regression and cover empty experts, padded tails, block sizes
32/64/128, and a non-power-of-two expert count.

Complete raw logs and compiler outputs are retained outside the worktree under
`/tmp/amdpilot-repo-j-a22f70b60735/evidence/`:

- `baseline_probe.txt` and `baseline_kernel.{ttir,ttgir,llir,amdgcn}`
- `pytest_ep_scatter_before.txt` and `pytest_ep_scatter_after.txt`
- `postfix_probe.txt` and `postfix_kernel.{ttir,ttgir,llir,amdgcn}`
- `pre_commit.txt`

The exact NVIDIA H20/PTX behavior and the model-level, multi-node DeepEP timeout
remain unverified because this job provided one gfx950 GPU and no applicable
model weights or distributed DeepEP setup.
