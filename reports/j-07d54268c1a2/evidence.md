# Evidence

Source paths inspected:

- `python/sglang/kernels/ops/attention/fla/layernorm_gated.py`
- `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`
- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/layers/attention/linear/gdn_backend.py`
- `test/registered/attention/test_qwen35_deterministic.py`
- upstream merged PR `https://github.com/sgl-project/sglang/pull/27869`

Environment:

```text
GPU: AMD Instinct MI355X
ISA target: gfx950
VRAM: 309220868096 bytes
Torch: 2.11.0+rocm7.2
HIP: 7.2.26015
Python: /tmp/amdpilot-repo-j-07d54268c1a2/venv/bin/python
Triton cache: /tmp/amdpilot-repo-j-07d54268c1a2/cache/triton
```

Focused GPU regression result:

```text
..                                                            [100%]
2 passed, 1 warning, 11 subtests passed in 4.53s
```

An additional direct probe measured `ROWS_PER_BLOCK=4` for both 1 and 1025
rows under deterministic mode, bitwise equality (`maxdiff=0.0`) for the shared
row, and agreement with the independent FP32 PyTorch reference.

The historical adaptive policy selects different launch geometries for those
shapes (`ROWS_PER_BLOCK=1` and `4`). That policy-level regression is caught by
the new launch-geometry test even though this AMD compiler/GPU happened to
produce equal values for the sampled row under both geometries.

Unavailable validation:

```text
Qwen/Qwen3.5-27B weights: unavailable on node
Required source-report topology: TP=2
Assigned topology: one MI355X GPU
Full server execution: not run
Native rebuild: not applicable (no native source change)
```
