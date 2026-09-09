# GLM-5.3 transformers-layout weight mapping investigation

## Result

This is a report-only investigation. It does not apply or duplicate the upstream
mapping change.

- Mirror `main` at `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4` loaded **0 of 13**
  intended transformers-layout tensors into their targets.
- Upstream candidate commit `d997f8532039306c9be91c7ebdcd02590e0c52ad`
  (sglang pull 38616) loaded **13 of 13** intended tensors with exact equality
  (`rtol=0`, `atol=0`, `max_abs_diff=0`).
- The candidate's focused test file passed: **7 passed**.
- Both revisions silently ignored an unexpected key and a missing key. Neither
  emitted a warning or error, and the missing target stayed unchanged. The
  candidate therefore validates the key/tensor mapping but does not resolve
  unexpected/missing-weight visibility.

The investigated behavior matches sglang issue 38618 and amdpilot-org/sglang
issue 73. The candidate is sglang pull 38616, whose parent commit is
`e54ff1efb90410afffb784191768fe55ce727a37`.

## Environment

- Image (operator-provided): `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
  with local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
  The image ID could not be independently queried from inside the container.
- GPU: one AMD Instinct MI300X, `gfx942`, UUID
  `GPU-f70f4ffec9a1c0e6`, 206,141,652,992 bytes VRAM.
- ROCm driver: `6.19.14.31400000`.
- Python: `/opt/venv/bin/python` (resolves to `/usr/bin/python3.10`), Python
  3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, imported from
  `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Torch HIP: `7.2.26015-fc0010cf6a`.
- Native Aiter module:
  `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.
- Current source: `/job/sglang/python/sglang/srt/models/glm5_next.py`.
- Candidate source:
  `/job/sglang-candidate/python/sglang/srt/models/glm5_next.py`.
- OS: Ubuntu 22.04.5 LTS.

No GLM checkpoint or model weights were downloaded. No compilation was required.
The only network operations were bounded retries for the mirror clone and the
upstream candidate fetch.

## Synthetic fixture

`validate_glm53_mapping.py` constructs a one-layer, language-only GLM-5.3-style
model with:

- two routed experts,
- mHC enabled,
- one KDA layer,
- a fused KDA projection,
- a packed KDA conv1d,
- no vision modules.

The synthetic transformers-layout fixture contains 13 tensors:

| Transformers key | SGLang target |
| --- | --- |
| `model.language_model.layers.0.attn_hc.fn` | `model.layers.0.hc_attn_fn` |
| `model.language_model.layers.0.attn_hc.base` | `model.layers.0.hc_attn_base` |
| `model.language_model.layers.0.attn_hc.scale` | `model.layers.0.hc_attn_scale` |
| `model.language_model.layers.0.ffn_hc.fn` | `model.layers.0.hc_ffn_fn` |
| `model.language_model.layers.0.ffn_hc.base` | `model.layers.0.hc_ffn_base` |
| `model.language_model.layers.0.ffn_hc.scale` | `model.layers.0.hc_ffn_scale` |
| `...self_attn.forget_gate.f_a_proj.weight` | fused KDA rows `[26:30]` |
| `...self_attn.forget_gate.f_b_proj.weight` | fused KDA batch row `[0]` |
| `...self_attn.forget_gate.dt_bias` | `self_attn.dt_bias` |
| `...self_attn.forget_gate.A_log` | `self_attn.A_log` |
| `...self_attn.conv1d.weight` | `self_attn.qkv_conv1d.weight` |
| `...mlp.experts.gate_up_proj` | `mlp.experts.w13_weight` |
| `...mlp.experts.down_proj` | `mlp.experts.w2_weight` |

The reference model receives direct `copy_` assignments to these targets. The
loaded model is then compared tensor-by-tensor with `torch.allclose(..., rtol=0,
atol=0)`. The fused KDA `f_a_proj` offset is 26, derived from the module's
output partition sizes `[8, 8, 8, 2, 4, 4]`.

## Observations

### Current mirror main

- `all_intended_tensors_match`: `false`.
- Intended tensors matched: `0 / 13`.
- Every affected target retained its initialized value.
- Unexpected key `model.language_model.layers.0.unexpected.weight` changed no
  parameter and produced no warning or error.
- Omitting `model.language_model.layers.0.ffn_hc.base` left
  `model.layers.0.hc_ffn_base` unchanged and produced no warning or error.

Raw result: `raw/glm53_current.json`.

### Upstream candidate

- `all_intended_tensors_match`: `true`.
- Intended tensors matched: `13 / 13`.
- Every per-tensor `max_abs_diff` was `0`.
- The fused KDA loader received `f_a_proj` as shard ID 4, shape `[4, 8]`,
  computed offset 26, and copied it into output dimension 0.
- Unexpected and missing-key behavior was unchanged from current `main`: both
  were silently ignored.

Raw result: `raw/glm53_candidate.json`.

### Candidate unit tests

Command:

```bash
cd /job/sglang-candidate
PYTHONPATH=/job/sglang-candidate/python \
  /opt/venv/bin/python -m pytest \
  test/registered/unit/models/test_glm5_next_hf_layout.py -q
```

Result: `7 passed, 4 warnings in 14.59s`.

## Reproduction

Run from the mirror root:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-d0fd75a5c6a5/validate_glm53_mapping.py \
  --source-root /job/sglang/python \
  --output /tmp/glm53_current.json
```

Run against the preserved candidate worktree:

```bash
PYTHONPATH=/job/sglang-candidate/python /opt/venv/bin/python \
  /job/sglang/reports/j-d0fd75a5c6a5/validate_glm53_mapping.py \
  --source-root /job/sglang-candidate/python \
  --output /tmp/glm53_candidate.json
```

The harness initializes a private one-rank Gloo process group, constructs the
tiny model on CUDA device 0, and synchronizes after loading.

## Limitations

- This validates the actual `Glm5NextForConditionalGeneration.load_weights`
  mapping on one gfx942 GPU, not full-server generation quality.
- No full GLM checkpoint was downloaded or loaded.
- The candidate was not merged or copied into this mirror PR because it is
  already an open upstream fix and validated as such.
- Unexpected/missing-weight visibility remains unresolved in both revisions.
- `ruff` was unavailable in the environment; the harness was syntax-checked and
  executed successfully.
