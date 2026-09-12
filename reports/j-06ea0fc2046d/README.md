# MiniMax-M3 thinking-control correction

Upstream issue: https://github.com/sgl-project/sglang/issues/32276

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2264

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2153

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2227

## Result

The review's two counterexamples were independently reproduced against exact
candidate commit `a69473368a934197e72bbf7b0794e78316c2e5ee`:

- `thinking={"type":"adaptive"}` became `thinking_mode="enabled"`.
- `reasoning_effort="none"` set only the generic boolean template keys and
  omitted MiniMax-M3's `thinking_mode="disabled"`.

The pinned MiniMax-M3 template at revision
`f0e1c1e04d40177e4673a22097036854f536e9c0` independently confirms that
`adaptive` and `enabled` are distinct: adaptive emits no generation prefix,
while enabled emits `<mm:think>` and instructs the model that it must reason.
The retrieved template had SHA-256
`11421244f67553498e5c8112dae02802025bcc4305ec45ad380af95c96f9fe64`.

The correction preserves the candidate's valid official-field support, passes
`adaptive` through unchanged, and adds `thinking_mode="disabled"` for the
generic `reasoning_effort="none"` path. Explicit template kwargs retain
precedence. Non-`none` reasoning effort deliberately does not synthesize a
MiniMax mode because it cannot choose between adaptive and forced reasoning.

## Evidence

- `raw/regression-before.txt`: 2 failures proving both candidate defects.
- `raw/regression-after.txt`: complete protocol module, 48 passed and 25
  subtests passed.
- `raw/normalization-matrix-after.txt`: corrected mapping and precedence matrix.
- `raw/minimax-template-branches.txt`: relevant branches from the pinned model
  template fetched from Hugging Face.
- `raw/pre-commit.txt`: all applicable hooks passed.
- `raw/gpu-inventory.txt`: assigned single gfx950 device inventory.

## Limitation

MiniMax-M3 weights and the reported TP=4 topology were unavailable. No GPU
inference was run, and the single assigned gfx950 cannot qualify the reported
distributed model execution. This change is supported by deterministic request
normalization and the actual pinned model template, not by a model-semantic
inference claim. There are no native changes, so no native rebuild applies.
