# Independent review of MiniMax-M3 thinking control

Upstream issue: https://github.com/sgl-project/sglang/issues/32276

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2353

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2314

Candidate commit: `00d239f664ecf11c7709af1ab9218bbb5468f8c7`

## Verdict

Recommendation: **accept**. The candidate fully resolves the original issue at
the deterministic serving request-normalization boundary.

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`,
`thinking={"type":"disabled"}` and `thinking={"type":"adaptive"}` were silently
discarded, while `reasoning_effort="none"` produced only the generic boolean
keys `thinking=false` and `enable_thinking=false`. The known-working explicit
`chat_template_kwargs.thinking_mode="disabled"` passed through unchanged.

At the exact candidate commit, the mappings are:

- official disabled -> `thinking_mode="disabled"`;
- official adaptive -> `thinking_mode="adaptive"`;
- top-level or nested `reasoning_effort="none"` -> generic false keys plus
  `thinking_mode="disabled"`;
- an explicit `chat_template_kwargs.thinking_mode` retains precedence;
- non-disabled effort does not invent a MiniMax mode.

The candidate's full protocol regression passed (48 tests and 25 subtests).
Independent adversarial cases covered nested effort, explicit-template
precedence, invalid official field values, generic high effort, and simultaneous
control fields. Simultaneous contradictory controls can leave generic boolean
keys inconsistent with `thinking_mode`; MiniMax-M3 consumes `thinking_mode`, so
this does not counterexample either original single-control request, but clients
should avoid contradictory controls.

The pinned MiniMax-M3 template at revision
`f0e1c1e04d40177e4673a22097036854f536e9c0` independently confirms that
`disabled`, `adaptive`, and `enabled` are distinct modes and that the generation
prefix branches directly on `thinking_mode`. Its downloaded SHA-256 was
`11421244f67553498e5c8112dae02802025bcc4305ec45ad380af95c96f9fe64`.

## Environment and limitations

Imports resolved to the candidate checkout under `/job/repo/python`, not an
installed wheel. The change is Python-only, so no native rebuild was applicable.
The assigned device was one AMD Instinct MI355X (`gfx950`). MiniMax-M3 weights
and the reported TP=4 topology were unavailable, so no full model-semantic or
distributed serving reproduction was attempted. GPU execution was not needed
for the deterministic request and template contract verified here.
