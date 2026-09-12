# Independent review of PR 2313

Candidate: https://github.com/amdpilot-org/sglang/pull/2313 at exact commit
`246dbf49a0281533134005bf8ee2f8358e965ba6`.

Upstream issue: https://github.com/sgl-project/sglang/issues/32521

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2317

Parent candidate: https://github.com/amdpilot-org/sglang/pull/2194 at exact
commit `7bdae6c8600db987d8605c7b7ca8f9a798d16355`.

Parent independent review: https://github.com/amdpilot-org/sglang/pull/2258

## Recommendation

`unverified`

The source-level changes address both known failure mechanisms, and I found no
new source-level counterexample. However, the prepared host cannot execute MLX,
so this review cannot establish that the original Hunyuan model can actually be
served or that batched decode is token-equivalent to `mlx_lm`. This is stronger
than test-only hardening: the implementation changes the failing production
paths. It is not a fully verified original-issue fix in this environment.

## Independent reproduction

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the
candidate's architecture regression produced `1 failed, 2 passed`. The MLX
case called `get_model_architecture` and raised the simulated Hunyuan resolution
error. This reproduces the original scheduler-initialization failure mechanism.

The exact parent candidate `7bdae6c8` was independently extracted and exercised
with Hunyuan's fourth positional shared-KV argument. It raised:

```text
TypeError: __call__() takes from 2 to 4 positional arguments but 5 were given
```

The Hunyuan call contract was checked against the uninstalled pinned
`mlx-lm==0.31.3` wheel. Its attention accepts `kv_states`, applies RoPE before
`query_layernorm`/`key_layernorm`, updates the supplied cache, and returns
`(output, kv_states)`.

## Candidate validation

At exact candidate commit `246dbf49`:

- The model-loader suite passed: `5 passed`. It verifies that MLX bypasses
  architecture resolution and that non-MLX and cached-resolution boundaries are
  preserved.
- The retained shared-KV routing probe passed and showed the extra positional
  value reaches delegated decode.
- The MLX attention suite collected 39 tests, including the Hunyuan-shaped
  regression, but every test skipped because `mlx` is unavailable.
- Python compilation and `git diff --check` passed.
- No native source changed; no native rebuild was applicable.

Inspection of `BatchedPoolCache` against `mlx-lm==0.31.3` found the expected
cache protocol (`offset`, `update_and_fetch`) and the delegated call preserves
Hunyuan's model-owned projections, shared KV, RoPE/norm ordering, attention, and
tuple return. The candidate also preserves the hand-written path only when the
fused AOT RoPE/pool-scatter kernel must own that operation.

## Environment and remaining limitation

The host is Linux x86_64 with ROCm 7.2 and one visible AMD Instinct MI350X
(`gfx950`). `mlx` is not installed and Metal/Apple Silicon is unavailable. The
AMD GPU cannot execute MLX, so GPU execution was not relevant or performed.
Hunyuan weights were not downloaded. Consequently there is no independently
observed `launch_server`, real batched decode, or token-equivalence result for
`mlx-community/Hunyuan-7B-Instruct-4bit`. The deterministic tiny Llama fixture
would only validate transport/engine execution and cannot qualify Hunyuan's
architecture semantics, so it was not substituted for the required model.

Raw logs and extracted sources were retained outside the checkout under
`/tmp/amdpilot-repo-j-9a4f9747ac86/evidence/` while revisions were switched.
