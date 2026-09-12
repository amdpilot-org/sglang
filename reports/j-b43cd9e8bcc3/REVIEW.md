# Hunyuan MLX correction-generation report

Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/2194 at exact commit `7bdae6c8600db987d8605c7b7ca8f9a798d16355`.

Independent review: https://github.com/amdpilot-org/sglang/pull/2258.

Related upstream implementation inspected: https://github.com/sgl-project/sglang/pull/37721.

## Reproduction before correction

The pinned `mlx-lm==0.31.3` wheel was downloaded without installing it or
changing the prepared environment. Its `mlx_lm/models/hunyuan.py` defines
attention with a fourth positional `kv_states` argument, applies
`query_layernorm`/`key_layernorm` after RoPE, and returns
`(attention_output, kv_states)`. The retained wheel is at
`/tmp/amdpilot-repo-j-b43cd9e8bcc3/downloads/mlx_lm-0.31.3-py3-none-any.whl`
(SHA-256 `758cfddf1180053b7613db76fad3d246a331a2a905808e1164a275621fc983b8`).

The candidate's exact `MLXAttentionWrapper.__call__` was extracted from Git and
executed by `reproduce_decode_contract.py` with that Hunyuan call shape:

```text
$ python reports/j-b43cd9e8bcc3/reproduce_decode_contract.py candidate-attention-wrapper.py
TypeError: __call__() takes from 2 to 4 positional arguments but 5 were given
exit code: 1
```

This is independent confirmation of the review counterexample. The candidate
does fix its narrower startup failure by bypassing Transformers architecture
resolution under MLX, and that change is preserved here.

## Correction

Normal batched decode now delegates attention math to the mlx-lm model through
a batched cache-protocol adapter. This preserves model-specific argument and
return arity, projection sharing, normalization placement, RoPE behavior, and
attention implementation. The existing hand-written implementation remains for
the opt-in fused AOT RoPE + pool-scatter path because that kernel must own RoPE.

The corrected source passes the same retained contract probe:

```text
$ python reports/j-b43cd9e8bcc3/reproduce_decode_contract.py python/sglang/srt/hardware_backend/mlx/kv_cache/attention_wrapper.py
PASS: shared-KV argument reaches delegated decode and preserves arity
exit code: 0
```

## Validation and limitation

The CPU architecture-resolution and boundary suite passed (`5 passed`). Source
compilation and `git diff --check` passed. The MLX attention suite collected 39
tests, including the new Hunyuan-shaped regression, but all were skipped because
this prepared Linux/x86_64 environment has no MLX/Metal runtime. The assigned
gfx950 AMD GPU cannot execute MLX. No model weights were downloaded, and no
`launch_server` or token-equivalence claim is made for
`mlx-community/Hunyuan-7B-Instruct-4bit`.
