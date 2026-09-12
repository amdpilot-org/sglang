# Independent review of PR 1387

Candidate: https://github.com/amdpilot-org/sglang/pull/1387 at `dc02b87987b39b87b6e3797d3637a47c551f6b3e`

Upstream issue: https://github.com/sgl-project/sglang/issues/35702

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1424

Recommendation: **accept**. The candidate fully resolves the original key-compatibility defect at the source level.

## Evidence

On the required base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the independent test wrote a real safetensors shard under the post-forward key `model.layers.0.self_attn.attn_mha.kv_b_proj.weight` and invoked the actual `ShardedStateLoader.load_model` loop with a fresh canonical-only model. It reproduced:

```text
alias_to_fresh: KeyError 'model.layers.0.self_attn.attn_mha.kv_b_proj.weight'
```

At the exact candidate commit, the same test passed and exact tensor values were present under the canonical parameter:

```text
alias_to_fresh: PASS exact values copied
direct_alias_present: PASS
unrelated_alias: KeyError 'model.layers.0.self_attn.attn_mha.q_proj.weight'
oversized_tensor: RuntimeError The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0
duplicate_alias_and_canonical: KeyError 'model.layers.0.self_attn.kv_b_proj.weight'
```

The candidate's own focused regression passed (`4 passed`). The broader model-loader unit directory produced `145 passed, 5 skipped, 6 failed`; all failures were in ModelOpt tests that reject or lack support on this ROCm environment and are unrelated to the changed alias resolution.

The implementation is appropriately narrow: it first honors an exact checkpoint key, then remaps only `.self_attn.attn_mha.kv_b_proj.` to `.self_attn.kv_b_proj.` when that canonical key actually exists. It does not conceal unrelated or missing keys. This directly repairs the mismatch caused when a forward registers the same `kv_b_proj` module under `attn_mha` and lexicographic storage deduplication retains the alias.

## Environment and limitations

The prepared interpreter imported SGLang from `/job/repo/python/sglang` and the loader from `/job/repo/python/sglang/srt/model_loader/loader.py`. It used PyTorch `2.11.0+rocm7.2`, HIP `7.2.26015`, and detected one AMD Instinct MI350X (gfx950, capability 9.5).

DeepSeek-V2-Lite weights were unavailable, so the complete Engine save-after-forward/load/generate sequence was not run. The original NVIDIA GH200/CUDA 13 environment was also unavailable. Those facts limit full-model and cross-architecture validation, but not the independently reproduced and corrected loader key contract. The tiny Llama fixture was not used because it cannot exercise DeepSeek MLA aliasing. No native files changed, so no native rebuild was applicable.

Raw command output was preserved outside the revision-switched checkout in `/job/review-evidence/j-c1ba3b35b695/` during review.
