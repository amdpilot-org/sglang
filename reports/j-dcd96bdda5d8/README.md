# Independent review of PR 2483

Reviewed `https://github.com/amdpilot-org/sglang/pull/2483` at exact commit
`56ea75a6caa285191b31e8f3051fa0ecb4972dba` against upstream issue
`https://github.com/sgl-project/sglang/issues/29857`.

## Verdict

Recommendation: **accept**. The candidate is test-only hardening, not a new
production fix. The recorded base already contains the issue-specific source
correction in `ModelConfig._config_draft_model`: Qwen3.5/Qwen3.6 wrapper draft
configs set `num_nextn_predict_layers = 1` on `hf_text_config`, which is the
object consumed by model-shape derivation. The candidate accurately preserves
that contract with focused tests.

The candidate's complete `TestEagleConfigurator` plus model-config tests passed
(14 tests). A controlled one-line reversion of the existing nested assignment
made the new wrapper regression fail, after which the checkout was restored.
Independent adversarial cases covered conditional, causal-LM, dense, MoE, and
InternS2 wrapper variants, including stale `64` values that must be overwritten.
All resolved to one MTP layer and reproduced the issue-specific arithmetic:
34,816 bytes/token corrected versus 163,840 bytes/token with the bad 64-layer
fallback.

## Scope and limitations

The assigned device is an AMD Instinct MI350X (`gfx950`) under ROCm 7.2, not the
reported NVIDIA RTX PRO 6000 Blackwell/CUDA system. The Qwen3.6-27B-NVFP4
weights were unavailable. Therefore no full-model serving run, NVIDIA runtime
reproduction, semantic check, or direct 50 GB VRAM recovery measurement is
claimed. No native sources changed, so no native rebuild was applicable. A
separate draft embedding/lm-head lifetime issue discussed in upstream PR 32468
can leave additional memory unavailable, but it is distinct from the reported
exact 5x KV cell-size inflation.

Raw outputs are retained in this directory.
