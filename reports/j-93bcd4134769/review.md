# Independent review of amdpilot-org/sglang PR 1189

Upstream issue: https://github.com/sgl-project/sglang/issues/36653

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1225

Candidate: https://github.com/amdpilot-org/sglang/pull/1189 at
`2cbfa20830d030560f04d9841dd75c2d2e7a42ea`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

**Accept**, with the qualification that this is a source-contract verification,
not an end-to-end reproduction on the reporter's system. The candidate fixes the
configuration loss that explains the reported tensor mismatch and preserves the
legacy DeepSeek and checkpoint-declared BF16 cases. I cannot mark the original
distributed serving failure fully resolved without its NVIDIA architecture,
two-node TP=2 topology, and checkpoint.

## Independent finding

The issue's proposed TP-sharding diagnosis does not match the failing dimension.
`FusedMoE._load_w13` shards `gate_up_proj` on its output dimension. The reported
dimension 1 mismatch, 4096 versus 2048, instead matches ModelOpt NVFP4 packing:
the implementation explicitly stores two FP4 input values per byte, making a
4096-wide hidden dimension occupy 2048 bytes.

On the recorded base, `Glm5NextForConditionalGenerationNextN` correctly resolves
a quantized layer to the supplied `modelopt_fp4` configuration, but
`DeepseekModelNextN.__init__` then unconditionally replaces that configuration
with `None`. A mocked construction through the real classes therefore gave the
draft decoder `quant_config=None`; the submitted regression failed on the base
with two failures and one pass.

At the exact candidate commit, the legacy DeepSeek exception moves to
`DeepseekV3ForCausalLMNextN._resolve_nextn_quant_config`, while the GLM subclass
retains ModelOpt FP4 unless its checkpoint ignore list declares the NextN layer
unquantized. This makes the draft decoder allocate ModelOpt-packed parameters
for the issue's stated quantized MTP layer.

## Validation

- Base: candidate regression produced `2 failed, 1 passed`, directly exposing
  loss of GLM's resolved FP4 configuration before decoder construction.
- Candidate: its focused regression produced `3 passed`.
- Candidate: the focused regression plus existing multimodal NextN coverage
  produced `5 passed`.
- Independent adversarial script passed cases for unrelated ignore entries, a
  dict-like quantization config, an exact BF16 NextN exclusion, `None`, a
  non-ModelOpt configuration, and the actual decoder-construction handoff.
- Fresh-process imports resolved both changed modules from `/job/repo/python`,
  so the checked-out candidate source, not an installed copy, was exercised.
- `git diff --check` passed. The candidate changes Python only; no native source
  changed and no native rebuild was applicable.

Raw command output, metadata, the exact candidate diff, and the independent test
script are retained in `reports/j-93bcd4134769/raw/`.

## Limitations and remaining counterexamples

The assigned device is a single AMD Instinct MI350X (`gfx950`) with ROCm 7.2.
The report requires two NVIDIA GB10 `sm_121` nodes and the
`LibertAIDAI/GLM-5.3-Flash-NVFP4` 320B checkpoint. Those weights and topology
were unavailable. Consequently I did not claim a complete model load, TP=2
distributed startup, GPU numerical comparison, generation correctness, or
performance validation. The deterministic tiny Llama fixture is not the GLM5
ModelOpt architecture and would not close this gap, so an unrelated serving
smoke was not used as proof.

The concrete remaining counterexample is the original command on the original
checkpoint/topology: it still needs to demonstrate that both ranks load all MTP
weights and generate successfully. BF16 upstream-checkpoint behavior and TP=1
full-checkpoint behavior also remain untested, as they did in the report.
