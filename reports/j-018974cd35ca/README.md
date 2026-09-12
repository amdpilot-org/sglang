# Investigation of Mllama ambiguous multi-image behavior

Upstream issue: https://github.com/sgl-project/sglang/issues/8174

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2509

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Result

No SGLang correction was made. The reported semantic failure is not isolated to
SGLang: the upstream issue contains a maintainer reproduction using Hugging Face
`MllamaForConditionalGeneration` at checkpoint revision
`9eb2daaa8597bf192a8b0e73f848f3a102794df5`, with the same inability to describe
the two images correctly. The maintainer linked Meta's statement that Llama 3.2
Vision does not support multiple images reliably.

Current main also does not claim complete Mllama multi-image support. In
`python/sglang/srt/models/mllama.py`, the model currently passes
`cross_attention_mask = None` immediately after the explicit comment
`TODO: support multi-image by this mask`. Implementing that missing mask could
change which image features text tokens may attend, but there is no issue-specific
evidence that it would overcome the same failure in the Transformers reference.
It would therefore be a speculative neighboring change rather than a justified
fix for the reported output.

## Reproduction and limitations

The exact reported unittest command was run against the prepared checkout. It
fails during test discovery because `TestMllamaServer` is commented out as
flaky on current main, so the original serving test is no longer executable.
The model repository is manual-gated, and the prepared private Hugging Face cache
contains no copy of the checkpoint. Consequently, neither the reported SGLang
generation nor an independent Transformers generation could be rerun locally.

The assigned device was confirmed as one AMD Instinct MI355X (`gfx950`) with
PyTorch ROCm 7.2 access. No Mllama model execution occurred on it. GPU inventory
is retained only as environment evidence and is not presented as validation of
the issue.

## Retained evidence

- `exact-reproduction.log`: raw output from the exact unittest target.
- `upstream-8174.txt`: issue discussion retrieved with `gh issue view`.
- `mirror-2509.txt`: mirror issue discussion retrieved with `gh issue view`.
- `hf-model-api.json`: Hugging Face API metadata showing `gated: manual` and the
  checkpoint revision used by the upstream reference reproduction.
- `source-evidence.txt`: current implementation and test-registration excerpts.
- `gpu-environment.txt`: ROCm and PyTorch device inventory; not model execution.

## Boundary assessment

- Single-image Mllama behavior is outside the reported defect and was not treated
  as proof of multi-image correctness.
- A tiny Llama transport fixture cannot instantiate the Mllama vision architecture
  or test its image cross-attention, so it was intentionally not substituted.
- A startup smoke test would not measure the semantic output and was not used.
- Multi-node execution is irrelevant to this report and was not attempted.
