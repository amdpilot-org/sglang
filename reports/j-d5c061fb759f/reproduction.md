# Qwen-VL literal marker correction

Candidate: https://github.com/amdpilot-org/sglang/pull/3423 at `f2e033dc00060fbab489fe5225dfb8fcdf061385`

Independent review: https://github.com/amdpilot-org/sglang/pull/3431

On the exact candidate, the focused candidate regression passed, including a
real 1x1 PNG decoded through `BaseMultimodalProcessor.load_mm_data`. Independent
exact-preservation assertions then failed in three concrete cases: text-only
user content, mixed image/user text, and mixed image/tool-result text. The
candidate passed spaced spellings to the chat template instead of the literal
`<|vision_start|><|image_pad|><|vision_end|>` submitted by the client.

The correction leaves text-only requests completely untouched. For a Qwen-family
vision request with an actual image, the chat template first renders the exact
client content. A separate deep copy is then neutralized and rendered for the
loader-facing prompt, so the legacy prompt scanner sees only the authoritative
attachment marker. This retains the candidate's one-image/one-marker loader
behavior without rewriting the ordinary content before its normal template
render.

Raw commands and outputs are retained in `evidence/`:

- `candidate-focused.txt`: candidate-defined regression passing.
- `counterexamples-failing-before.txt`: three independent preservation failures.
- `counterexamples-passing-after.txt`: four focused tests and nine subtests passing.
- `broad-tests.txt`: 160 tests and 92 subtests passing.
- `gpu-check.txt`: one MI350X exact-reference matrix multiplication.

Qwen3.8-27B weights were unavailable. Therefore this qualifies the request
models, render boundary, legacy-loader selection, and real image decode, but not
a full Qwen3.8-27B server launch or semantic inference. The tiny Llama fixture
cannot qualify the Qwen-VL architecture and was not used as substitute evidence.
