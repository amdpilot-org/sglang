# Independent review of PR 3465

Candidate: https://github.com/amdpilot-org/sglang/pull/3465 at
`205fa94eae23691877523b111d46d25ccc4ae1a3`

Upstream issue: https://github.com/sgl-project/sglang/issues/35582

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3468

Recommendation: accept.

The recorded base independently reproduced the original mismatch through the
actual prompt rendering and legacy multimodal loading boundary: two complete
Qwen vision markers were rendered for one real PNG, and loading raised the
empty-detail RuntimeError after exhausting the single image.

The exact candidate separates the model-faithful render from a loader-only
render. Literal markers in client text remain exact in the string encoded for
the model, while only the loader copy is neutralized. Adjacent text parts are
coalesced before neutralization, so markers split across part boundaries do not
reappear when the template concatenates them. The loader sees one attachment
marker per actual image and successfully decodes the PNG.

Independent adversarial coverage extended the candidate tests to system text,
tool text, repeated markers, and a marker split across three adjacent user text
parts. It found no remaining counterexample within the original parser and
loader contract. Structural malformed image references are rejected by request
validation, and text-only marker spelling remains unchanged.

Qwen3.8-27B weights and its full serving architecture were unavailable, so no
end-to-end OpenAI HTTP generation or semantic inference is claimed. The tests
qualify the checked-out serving prompt-rendering path, real PNG decoding, and
legacy multimodal loader boundary. The tiny Llama fixture was not used because
it cannot qualify Qwen-VL behavior. No GPU execution was needed or claimed.
No native source changed, so no native rebuild was applicable.
