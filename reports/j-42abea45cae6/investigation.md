# Correction generation 2 investigation

Reviewed candidate PR https://github.com/amdpilot-org/sglang/pull/1754 at exact
commit `2ce6f8e16316f58a3f88569030a95778238a0fec` and the independent review PR
https://github.com/amdpilot-org/sglang/pull/1857 against prepared base
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

The candidate's two commits were preserved. Before further source changes, the
review counterexamples were independently reproduced with the prepared Python
environment. Mistral canonical arrays worked only with the literal `", "` and
failed for valid JSON separators `","`, `",\n"`, and `",\t"`, returning only
the first valid call and exposing the remaining calls as normal text. Hermes
whole and coarse batches leaked one closing `</tool_call>` tag for valid-only
input and two tags when an unknown call was inserted.

The correction changes Mistral's JSON-object separator to the comma alone, so
the JSON parser handles any legal following whitespace, and routes Hermes text
before a subsequent opening tag through its existing closing-tag cleaner.
Regression coverage includes all four separators and whole/coarse Hermes
batches containing an unknown call.

After the correction, the reproduction preserves both valid calls, drops only
the unknown call, and emits no normal text for every case. The focused file
passes 18 tests; the complete function-call unit directory passes 556 tests and
21 subtests.

This is deterministic pure-Python parser behavior. No GPU, model weights,
serving path, model architecture, native build, or distributed execution was
needed or claimed.
