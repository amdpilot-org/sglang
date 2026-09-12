# Anthropic tool-reference correction evidence

Upstream issue: https://github.com/sgl-project/sglang/issues/35692

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1526

Candidate parent PR: https://github.com/amdpilot-org/sglang/pull/1377

Independent review parent PR: https://github.com/amdpilot-org/sglang/pull/1487

Candidate commit reviewed: `bec2f7f5454321b5ce30b105b02a9cce0117c363`.

## Reproduction

The candidate code was applied unchanged to prepared base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. An adversarial template used
`tool.function.defer_loading` to filter schemas, assigned the string
`tool_reference` to an unrelated diagnostic variable, and rejected every
non-text content part.

The two focused tests failed before the correction:

- capability detection returned `True` instead of `False`;
- actual Anthropic conversion forwarded a structured reference and Jinja
  rendering raised `ValueError: Unexpected item type in content.`

The complete output is in `raw/failing-before.txt`.

## Correction

The candidate's valid generic fallback, deferred-schema gating, and native GLM
path are preserved. Native capability detection now requires an executable
Jinja comparison between `tool_reference` and a content item's `type`, in
addition to use of `defer_loading`. Both attribute (`item.type`) and bracket
(`item["type"]`) access are covered.

After the correction, the adversarial conversion renders the text fallback and
the referenced schema is unlocked without forwarding structured content. The
focused counterexample passed 2 tests, and the combined tool-reference and
Anthropic serving suites passed 77 tests plus 5 subtests. Raw output is retained
under `raw/`.

## Limitations

This is CPU-only request conversion and real Jinja rendering. No GPU execution
is relevant to this path. No model weights or full HTTP server were used, so
transport, generation, model semantics, and distributed execution are not
claimed. No native source changed, so no native rebuild was applicable. The
prepared interpreter does not contain `ruff`; its failed invocation is retained
in `raw/ruff.txt`.
