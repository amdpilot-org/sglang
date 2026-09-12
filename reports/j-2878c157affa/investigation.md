# Investigation: parser-less reasoning structural tags

Upstream issue: https://github.com/sgl-project/sglang/issues/36675

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1121

## Finding

At base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`,
`xgrammar_reasoning` was unreachable. `_get_reasoning_from_request()` returned
`False` whenever no reasoning parser was configured, while
`xgrammar_reasoning` additionally required the parser to be absent.

Changing the parser check from `is None` to `is not None` would regress the
ownership established by upstream PR 25676: when a reasoning parser is
configured, `ReasonerGrammarBackend` owns the reasoning prefix and the inner
tool structural tag must cover only the suffix.

The narrow failing case is a parser-less server with a detected reasoning
toggle in its chat template. A default-enabled template, or a default-disabled
template explicitly enabled by the request, prompts a reasoning prefix even
though SGLang does not parse that prefix into `reasoning_content`. The Qwen3
Coder structural tag must therefore allow arbitrary text through `</think>`
before requiring a tool call.

The fix separates the template reasoning state from the parser-owned output
state. `_get_reasoning_from_request()` retains its parser requirement, while
the XGrammar call uses the template state only when no reasoning parser owns
the prefix.

## Evidence

- `raw/regression-before.txt`: the new regression fails in the two enabled
  parser-less cases on the base implementation.
- `raw/regression-after.txt`: the same regression and the existing
  parser-owned boundary pass after the fix.
- `raw/qwen3-coder-structural-tag.txt`: the installed XGrammar Qwen3 Coder tag
  begins directly with `<tool_call>` when reasoning is false and includes an
  unrestricted prefix ending in `</think>` when reasoning is true.
- `raw/serving-chat-suite.txt`: the complete serving-chat unit file passes.
- `raw/gpu-sanity.txt`: one assigned AMD Instinct MI350X was used for an
  independent Torch CPU/GPU numerical sanity check. This Python control-flow
  fix does not require GPU execution.

## Limitations

Qwen3.5 model weights were unavailable and its registered test requires eight
GPUs. No full-model HTTP, semantic-accuracy, or distributed reproduction was
performed. The deterministic tiny Llama fixture is not suitable evidence for
Qwen3.5 template/parser behavior, so it was not substituted for the affected
architecture.
