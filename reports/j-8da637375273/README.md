# Kimi-K3 required-tool timeout investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/37430

Mirror issue: https://github.com/amdpilot-org/sglang/issues/961

## Result

The prepared source at `358c163250ad3b1f62939b01ce1314a0a31a0365`
already contains a directly related correction merged as upstream PR #34881,
`Stop losing Kimi-K3 tool calls to reasoning, constraint conflicts, and
truncation` (merge commit `307a90f6d3c73ad4cbf3d90e817bc23617580475`).
No additional source correction is justified by the available evidence.

The original 2-prefill/2-decode, TP8/DCP8, DSPARK Kimi-K3 deployment could not
be recreated on the assigned single gfx950 GPU. The Kimi-K3 and draft-model
weights were not available, and the report does not identify the SGLang commit
that produced its failures. Consequently, this report does not claim a full
model, semantic-accuracy, speculative-decoding, or distributed-topology
reproduction.

## Issue-specific evidence

The issue follow-up reports responses containing raw Kimi control markers,
missing structured tool calls, and runaway generation. PR #34881 predates the
issue report and fixes paths with the same concrete failure shape:

- a Kimi-K3 tools channel emitted before the reasoning close marker was
  classified as reasoning and silently lost;
- required native-format output was incorrectly sent through a JSON-array
  fallback when a complete native call was absent;
- truncated tool sections vanished at stream completion; and
- required tool calls could conflict with another output constraint.

The regression from that PR was copied unchanged into a temporary worktree at
its exact parent, `8bb106cee94c1ccb5625cdb78ef0ee3e0cad3918`. With imports resolved
from that worktree, the non-streaming case and the single-chunk streaming
boundary failed because the entire tools channel appeared in
`reasoning_text`. Three smaller streaming chunk boundaries happened to pass,
which is useful independent evidence that this defect was chunk-boundary
sensitive rather than a universal parser failure. See
`raw/prefix_reasoning_regression.log`.

On the prepared current source, all five versions of that regression pass.
The full Kimi-K3 detector and reasoning-parser files pass 75 tests, including
independent chunk sizes, multiple calls, plain text, incomplete markers, and
truncated tool sections. Four focused serving tests also pass, with 12
subtests covering native-parser fallback avoidance, truncated native calls,
required/output-constraint conflicts, and the `auto` boundary. See the two
`raw/current_*.log` files.

## GPU and fixture decision

`raw/gpu_inventory.log` records the assigned AMD Instinct MI350X (`gfx950`).
No GPU inference was run for this result. The investigated correction is in
CPU request encoding/parsing, and the required Kimi-K3 2P2D topology needs far
more than the assigned single GPU.

The deterministic tiny-Llama fixture and subreaper runner from
amdpilot-org/sglang PR #649 at
`f1d603677ca76a9ea21124a544e405c5b0cbd315` were inspected. They launch a
two-layer random Llama and probe generic `/generate` and completion transport.
They do not send chat tools and cannot qualify Kimi-K3 special-token encoding,
the Kimi-K3 parser, DSPARK, TP8/DCP8, or PD disaggregation. Running that fixture
would therefore be an unrelated startup/transport smoke and was deliberately
not used as evidence that this issue is solved.

## Reproduce the focused checks

Current source:

```bash
/tmp/amdpilot-repo-j-8da637375273/venv/bin/python -m pytest -q \
  test/registered/function_call/test_kimik3_detector.py \
  test/registered/unit/parser/test_kimik3_reasoning_parser.py

/tmp/amdpilot-repo-j-8da637375273/venv/bin/python -m pytest -q \
  test/registered/unit/entrypoints/openai/test_serving_chat.py \
  -k 'required_tool_choice_skips_json_fallback_for_native_parser or truncated_native_tool_call_logs_and_drops or required_tool_choice_rejects_conflicting_output_constraint or auto_tool_choice_keeps_response_format_without_raising'
```

The pre-fix log records its resolved module path and assertion failures. Its
temporary worktree was outside the repository at
`/tmp/amdpilot-repo-j-8da637375273/prefix`.
