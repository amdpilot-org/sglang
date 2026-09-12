# Investigation report: j-f91eeceef600

Source issue: https://github.com/sgl-project/sglang/issues/36734

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1114

Outcome: `candidate_verified`

## Finding

The prepared base reproduces a concrete GLM-family mechanism for the reported
empty-answer symptom.  The GLM chat template opens `<think>` in the generation
prompt.  If the model skips thinking, emits its answer directly, and stops on
EOS without emitting `</think>`, the base parser cannot distinguish that valid
answer from a generation cut off by `max_tokens`.  It therefore returns the
answer only as `reasoning_content`, leaving OpenAI `content` empty.

Upstream PR https://github.com/sgl-project/sglang/pull/37644 was already open
with an evidence-based correction for this mechanism.  This branch carries
that exact candidate commit (`8a4151f6fd793d8de16269e4748b7bbb8be3d4a4`)
rather than creating a competing implementation.

The correction passes the engine finish reason into the reasoning parser and
only reclassifies an unclosed, template-opened GLM reasoning block when the
model stopped on its own EOS.  Length truncation, aborts, user-provided stop
strings, tool calls, explicitly closed reasoning, client-opened assistant
continuations, and non-GLM parsers retain their prior classifications.  The
behavior is configurable through `skipped_think_as_content`.

## Reproduction and verification

The candidate's tests were retained while its three production files were
temporarily restored to base `358c163250ad3b1f62939b01ce1314a0a31a0365`.
`TestSkippedThinkAsContent` then produced 27 failures and 2 passes.  After the
production patch was restored, the same class produced 26 passes plus 3
passing subtests.  Raw outputs are retained at:

- `/tmp/amdpilot-repo-j-f91eeceef600/failing-before-parser.txt`
- `/tmp/amdpilot-repo-j-f91eeceef600/passing-after-parser.txt`
- `/tmp/amdpilot-repo-j-f91eeceef600/full-targeted-tests.txt`

The complete focused parser and OpenAI chat/responses suites passed: 330 tests
and 146 subtests.  These cover both non-streaming and streaming response
construction.

## GPU qualification

The qualified deterministic tiny-Llama fixture and subreaper runner from
amdpilot-org/sglang PR 649 at commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315` were inspected and run with
`HIP_VISIBLE_DEVICES=0`.  The source-checkout server used the assigned AMD
Instinct MI350X (`gfx950`) and the Triton attention backend.  `/generate`,
OpenAI completions, batched generation, and streaming each returned HTTP 200;
the stream contained eight SSE data events.  GPU memory returned to the idle
298110976 bytes after the owned process group was terminated and descendants
were reaped.

Raw fixture/server evidence is retained under:

- `/tmp/amdpilot-repo-j-f91eeceef600/tiny-random-llama/fixture-manifest.json`
- `/tmp/amdpilot-repo-j-f91eeceef600/server-evidence/`
- `/tmp/amdpilot-repo-j-f91eeceef600/gpu-info.txt`
- `/tmp/amdpilot-repo-j-f91eeceef600/gpu-after.txt`

This GPU run qualifies transport and engine execution only.  The random tiny
Llama fixture has neither the GLM-5.1 nor Kimi-K2.5 architecture and cannot
validate LongBench V2 semantics.

## Limitations

The reported GLM-5.1 TP8 and Kimi-K2.5 workloads require model weights and
eight GPUs; only one assigned GPU was available and those weights were not
present.  Consequently the original stochastic, long-context model behavior
was not reproduced end to end.  The candidate is enabled by default for the
`glm45` parser because its template-opened reasoning makes EOS discriminating;
it does not claim to resolve every possible cause of empty Kimi-K2.5 output.
No native library change or rebuild was involved.
