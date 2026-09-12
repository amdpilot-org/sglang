# Independent review of amdpilot-org/sglang PR 1595

Candidate reviewed: `f19a67ec7f042a4c7836ca468b50e99926a76987`

Upstream issue: https://github.com/sgl-project/sglang/issues/34677

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1631

## Recommendation

Request changes. The candidate is a real partial fix, not merely test-only
hardening: it fixes the issue's exact JsonArrayParser and Qwen25Detector
one-character reproduction and preserves indices/state in those cases. It does
not fully satisfy the original contract across the stated affected formats and
stream boundaries.

## Reproduction and verification

The prepared checkout was exactly the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365` on branch
`amdpilot/j-f9d0a2f9759f`. With `PYTHONPATH=/job/repo/python` and the prepared
interpreter, the issue's exact character-at-a-time reproducer produced:

```text
{}
{0: {'name': 'get_time', 'arguments': '{"city": "Tokyo"}{"tz": "JST"}'}}
```

After temporarily checking out the exact candidate, import inspection showed
both `base_format_detector` and `json_array_parser` loaded from
`/job/repo/python/sglang/srt/function_call/`, not an installed wheel. The same
reproducer then preserved both valid calls with indices 0 and 1. The two focused
unit files passed: 248 tests and 4 subtests.

## Remaining counterexamples

1. Mistral canonical-array streaming is not fixed. With one-character chunks
   for `[TOOL_CALLS] [weather, unknown, time]`, only weather is returned. The
   remainder, including the unknown and valid time call, is surfaced as normal
   text. `MistralDetector.parse_streaming_increment` delegates to the base only
   while its marker remains buffered; after the first call it handles the
   separator itself and flushes it as text. The candidate neither changes that
   override nor tests Mistral, despite Mistral being in the original issue's
   affected-format list.

2. Preserved trailing bytes can remain stranded when the stream ends. For each
   of JsonArrayParser, Qwen25Detector, HermesDetector, Llama32Detector,
   MistralDetector, and TrinityDetector, first stream a valid call character by
   character, then supply a final chunk containing a complete unknown call and
   a trailing valid call, followed by `finish()`. No trailing valid call is
   returned. Its bytes remain in `_buffer` (except Mistral's separate text-flush
   behavior) because the unknown-call branch returns immediately and the base
   `finish()` is a no-op. This directly violates the requirement to retain calls
   batched behind the unknown call.

3. Coarser chunking exposes the same lack of buffer draining. For example, a
   31-character JsonArrayParser stream returns the trailing tool's name but
   leaves its arguments and closing array buffered at end-of-stream.

The candidate's whole-wire test calls the parser with two artificial empty
increments after the model output. That can drain some state but is not evidence
that the normal end-of-stream path does so.

## Scope and environment

This is pure Python parsing. The candidate changes no C++/HIP/native source, so
no native rebuild was applicable. No GPU, model weights, HTTP server, model
architecture, semantic accuracy, or distributed workload was required or
claimed. Testing used the prepared Python environment with Torch
`2.11.0+rocm7.2`; GPU execution was intentionally not performed because it
cannot add evidence for this parser contract.

Raw command output and the standalone reproducers are retained outside the
revision-switched checkout under `/job/review-evidence-j-f9d0a2f9759f/`.
