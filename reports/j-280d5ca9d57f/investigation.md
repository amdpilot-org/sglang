# Independent review of PR 1593

Candidate: https://github.com/amdpilot-org/sglang/pull/1593

Exact commit: `41d3846aea50aee5a478156a0797d6a7d3fa35dd`

Upstream issue: https://github.com/sgl-project/sglang/issues/35295

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1628

## Finding

Request changes. The candidate fixes the reported word-extension behavior and
punctuation-only growth in `StreamingASRState.update()`, including the three
counterexamples inherited from PR 1504. It does not apply equivalent
boundary-aware handling in `StreamingASRState.finalize()`.

This matters to the original contract rather than being a neighboring edge
case. `process_asr_chunk()` sets `full_transcript` and calls `finalize()` for
the final chunk. Both the HTTP chunked streaming implementation and the
realtime session call `process_asr_chunk(..., is_last=True)`.

After a prior chunk emits `hello world`, a final transcript of `hello world,`
returns `world,` and records `hello world world,`. The expected delta is `,`
and the expected accumulator is `hello world,`. The same duplicated-word
failure was independently observed for closing quote, Unicode ellipsis, em
dash, `?!`, and a fullwidth closing parenthesis.

## Revision and import verification

The failing-before checkout was the image-prepared branch at the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`; it did not differ from the
recorded base. The candidate was tested detached at exactly
`41d3846aea50aee5a478156a0797d6a7d3fa35dd`, then the checkout was returned to
`amdpilot/j-280d5ca9d57f` before this report was committed.

Tests used `PYTHONPATH=/job/repo/python` and confirmed the imported module was
`/job/repo/python/sglang/srt/entrypoints/openai/streaming_asr.py`. Thus the
candidate source, rather than an installed copy, was exercised.

The candidate changes only Python, tests, and report files. No native source
or generated native library changed, so a native rebuild was not applicable.

## Environment and limitations

The prepared environment has Python 3.12, Torch `2.11.0+rocm7.2`, HIP
`7.2.26015`, and one visible AMD Instinct MI350X (`gfx950`). No GPU execution
was used because the defect and counterexample are deterministic Python string
state transitions. No Qwen3-ASR weights were available or used, so model
transcription quality was not assessed. No live network server was launched;
the shared production final-chunk helper was exercised with a deterministic
fake tokenizer manager and adapter. This qualifies state and serving control
flow, but not model inference or end-to-end network transport.
