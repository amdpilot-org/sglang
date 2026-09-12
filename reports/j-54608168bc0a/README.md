# Independent review of PR 2158

Candidate `e3d9b98c105efbf452559ebc69efae53b771f16b` was reviewed against
the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original open
issue. The recommendation is **accept**, as a verified narrow mitigation, but
`fully_resolves_original` is false.

On the base revision, the prepared interpreter imported SGLang from this
checkout and constructed a 32-worker grammar executor while `os.cpu_count()`
reported 256. An issue-specific assertion requiring at most 8 workers failed.
At the exact candidate revision, the same probe constructed 8 workers, the
candidate regression passed, and independent boundary and override cases
passed. The installed xgrammar import reports a `GrammarCompiler` default of
8 internal threads.

The change does not make automatic sizing cgroup- or affinity-aware. Under a
two-CPU affinity restriction, `os.cpu_count()` still reported 256 and the
candidate still selected 8 outer workers. Combined with xgrammar's default,
that permits up to 64 nested compiler threads. Operators can avoid this with
the new explicit override, but the original automatic container-awareness and
measured CFS/decode-stall claims remain unresolved and unverified here.

This checkout contains no candidate native-source changes, so no native rebuild
was applicable. The host exposed one AMD Instinct MI355X/gfx950 GPU, but the
reviewed path is CPU-only and no GPU execution was performed. The cgroup v2
controller reported `cpu.max=max 100000`; therefore actual CFS throttling and
model-serving decode latency could not be reproduced.

Raw command output and source/import paths are retained under `raw/`.
