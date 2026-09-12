# Independent review of PR 3244

Candidate: https://github.com/amdpilot-org/sglang/pull/3244  
Exact commit: `87e34445ba8ea528d5d65faab47853f575bf8846`  
Upstream issue: https://github.com/sgl-project/sglang/issues/13809  
Mirror issue: https://github.com/amdpilot-org/sglang/issues/3248

## Recommendation

`request_changes`. The candidate is a useful partial fix for ordinary tensor
parallelism, but it does not fully implement the original feature contract.
It intentionally restores the compile-on-every-rank and mask-on-every-rank
path when speculative decoding is enabled or an attention context-parallel
group is active.

## Independent findings

The recorded base commit reproduced the original defect on a simulated
non-entry TP rank: processing one JSON-schema request called
`get_cached_or_future_value` once and retained a `Future` grammar object.

At the exact candidate commit, the same probe showed:

```text
ordinary_tp_non_entry: lookup_calls=0, PlaceholderGrammarObject
speculative_tp_non_entry: lookup_calls=1, Future
context_parallel_tp_non_entry: lookup_calls=1, Future
```

Thus ordinary TP gains entry-rank-only compilation and masking plus an
authoritative token broadcast, while the two reviewed counterexamples remain.
The candidate's focused regression suite passed (97 tests), compileall passed,
and an independent single-GPU mask/argmax calculation matched its direct
allowed-token reference.

## Environment and architecture limits

The prepared interpreter imported `sglang` and all inspected changed modules
from `/job/repo/python`, using Torch 2.11.0+rocm7.2. Exactly one AMD Instinct
MI350X was visible. Consequently, real multi-rank TP, speculative TP, attention
context-parallel TP, distributed serving, and performance reduction were not
executed. The counterexamples are source-path/behavioral reproductions, not a
claim of distributed numerical validation. The tiny Llama fixture cannot
qualify these unavailable distributed architectures and was not used as a
substitute.

No C/C++/HIP/native files differ from the base, and repository metadata has no
task-specific native artifact, so no native rebuild was applicable.

## Commands

See `result.json` and `raw/evidence.txt` for exact commands and outputs.
