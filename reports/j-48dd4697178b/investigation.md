# Independent review of amdpilot-org/sglang PR 1754

Reviewed exact candidate commit `2ce6f8e16316f58a3f88569030a95778238a0fec`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the
contract in upstream issue 34677.

## Recommendation

Request changes. The candidate is a substantial partial fix, but it does not
fully resolve the original issue.

On the recorded base, the supplied review reproduction independently reproduced
the three known failures: Mistral returned only the first valid call and exposed
the remainder as normal text; all six detectors lost the trailing valid call
when the final chunk contained an unknown call followed by a valid call; and a
coarsely chunked JSON array retained the trailing valid call in `_buffer`.

At the exact candidate commit, the candidate's focused tests passed (256 tests
and 4 subtests), the full function-call suite passed (551 tests and 21
subtests), and its reproduction passed for the particular separators it used.
The candidate correctly preserves calls for JSON-array, Qwen25, Hermes,
Llama32, Trinity, and Mistral inputs covered by those tests.

However, Mistral canonical-array streaming remains dependent on the literal
separator `", "`. JSON permits whitespace after a comma to be absent or to be
a newline/tab. With one-character chunks, the valid canonical arrays using
`","`, `",\n"`, or `",\t"` still return only `get_weather`; the unknown call
and trailing valid `get_time` call are exposed as normal text. This is the same
original-issue contract violation, not an unrelated smoke. The added Mistral
test only covers `", "`.

The independent matrix also observed that coarse/whole Hermes batches with an
unknown call add an extra `</tool_call>` to normal text compared with the
valid-only baseline. This is secondary evidence of remaining collateral text
corruption; the Mistral valid-JSON counterexample is sufficient by itself for
the request-changes recommendation.

## Environment and scope

Both revisions imported `sglang` and the changed detector modules from
`/job/repo/python`, using
`/tmp/amdpilot-repo-j-48dd4697178b/venv/bin/python`. No native files changed in
the candidate, `repository-environment.json` declares no native component, and
no native rebuild was applicable. The parser bug is deterministic pure Python,
so no GPU, model weights, HTTP server, architecture-specific model behavior, or
distributed execution was exercised or claimed.

Raw outputs are retained in `raw/`. The checkout was returned to
`amdpilot/j-48dd4697178b` at the recorded base before this report was committed.
