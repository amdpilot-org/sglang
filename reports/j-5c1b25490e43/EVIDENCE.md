# Correction evidence

Candidate: https://github.com/amdpilot-org/sglang/pull/2901 at
`ad0162d0c4c667916f85904b30cd356d38feee84`

Independent review: https://github.com/amdpilot-org/sglang/pull/2972

The candidate's 10 tests passed, but an independent matrix reproduced host-RAM
fallbacks for missing, raising, zero, negative, and invalid Metal limits. The
candidate returned 64 GiB total and 60 GiB available in every case. Its exact
registered-test hook also failed because a unit test registered an MLX suite.

After correction, 23 focused tests pass. The same independent matrix observes
an explicit fail-closed `RuntimeError` for every unsafe total/available query,
and the complete applicable pre-commit run passes, including registered-test
taxonomy.

Raw command output is retained at
`/job/reports-j-5c1b25490e43-raw/`. This host has no Apple MPS device, so native
Metal values and real UMA pressure remain unverified; no percentage-of-host-RAM
heuristic was introduced.
