# Piggyback load report verification

This change restores the original issue's output-stream piggyback path on top
of the current `LoadSnapshot` implementation. Generation and embedding output
messages carry the scheduler snapshot; detokenization and multi-tokenizer
fanout retain it; and each HTTP worker caches the newest snapshot per DP rank.
`/v1/loads` merges that cache with the existing watch-mode reader, choosing the
newer timestamp and preserving the watch result for idle ranks.

The exact failing-before, passing-after, lint, fixture-generation, server, HTTP
response, and cleanup logs are retained outside the worktree under
`/tmp/amdpilot-repo-j-ee152f8ccad8/`. The complete machine-readable claims and
limitations are in `result.json`.

The live qualification used the deterministic tiny Llama fixture from
amdpilot-org/sglang PR 649 at commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315`. It proves transport and engine
execution on the assigned MI355X, not semantic accuracy or distributed routing.
