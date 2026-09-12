# Consolidated correction

Candidate parent: https://github.com/amdpilot-org/sglang/pull/2549 at `328bee322dc98cd4aed9f0cfa04f02bc4fcf5b77`

Independent review parent: https://github.com/amdpilot-org/sglang/pull/2598

The exact candidate was imported from a detached worktree and independently
reproduced both review counterexamples. Its matcher accepted any underscore
remainder after an MLA suffix, so clearing `_model` deleted both
`_model_variant` and `_model_0_1` foreign deployments.

The consolidated matcher retains the candidate's SHA-256 anchor, scoped clear,
component cleanup, and refusal of empty/degenerate scopes. It now recognizes
only the component suffix grammar emitted by `HiCacheNixl`, which preserves the
two reviewed foreign deployments. Commands and raw outputs are recorded in
`result.json` and `raw/`.
