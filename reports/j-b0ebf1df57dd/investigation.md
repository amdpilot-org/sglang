# Correction-generation investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/38452

Mirror issue: https://github.com/amdpilot-org/sglang/issues/684

Candidate PR: https://github.com/amdpilot-org/sglang/pull/542

Independent review PR: https://github.com/amdpilot-org/sglang/pull/668

Candidate commit: `4728bc00c490439134d4ddcc2782ad291effb5d9`

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Result

The independent review's counterexamples reproduce. The candidate is test-only and its new test passes, but it does not represent the source issue's state or scheduler path.

On this base, `UnifiedTreeNode.backuped` is derived from the Full component's non-null `host_value`. `_evict_host_leaf` evicts every component at all layers and removes the leaf from its parent. Consequently, normal device eviction followed by host eviction cannot leave the requested tree node both `evicted` and `backuped`.

The candidate test confirms the different behavior in its assertions:

- Its partial case expects `last_host_node == best_match_node`, manually computes the missing suffix, and calls `prefetch_from_storage` directly.
- Its complete case expects `last_host_node` to be root and directly submits the entire key.
- It never calls `Scheduler._prefetch_kvcache`, so it cannot expose an empty `new_input_tokens` slice caused by scheduler accounting.
- It sets `prefetch_threshold=1` and uses sequences only four pages long instead of validating the default 256-token threshold with a longer prefix.
- The candidate changes no production file.

The direct file-backend round trip, restored-value comparisons, repeated prefix, and unavailable-key miss are valid tests of storage itself. They do not demonstrate recall through a matched backup-only stub. Retaining that test while presenting it as an issue regression would preserve a false claim, so this correction records the limitation and makes no speculative production change.

## Evidence

The exact candidate test and the existing host-leaf eviction test were run with the prepared interpreter. Result: 22 passed, 16 skipped. PyTorch detected one AMD Instinct MI355X through ROCm 7.2. The complete raw output is retained at `/job/evidence-j-b0ebf1df57dd/candidate-tests.txt` in the job workspace.

No native source changed, so no native rebuild was applicable.
