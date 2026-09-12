# Correction-generation review of PR 2975

Candidate: https://github.com/amdpilot-org/sglang/pull/2975 at `e5c2189055820624395769913d2b1b94a7c09c56`

Independent review: https://github.com/amdpilot-org/sglang/pull/3082

Upstream issue: https://github.com/sgl-project/sglang/issues/13809

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3105

## Outcome

The two reported counterexamples reproduce. The candidate remains a valid fix for ordinary tensor-parallel constrained decoding, and those changes are preserved here, but it is not a complete implementation of the unqualified feature request.

`reproduce_counterexamples.py` invokes the checked-out candidate implementation on a simulated non-entry rank. Both configurations for which candidate initialization selects `tp_grammar_entry_only = False` submit one grammar backend lookup and receive a `Future`; ordinary TP selects the placeholder path and submits none.

Removing the two guards is not a justified correction:

- Speculative verification calls `build_grammar_vocab_mask`, which traverses and advances the real per-request grammar FSM over every draft-tree position. A non-entry placeholder cannot supply that state, and the normal sampler's final-token broadcast does not synchronize speculative accept lengths, accept indexes, or intermediate tree decisions.
- With DP attention/context parallelism, request distribution spans the combined attention CP and TP topology, while `GrammarManager.grammar_sync_group` and the sampler token collective are scoped only to `attn_tp_group`. Entry-only compilation across CP therefore requires a new combined ownership and synchronization design, not deletion of the CP guard.

Only one GPU was assigned. True speculative TP and context-parallel TP serving could not be executed, and the provided tiny model cannot create those missing distributed architectures. Per the task boundary, this report does not turn that absence into a speculative source change.

## Reproduction

```bash
/tmp/amdpilot-repo-j-109181fc89fe/venv/bin/python \
  reports/j-109181fc89fe/reproduce_counterexamples.py

/tmp/amdpilot-repo-j-109181fc89fe/venv/bin/python -m pytest -q \
  test/registered/unit/constrained/test_grammar_manager.py \
  test/registered/unit/sampling/test_sampling_batch_info.py \
  test/registered/unit/layers/test_sampler_grammar_tp_sync.py
```

The first command reports one lookup for each remaining counterexample and zero for ordinary TP. The focused candidate suite reports 97 passing tests.

## Retained evidence

Raw outputs are under `reports/j-109181fc89fe/raw/`. The candidate's original failing-before/passing-after evidence and two-process collective evidence remain under `reports/j-06841fd1be4d/`.
