# Correction generation 2: TP constrained decoding

Candidate parent: https://github.com/amdpilot-org/sglang/pull/3167 at `efa4ba5a75b21b2867e55582e7b1b98f7cf2b81f`

Independent review parent: https://github.com/amdpilot-org/sglang/pull/3202

Upstream issue: https://github.com/sgl-project/sglang/issues/13809

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3205

## Outcome

Both review counterexamples reproduce against the candidate implementation. The candidate's valid ordinary-TP correction is preserved, but no additional source change is justified with the available architecture.

The reproducer invokes `GrammarManager.process_req_with_grammar` on a simulated non-entry rank. Candidate initialization selects `tp_grammar_entry_only = False` when speculative decoding is active or an attention context-parallel group is active. Each configuration therefore submits one grammar backend lookup and retains a real grammar future. Ordinary TP submits no lookup and installs `PlaceholderGrammarObject`.

Deleting either guard is not a safe correction:

- Speculative verification builds a mask for every draft-tree position by traversing and mutating the request's grammar FSM. The ordinary sampler broadcast synchronizes only its final token IDs; it does not synchronize speculative masks, accept lengths, accept indexes, intermediate tree decisions, or FSM state.
- Context-parallel request fanout covers orthogonal attention-TP and attention-CP groups. The candidate's grammar readiness group and sampler token collective cover only attention TP. Entry-only ownership across active CP therefore needs a combined ownership/readiness/token synchronization design and distributed validation.

Only one AMD Instinct MI350X was assigned. True speculative TP and context-parallel TP execution were unavailable, and the tiny Llama transport fixture cannot manufacture those distributed topologies. That absence is not used as evidence for a speculative source edit.

## Reproduction

```bash
/tmp/amdpilot-repo-j-3d84fe484c6b/venv/bin/python \
  reports/j-3d84fe484c6b/reproduce_counterexamples.py

/tmp/amdpilot-repo-j-3d84fe484c6b/venv/bin/python -m pytest -q \
  test/registered/unit/constrained/test_grammar_manager.py \
  test/registered/unit/sampling/test_sampling_batch_info.py \
  test/registered/unit/layers/test_sampler_grammar_tp_sync.py
```

The first command reports one lookup in both remaining configurations and zero in ordinary TP. The preserved focused suite reports 97 passing tests. The candidate's original failing-before/passing-after ordinary-TP evidence and two-process collective evidence remain under `reports/j-06841fd1be4d/`.
