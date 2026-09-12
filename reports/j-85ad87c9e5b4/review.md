# Correction-generation review

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/662 at
`6eaf6c114f3dd23ad901e47d2cf8e2532ade22a6`.

Independent review: https://github.com/amdpilot-org/sglang/pull/706.

Upstream issue: https://github.com/sgl-project/sglang/issues/22935

Mirror issue: https://github.com/amdpilot-org/sglang/issues/710

## Outcome

The candidate is rejected as a correction for the reported zero-hit behavior.
It changes no runtime source and its regression explicitly expects the original
`[A,B,C]` then fresh-prefill n-1 `[A,B]` lookup to return zero indices.

All three review counterexamples were independently confirmed on the exact
candidate commit before this report was prepared:

1. A compressed `[10,20,30]` leaf whose only Mamba slot is owned at depth 3 is
   split by lookup `[10,20]`; the new depth-2 node has no slot and the returned
   `device_indices` length is 0.
2. Divergence at depth 32 before the first owned depth-64 checkpoint returns 0.
3. The candidate contains only tests and reports. It provides neither a producer
   of arbitrary-depth recurrent-plus-convolution snapshots nor an executor path
   that retains deeper attention KV while replaying recurrent layers.

The positive controls matter: inserting a genuinely owned checkpoint at depth 2
returns 2 indices, and inserting one at arbitrary depth 37 returns 37 indices.
Thus the radix cache can represent and reuse an arbitrary-depth checkpoint when
the execution backend supplies the complete state. The defect is not safely
corrected by copying a descendant state onto a split parent: that state belongs
to a different token depth.

## Why no runtime patch is justified here

`mamba_value` indexes a whole model state slot, including recurrent and
convolution state across layers. Current tracking obtains recurrent intermediate
states only at backend-supported checkpoint boundaries, while convolution
tracking must capture the matching token window. The scheduler also advances all
layers from one common cached-token boundary. There is no current path that starts
attention at a deeper structural KV hit and separately replays recurrent layers.

Implementing either missing facility changes model execution semantics and needs
hybrid-Mamba weights for cold/warm output and logprob equivalence. No such weights
were prepared for this job. Per the investigation boundary, absence of those
weights is not evidence for a speculative source change. Consequently there is
no honest failing-before/passing-after runtime correction in this delivery; the
remaining limitation is recorded rather than mislabeled as fixed.

## Evidence

Exact-candidate focused suite:

```text
20 passed, 17 warnings in 14.14s
```

Independent MI350X actual-cache probe at the candidate commit:

```text
leaf_depth_3_n_minus_1_count 0
exact_depth_2_count 2
exact_depth_2_reference_equal True
arbitrary_depth_37_count 37
arbitrary_depth_37_reference_equal True
divergence_before_first_checkpoint_count 0
```

The same focused suite on the prepared delivery base passed 20 tests, and the
same GPU probe produced identical measurements. Raw outputs and the probe source
are retained beside this report.

## What this delivery preserves

The candidate's valid ownership rule is retained as a focused regression and
expanded with exact depth-2 and arbitrary depth-37 positive controls. These tests
prevent a tempting but incorrect split-state relabeling change while documenting
the precise capability an eventual checkpoint producer must provide.

## Remaining work

An actual correction requires one of:

- a backend-supported snapshot of complete recurrent and convolution state at
  the desired arbitrary prefix depth, followed by insertion at that exact depth;
  or
- a new execution mode that retains structurally matched attention KV while
  replaying recurrent layers from their last owned checkpoint.

Either direction requires supported hybrid-model end-to-end numerical validation.
No native source changed, so a native rebuild was not applicable.
