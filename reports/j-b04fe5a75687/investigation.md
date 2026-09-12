# DSpark compact confidence runtime-gamma investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/34023

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3339

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The prepared source still contained the reported defect. The DSpark worker resolves
its runtime gamma from `speculative_num_draft_tokens`, but
`DeepseekV4ForCausalLMDSpark.compute_confidence()` reshaped the post-HC tensor and
built its Markov context with the model's checkpoint-derived `self.gamma`.

Related upstream fixes were inspected before implementation:

- https://github.com/sgl-project/sglang/pull/34024 derives the runtime window from
  the post-HC tensor.
- https://github.com/sgl-project/sglang/pull/31016 passes runtime gamma into the
  confidence method explicitly.

Both remained open during this investigation. The implemented correction follows
the narrower tensor-derived approach from PR 34024: the affected tensor is the
authoritative source for the window already produced by the runtime draft path,
and no duplicate gamma parameter needs to be threaded through the planner.

Evidence is retained in this directory:

- `failing-before.log`: runtime gamma 3 and 7 fail against checkpoint gamma 5;
  the native gamma 5 boundary passes.
- `focused-tests.log`: the fixed regression plus DSpark argument-resolution tests.
- `gpu-numerical.log`: exact comparisons on the assigned gfx950 for gamma 3, 5,
  and 7.

No model weights were available. This result therefore does not claim full model,
decode-graph, HTTP serving, SM120, or multi-node/multi-GPU validation.
