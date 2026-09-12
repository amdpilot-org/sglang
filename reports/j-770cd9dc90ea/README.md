# DSPARK invalid-probability investigation

The prepared source at `358c163250ad3b1f62939b01ce1314a0a31a0365`
already contains a targeted recovery for the failure signature in the issue.
In the eager DSPARK draft sampling path, `torch.softmax` output is passed through
the unconditional `dspark.draft.probs` invariant recovery before
`torch.multinomial`. An all-NaN probability row is replaced by a token-0 one-hot
row. The invariant's signal level is optional, but its recovery is always active.

The added regression demonstrates the failing-before condition directly:
`torch.multinomial` rejects an unrecovered all-NaN probability tensor with the
reported "probability tensor" error. It then covers the existing fix for two
independent sources of a NaN softmax row (all `-inf`, and a positive `inf` logit),
and verifies that valid probability rows are unchanged.

The GPU evidence in `raw/gpu_dspark_probability_recovery.txt` ran this recovery
and the subsequent multinomial on the assigned AMD Instinct MI350X (`gfx950`),
with a float64 CPU softmax used as an independent reference for the valid row.

## Limitations

The reported DeepSeek-V4-Flash-0731 weights, eight H100 GPUs, CUDA runtime, TP=8,
and week-long production traffic were unavailable. Therefore this does not
claim full-model, CUDA, distributed, semantic-accuracy, or intermittent-load
reproduction. The available single-GPU fixture validates the exact probability
containment immediately before the implicated multinomial operation.
