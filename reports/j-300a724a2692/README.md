# PP + TP tensor-dictionary corruption investigation

The base implementation sliced every divisible tensor before PP send and
all-gathered the slices across attention-TP ranks after receive. The baseline
four-process reproducer demonstrates that rank-distinct tensors become a mixed
tensor on both receiving ranks, while a replicated tensor remains correct.

The correction adds optional per-key allowlists to all tensor-dictionary
communication variants and makes the scheduler optimize only `hidden_states`
and `residual`, the established TP-replicated PP entries. Unknown model proxy
and auxiliary entries are sent whole.

See `result.json` for commands, outcomes, evidence paths, and limitations.
