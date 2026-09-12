# Investigation of SGLang issue 33846

Upstream issue: https://github.com/sgl-project/sglang/issues/33846

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2720

The prepared base already contains the issue reporter's confirmed correction:
`KDAAttnBackend.needs_cpu_seq_lens` is `False`. This prevents
`decide_needs_cpu_seq_lens()` from making `FutureMap` publish the per-step CPU
sequence-length mirror solely for KDA. Before upstream PR 32219, KDA inherited
the base backend's `True`; the resulting synchronization was the D2H wait
identified in the report. The reporter subsequently tested the isolated
one-line change on the original 8xMI350X Kimi-K3/DSPARK workload for ten c=1
rounds without a hang. Those upstream comments and PR metadata are retained in
`raw/`.

This change was not duplicated. A focused regression now locks down both the
KDA/DSPARK decision and independent boundaries: a backend that consumes host
lengths, absent backend slots, TBO, and ngram. As a negative control, changing
the KDA flag back to `True` made two regression cases fail; the unmodified
prepared source passes all four.

The assigned MI355X (gfx950) ran the existing KDA Triton attention fixture. All
10 subtests passed against the independent pure-PyTorch recurrence, including
single-request extend cases, page boundaries, ragged batches, and bsz=1 decode.
This establishes real gfx950 kernel execution and numerical agreement, but it
does not reproduce or qualify the original full serving workload.

## Limitations

- Only one gfx950 GPU was assigned; the report requires TP8 on eight MI350X.
- Kimi-K3 MXFP4 weights and the DSPARK serving configuration were unavailable.
- The original approximately 3% nondeterministic hang therefore could not be
  reproduced locally, and the reporter's original-scale validation remains the
  evidence that the already-merged source correction resolves it.
- No native source changed and no native rebuild was needed.
