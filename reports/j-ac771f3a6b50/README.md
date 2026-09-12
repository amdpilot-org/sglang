# Investigation: heterogeneous TP for non-MLA SWA P/D

Upstream issue: https://github.com/sgl-project/sglang/issues/38552

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2671

## Result

The reported behavior is reproducible at the exact NIXL dispatch point in the
prepared source. It is an explicit unsupported-architecture guard, not an
incidental transport failure. The same guard remains on mirror `main` at
`bd45cd50ca900dd821f829ca9adfbf9aa3336bda` after checking current related
changes.

No source fix is proposed. Removing the guard alone would be unsafe: the main
non-MLA KV path selects `send_kvcache_slice` for heterogeneous TP and builds
head-aware source/destination transfer lists, while `maybe_send_extra` sends the
SWA sub-pool through `_send_kvcache_generic`, which copies whole per-rank cache
items. Since the SWA pool is itself an MHA KV pool with TP-dependent head
geometry, whole-item copying does not implement the required gather/scatter or
replication across TP ranks.

## Reproduction

Run from the repository root:

```bash
/tmp/amdpilot-repo-j-ac771f3a6b50/venv/bin/python \
  reports/j-ac771f3a6b50/reproduce_heterogeneous_swa_guard.py
```

The fixture calls the actual `NixlKVManager.maybe_send_extra` implementation.
It does not construct a NIXL transport because the reported exception is raised
before transport. It verifies:

- non-MLA SWA with prefill TP 2 and decode TP 4 reproduces the reported error;
- non-MLA SWA with equal TP reaches one generic transfer call;
- MLA SWA with different TP reaches one generic transfer call.

Raw output is retained in `raw/reproduction.txt`; source/current-main evidence
is in `raw/source_evidence.txt`; device discovery is in `raw/environment.txt`.

## Limitations

Only one AMD Instinct MI350X (`gfx950`) GPU was assigned. Heterogeneous TP needs
multiple ranks, so no GPU transfer or full P/D serving execution was possible.
GPT-OSS-120B weights were not available. The tiny Llama HTTP fixture is not used
because it cannot qualify GPT-OSS's hybrid SWA architecture or a heterogeneous
TP workload. No claim is made about model semantics, NIXL multi-rank execution,
or a full deployment.

