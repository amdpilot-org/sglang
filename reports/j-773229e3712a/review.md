# Independent review of PR 2549

Candidate: https://github.com/amdpilot-org/sglang/pull/2549 at
`328bee322dc98cd4aed9f0cfa04f02bc4fcf5b77`

Upstream issue: https://github.com/sgl-project/sglang/issues/32693

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2499

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2557

## Verdict

Request changes. The candidate is a meaningful partial fix: it scopes clear calls,
refuses empty/degenerate suffixes, preserves unrelated files, and fixes the
reported suffix-at-tail collision. Its own five regression tests pass.

It does not fully provide the original issue's required per-instance isolation.
For an MLA deployment with suffix `_model`, the matcher accepts every valid cache
filename whose post-hash portion starts with `_model_`. Consequently it deletes:

- another MLA deployment named `model_variant` (`_model_variant`), and
- a non-MLA deployment of the same model (`_model_0_1`).

Those names are indistinguishable from the candidate's permitted
underscore-prefixed component grammar without additional namespace metadata.
The direct production `NixlFileManager.clear(suffix="_model")` reproduction
deleted both foreign files while correctly preserving the prior tail-overlap
case `_tenant_model` and an unrelated file.

## Evidence

- Recorded base `358c1632...`: the production file manager deleted the owned
  file, another deployment's file, and an unrelated file. This reproduces the
  original issue.
- Exact candidate `328bee3...`: its checked-in five-test regression passed.
- Exact candidate `328bee3...`: an independent prefix-overlap regression failed;
  `foreign_mla_prefix` and `foreign_mha_same_model` were both deleted.
- The prepared interpreter imported
  `/job/repo/python/sglang/srt/mem_cache/storage/nixl/nixl_utils.py`, confirming
  the checked-out source rather than a wheel copy was exercised.
- The candidate changes Python only. No native source or native artifact changed,
  so no native rebuild applies.

Raw outputs are retained under `raw/`.

## Limitations

This CPU-only filesystem defect was exercised directly through the production
file manager and routing. No GPU, model weights, NIXL plugin, HTTP server,
multi-rank process, or distributed mount was needed or used. Therefore this
review makes no GPU-transfer, model-architecture, semantic-accuracy, race, or
full serving-stack claim.
