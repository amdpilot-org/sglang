# NIXL P/D role replacement investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/33789

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1737

Outcome: **not_reproduced** on the available constrained topology.

Current main (`358c163250ad3b1f62939b01ce1314a0a31a0365`) served requests after
both Decode-only and Prefill-only replacement in a real NIXL P/D deployment.
No production source change is proposed without a failing current-main
reproducer.

## Related fix review

Upstream PR https://github.com/sgl-project/sglang/pull/34289 is still open and
is not contained in this checkout. It targets Prefill replacement behind a
stable service address, using an instance-id header to detect a changed
Prefill. Its description explicitly leaves Decode replacement unresolved.
The source issue also contains a current-main report that Decode replacement
recovered in same-address and endpoint-changing tests.

## Reproduction evidence

The probe used the campaign-qualified deterministic tiny Llama fixture from
amdpilot-org/sglang PR 649 at exact commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315`. Weights were generated under
`/tmp/amdpilot-repo-j-3ec1a4e9c131/tiny-random-llama`, outside the checkout.

One Prefill, one Decode, and the mini router ran from this checkout with NIXL
UCX on loopback, sharing the assigned AMD Instinct MI355X (`gfx950`). The
successful request sequence was:

| Phase | HTTP | Completion IDs | Elapsed |
|---|---:|---|---:|
| baseline | 200 | `12,100,124,43,124,43,124,51` | 4.266 s |
| Decode replaced, Prefill retained | 200 | same | 1.721 s |
| both roles restarted (control) | 200 | same | 1.689 s |
| Prefill replaced, Decode retained | 200 | same | 0.049 s |

The independent Transformers GPU reference produced the same completion token
IDs. Full requests, responses, commands, PIDs, ports, and role logs are under
`evidence/single_gpu_nixl_skip_warmup/`; the independent reference is
`evidence/transformers_reference.json`.

An initial run without `--skip-server-warmup` could not reach the replacement
sequence because the Decode health endpoint remained 503 during built-in
disaggregation warmup. Those logs are retained under
`evidence/single_gpu_nixl/` and are not counted as evidence for or against the
reported replacement bug.

## Limitations

This was not the reported two-H100, Qwen3-0.6B, NIXL 1.3.2 deployment. Both
roles shared one gfx950 and used loopback UCX with a tiny random Llama. The
fixture validates HTTP transport, real engine execution, KV transfer, and role
process replacement only; it cannot qualify Qwen3 semantics, independent GPU
failure domains, Kubernetes Service behavior, cross-node UCX, in-flight
replacement, or NIXL 1.3.2. The open Prefill replacement candidate specifically
depends on stable-service/draining-process behavior that this direct loopback
address did not reproduce.

## Commands

```bash
/tmp/amdpilot-repo-j-3ec1a4e9c131/venv/bin/python reports/j-3ec1a4e9c131/create_tiny_llama.py
/tmp/amdpilot-repo-j-3ec1a4e9c131/venv/bin/python reports/j-3ec1a4e9c131/run_nixl_replacement_probe.py
/tmp/amdpilot-repo-j-3ec1a4e9c131/venv/bin/python -m pytest -q test/registered/unit/disaggregation/test_nixl_backend_basic.py test/registered/unit/disaggregation/test_receiver_connection_pool.py
```
