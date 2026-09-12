# Independent review of PR 1263

Upstream issue: https://github.com/sgl-project/sglang/issues/37215

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1297

Candidate: https://github.com/amdpilot-org/sglang/pull/1263 at exact commit
`c26eb16eb26ef1952b1cf882ceeb3521d808854b`.

## Recommendation

Accept. The candidate fully resolves the original issue's reported
single-node `DP=8, TP=1` contract.

This is a source fix, not test-only hardening. Each reported DP replica has a
singleton process group, and the candidate removes its TCP listener entirely
by passing an in-process `HashStore` to PyTorch. It also corrects explicit
`--nccl-port` handling to derive `30101..30108`, retains non-ephemeral
allocation where a multi-rank group still requires TCPStore, and rejects
missing/exhausted safe-port ranges instead of restoring the old fallback.

## Failing-before evidence

The prepared checkout exactly matched the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

An independent real PyTorch probe held a listener on the selected address and
called SGLang's `init_distributed_environment` for a one-rank Gloo group. The
base attempted TCPStore bind and failed with `DistNetworkError`, code `-98`,
`EADDRINUSE`. Separately, eight `PortArgs.init_new` calls for explicit port
30101 returned eight copies of 30101. The base also lacked a dedicated
fail-closed rendezvous allocator.

These reproduce the original failure mechanism and all three counterexamples
recorded by the prior independent review.

## Candidate evidence

The same occupied-port singleton probe passed at the exact candidate because
the process group used `HashStore`. The explicit-port probe returned
`[30101, 30102, 30103, 30104, 30105, 30106, 30107, 30108]`. Mocked missing
procfs and full-range exhaustion both raised `RuntimeError` and did not fall
back to `bind(0)`.

The focused candidate suite passed 41 tests. A broader run produced 262 passes
and two unrelated failures: this ROCm checkout intentionally rejects
deprecated prefill context parallelism while those tests expect later
validation. Neither failing test covers a changed line or the issue contract.

The import paths were verified as `/job/repo/python/sglang/...`; PyTorch came
from `/opt/venv/lib/python3.12/site-packages/torch`. A real NCCL-backend test on
the assigned AMD Instinct MI355X (`gfx950`, ROCm 7.2) initialized the singleton
group while the advertised port was occupied and completed GPU `all_reduce`
with value 7.0.

No C/C++/HIP/CUDA source changed, so there was no native component to rebuild.
All five changed production Python modules passed `compileall`.

Raw outputs and the reusable adversarial probe are preserved outside the
revision-switching checkout at `/job/review-evidence-j-246837d56870/`.

## Scope and remaining limitation

The host does not provide eight NVIDIA H800 GPUs, CUDA, or the reported model
weights, so the exact full serving launch was not possible. The deterministic
tests directly exercise the failing TCPStore boundary and the DP port mapping,
and the assigned GPU qualifies the singleton NCCL execution path only.

For topologies outside the original report (`TP>1` or `PP>1`), multi-rank
groups still require TCPStore. Non-ephemeral selection prevents the reported
automatic ephemeral-port reuse, but cannot defeat an intentionally competing
listener that binds during the remaining close-before-bind interval.
