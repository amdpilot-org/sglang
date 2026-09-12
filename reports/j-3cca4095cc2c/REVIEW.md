# HiCache host-memory correction evidence

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/830 at
`3d10c70a56be17ed89300d26ffb77faf67ad992e`.

Independent review: https://github.com/amdpilot-org/sglang/pull/921.

The candidate's TP-local `MIN` collective fixes staggered ranks inside one TP
group, but its synchronization domain does not match the host-wide divisor.
The four-process failing-before regression creates two co-located TP-like Gloo
groups. The first group receives 22.5 GiB per rank, while the later group
receives 12.5 GiB and rejects a 20 GiB pool although four such pools plus the
10 GiB reserve fit in the initial 100 GiB.

The consolidated correction takes one world-group synchronized snapshot at the
first, common HiCache pool check and caches it. Later pools use per-process
cumulative reservations against the fixed per-rank share, so their checks do
not depend on live-memory timing and do not require collectives whose call
counts could differ across pipeline stages. The corrected four-process run
gives all ranks 22.5 GiB initially; the later group sees 2.5 GiB remaining for
its 2 GiB sidecar without another collective.

The exact 8xH200 GLM-5.2-W4AFP8 serving workload was unavailable. The prepared
environment exposes one AMD Instinct MI355X (`gfx950`) and no model weights.
No model-serving, H200, multi-node, or GPU execution claim is made. The guard
still intentionally uses the existing equal per-rank host share, so workloads
with materially unequal per-rank host-pool requirements can remain
conservative even when their aggregate allocation would fit.
