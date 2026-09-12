# AITER workspace budget investigation

The prepared base (`358c163250ad3b1f62939b01ce1314a0a31a0365`) did not
contain the solution proposed in upstream PR 18263. It passed the whole profiled
budget into KV pool sizing and derived `max_running_requests` afterward, while
the legacy AITER backend allocated a workspace proportional to that request
count outside the budget.

`reproduce_before.log` records the failing-before result from the actual
`KVCacheConfigurator._resolve_memory_pool_config`: the issue-shaped fixture
overcommitted a 219,721,119,039-byte budget by 69,793,139,393 bytes.
`reproduce_after.log` records the same fixture after the correction, with 79,167
bytes of headroom.

The implementation uses one workspace-size function for both budgeting and the
actual `torch.empty` call. Rather than assuming every pool is linear in a single
cell size, it binary-searches the existing pool configurator and request
resolver. This covers alignment, explicit request limits, user token caps, and
architecture-specific fixed pool costs. The reservation is skipped for MLA,
SWA (which forces unified attention), and explicitly enabled AITER unified
attention because those paths do not allocate this legacy workspace.

The assigned gfx950 GPU check is retained in `gpu_workspace_check.log`. It is
not a substitute for the unavailable Qwen weights or the report's MI355X, and
no full-model serving reproduction is claimed.
