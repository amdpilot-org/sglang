# Investigation evidence

Upstream issue: https://github.com/sgl-project/sglang/issues/34772

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1498

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The exact historical `should_offload(server_args, component_name=...)` call no
longer exists on this base. The residency refactor instead records the decision
through `ServerArgs.should_start_component_on_cpu`. However, the native
Transformers path did not consult that decision for non-quantized components
until after `from_pretrained` returned. Consequently a customized-load failure
could still materialize a large encoder under the process's active GPU device
before the common finalizer moved it to CPU.

Upstream PR https://github.com/sgl-project/sglang/pull/38594 was open during
investigation and identified the same current-main failure mode. It was not
merged into the prepared base. This change independently narrows the correction
to non-quantized native Transformers loading: CPU-start requests now receive a
CPU `device_map` during `from_pretrained`; resident and existing quantized paths
are unchanged.

Raw test and GPU-placement output is retained under `raw/`. The assigned GPU
check used a generated tiny T5 fixture outside the worktree at
`/tmp/amdpilot-repo-j-cad04e9dee25/tiny-t5-cpu-placement`.
