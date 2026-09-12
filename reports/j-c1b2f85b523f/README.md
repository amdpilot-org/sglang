# Investigation: canonical Mamba radix-cache strategy in YAML

Upstream issue: https://github.com/sgl-project/sglang/issues/36326

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1207

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Outcome: `not_reproduced`

The prepared base already avoids the reported collision. The real
`ServerArgs` parser contains one action for `mamba_radix_cache_strategy`:
the canonical `_StoreAction` for `--mamba-radix-cache-strategy`. A YAML value
of `no_buffer` merges and parses successfully, as does the canonical CLI form.

The relevant existing change is `db272201` (`[Config] Retire
get_global_server_args, and clear the deprecated flags that have a replacement
(#38375)`). That commit removed the `--mamba-scheduler-strategy`
`DeprecatedAliasStoreAction` registration. It is an ancestor of the recorded
base, so the shared-destination state described in the issue is absent here.

No production patch was added. Although a throwaway parser can still expose
the generic collision if it manually co-registers canonical and deprecated
actions with one destination, no `DeprecatedAliasStoreAction` is registered by
the current `ServerArgs`. Applying the broader proposed guard would therefore
address a hypothetical future registration rather than the issue as it exists
in this checkout.

Evidence is retained in `raw/`:

- `repro-before.txt`: issue-specific real-parser YAML reproduction, successful.
- `parser-boundaries.txt`: canonical YAML/CLI success, retired alias rejection,
  and unsupported-only custom-action rejection.
- `test-config-integration.txt`: existing config integration suite, 4 passed.

GPU execution and a full Qwen/Qwen3.6-27B server launch were not performed.
The reported failure happens before device or model initialization, and the
requested four-GPU NVIDIA/model setup is not represented by the assigned
single gfx950 GPU. No serving, model-accuracy, or distributed claim is made.
