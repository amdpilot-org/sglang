# AITER workspace planning investigation

## Conclusion

Current `main` does **not** account for the AITER attention workspace while deriving `max_total_num_tokens`. The planner spends the controlled budget on KV cache first, then `AiterAttnBackend.__init__` allocates a separate `workspace_buffer`. In the synthetic 64 MiB case, predicted and allocated workspace bytes were both 17,448,304,640, leaving -16.25 GiB of headroom before KV cache allocation.

Upstream PR 20890 reduces the workspace by capping `max_num_partitions` with `max_total_num_tokens`, but it does not reserve workspace bytes in the planner. Upstream PR 18263 does reserve the workspace by reducing the KV byte budget before cache allocation; in the 20 GiB case it reduced planning bytes to 4,026,531,840 and preserved exactly 0 bytes of headroom.

Upstream issue 18262 remains open. PR 18263 is open; PR 20890 is closed.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one `AMD Instinct MI300X`, `gfx942`, 206,141,652,992 bytes
- Python: `/opt/venv/bin/python` -> `/usr/bin/python3.10`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Installed SGLang source: `/sgl-workspace/sglang/python`
- AITER native module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- Working clone: `/job/sglang`

## Tested revisions

- Current mirror `main`: `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`
- Upstream PR 20890 head: `236f1244f2a2b8b59f5641bae493bb5db33603af`
- Upstream PR 18263 head: `aa321d382c56a78ce6ee6cb2a1961dbb675dc56e`

## Method

The harness uses a synthetic MHA configuration with AITER enabled, a controlled byte budget, and the real `AiterAttnBackend.__init__` allocation path. It records:

- `max_total_num_tokens`
- `max_running_requests`
- `max_num_partitions`
- predicted workspace bytes
- allocated workspace bytes
- backend allocator delta
- predicted KV cache bytes
- headroom before cache allocation

No model weights were downloaded and no full server was launched.

## Raw results

| Revision | Budget | `max_total_num_tokens` | `max_running_requests` | Partitions | Predicted workspace | Allocated workspace | KV bytes | Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current `main` | 64 MiB | 8,192 | 2,048 | 512 | 17,448,304,640 | 17,448,304,640 | 67,108,864 | -17,448,304,640 |
| PR 20890 | 64 MiB | 8,192 | 2,048 | 32 | 1,090,519,040 | 1,090,519,040 | 67,108,864 | -1,090,519,040 |
| current `main` | 20 GiB | 2,621,440 | 4,096 | 512 | 34,896,609,280 | 34,896,609,280 | 21,474,836,480 | -34,896,609,280 |
| PR 18263 | 20 GiB | 491,520 | 2,048 | 512 | 17,448,304,640 | 17,448,304,640 | 4,026,531,840 | 0 |

## Failures and limitations

- PR 20890 was first run with a 1 GiB budget. That made `max_total_num_tokens` equal `context_len`, so its cap was inactive; the run was repeated at 64 MiB.
- PR 18263 was first exercised through `config_from_budget` directly, which bypassed `_profile_available_bytes` and therefore did not test its solver. The harness was corrected to call the planning path.
- PR 18263 was then run at 64 MiB, which is smaller than its 16.25 GiB workspace requirement; `MemoryPoolConfig.__post_init__` raised `RuntimeError: Not enough memory`. The successful comparison used a 20 GiB budget.
- No numerical accuracy gates were changed or run. This investigation is limited to memory planning and allocation.
- Upstream issue 18262 and PRs 18263 and 20890 were read only; no upstream issue, PR, or comment was modified.

## Reproduction

See `commands.txt` for the exact bounded command sequence and `measure_aiter_workspace.py` / `measure_aiter_workspace_legacy.py` for the harnesses.
