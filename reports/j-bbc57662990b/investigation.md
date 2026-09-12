# Investigation evidence

- Upstream issue: https://github.com/sgl-project/sglang/issues/36500
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1164
- Related open upstream candidate inspected: https://github.com/sgl-project/sglang/pull/36517
- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

At the prepared base, `ExternalCorpusManager.remove()` unconditionally called the
worker removal method even when `_pending_load` named the same corpus. The
before-fix regression held `add_external_corpus()` inside the background worker,
then called the real manager's `remove()` method. It returned `success=True`.

The correction implements the issue's explicit-rejection option. It rejects
only the corpus ID represented by `_pending_load`. Deterministic boundary tests
show that a different already-loaded corpus remains removable while the load is
pending, and that the newly loaded corpus is removable after
`check_pending_load()` commits it.

The upstream candidate uses the same policy but its sleep-based tests were not
copied. This change uses `threading.Event` synchronization to establish that the
load is running and to avoid timing-dependent assertions.
