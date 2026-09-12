Issue inspection on 2026-09-12 found the upstream issue still open and current at
https://github.com/sgl-project/sglang/issues/35884.

The related upstream candidate is open PR 35896:
https://github.com/sgl-project/sglang/pull/35896

That candidate identifies two coupled requirements also confirmed against the
prepared base: call `TokenizerManager.abort_request(rid)` before removing the
tracked state, and prevent the scheduler's RID-prefix health filter from
discarding the resulting `AbortReq`. The prepared base contained neither part.
No upstream participants were tagged or notified.
