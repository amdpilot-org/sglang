# Mamba short-prefix ownership review

Upstream issue: https://github.com/sgl-project/sglang/issues/22935

Mirror issue: https://github.com/amdpilot-org/sglang/issues/623

Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/564 at
`ebba936894d3cd195c0795c40ce6cfca2bf6c8a2`.

## Decision

Reject the candidate as a fix for the original exact `[A,B,C]` then n-1
`[A,B]` case. It adds a useful, real interior checkpoint for sufficiently
long prefills, but its own regression asserts that the three-token lookup
still returns zero. It therefore does not establish the claim in the source
issue for short prefixes.

The current zero result is required by recurrent-state ownership. The only
stored state after the first request belongs to depth 3. `_split_node()` must
leave that state on the `[C]` descendant; attaching it to the new `[A,B]`
parent would use the state after C as the state before C and corrupt the next
forward pass.

The scheduler treats every returned device index as already computed. The
current hybrid execution path has no operation that reuses attention KV for a
prefix while independently replaying those same tokens through recurrent
layers. Extend-time state/conv snapshots are exposed on the checkpoint grid,
not at arbitrary token depths. Thus an exact depth-2 state cannot be recovered
from the depth-3 state under this architecture.

## Minimal correct semantics

- Return KV only through the deepest node that owns a state at exactly that
  depth.
- Keep the descendant state on the descendant when splitting an edge.
- If a real interior state was captured during the original forward, insert it
  as a separately owned checkpoint; this preserves supported long-prefix
  reuse.
- If divergence occurs before the first checkpoint, replay from the previous
  owned checkpoint (the root in the tested case), which means a zero-token
  cache hit.

Making the exact three-token case nonzero requires an architectural addition:
either capture an exact penultimate recurrent and convolution state during the
first prefill (including arbitrary-position backend support and radix/page
ownership), or add split replay that can recompute recurrent state while
retaining compatible attention KV. Neither exists in the reviewed base or
candidate.

## Evidence

- Base focused/surrounding tests: `pytest.txt` (20 passed).
- Candidate exact-commit overlay: `candidate-validation.txt` (19 passed); both
  candidate and independent tests measured zero for the original short case.
- Assigned-GPU actual-cache probe: `gpu_cache_probe.txt`; MI350X tensors
  measured short=0, full leaf=3, real checkpoint=64, and the returned indices
  exactly matched independent GPU `arange` references.

No model weights were prepared, so end-to-end hybrid-model logits were not
measured. No native code changed and no native rebuild was applicable.
