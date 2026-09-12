# Independent review of candidate PR 3450

- Upstream issue: https://github.com/sgl-project/sglang/issues/37553
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3325
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3455
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Candidate reviewed: `32b01f36846b86eba8b04bc1a40806b0fd110d69`
- Recommendation: **accept**

## Findings

The prepared checkout initially matched the recorded base exactly. On that base, a real-GPU 2 -> 7 -> 1 batch sequence reproduced the original contract violation in both implementations. EAGLE retained and returned 462 elements when 66 were required after shrink; DFlash retained and returned 168 when 24 were required. Zero-length and empty-batch calls likewise returned the prior oversized tensors.

At the exact candidate commit, the submitted regression suite passed (5 tests). An independent sequence covering batch growth, shrink, zero sequence length, and empty batch also passed for both EAGLE and DFlash. Every returned mask had exactly the formula-derived length, the original nontrivial `[False, True, False]` prefix was preserved, padding was `True`, and `self.custom_mask` retained its original identity and three-element size.

The candidate is a full fix for the original issue's two stated defects: retained padding no longer grows with peak batch size, and callers no longer receive stale oversized padding after shrink. It is not merely test hardening. The local `torch.cat` still creates a temporary exact-size padded result when padding is required, but it is not stored on the reusable input; this is bounded per-call temporary allocation, not the reported retained growth. Slicing covers the already-large case without allocation.

Source inspection confirmed that the changed modules imported from `/job/repo/python/sglang/srt/speculative/`, not from an installed wheel. The patch changes Python and tests only; it contains no C++/FlyDSL/native source, so a native rebuild was not applicable. The real `create_flashinfer_kv_indices_triton` GPU kernel was executed by the focused tests.

## Commands and measurements

```text
/tmp/amdpilot-repo-j-378c4f34c991/venv/bin/python /job/review-evidence-j-378c4f34c991/independent_mask_contract.py
```

Exit 0 on the base. Measured EAGLE required/returned after shrink as 66/462 and DFlash as 24/168; retained sizes were 462 and 168.

```text
/tmp/amdpilot-repo-j-378c4f34c991/venv/bin/python -m pytest -q test/registered/unit/spec/test_custom_mask_padding.py
```

Exit 0 at the candidate: 5 passed. This covered both implementations' growth and shrink behavior, retained identity, prefix values, shrink storage view, and DFlash `None`.

```text
/tmp/amdpilot-repo-j-378c4f34c991/venv/bin/python /job/review-evidence-j-378c4f34c991/independent_mask_contract.py --expect-fixed
```

Exit 0 at the candidate. For each implementation, tested `(batch, seq_len)` values `(2,5), (7,5), (1,5), (4,0), (0,0)` and asserted exact formula-derived output length, retained tensor identity/size, original prefix preservation, and true-only padding.

```text
/tmp/amdpilot-repo-j-378c4f34c991/venv/bin/python -m py_compile python/sglang/srt/speculative/eagle_info.py python/sglang/srt/speculative/dflash_info.py test/registered/unit/spec/test_custom_mask_padding.py
git diff --check 358c163250ad3b1f62939b01ce1314a0a31a0365 32b01f36846b86eba8b04bc1a40806b0fd110d69
```

Both exited 0.

## Environment and limitations

Tests used the one visible AMD Instinct MI350X (`gfx950`) GPU with Torch `2.11.0+rocm7.2`, HIP `7.2.26015`, and the prepared interpreter. NUMA balancing was enabled and AIter emitted its standard warning; it did not prevent execution. No EAGLE or DFlash model weights were supplied, so this review does not claim full HTTP/model-serving, semantic generation, long-duration allocator fragmentation, or distributed/multi-node validation. The deterministic tiny Llama fixture would not qualify EAGLE/DFlash architecture or semantics and was therefore not substituted for the focused implementation test.

Raw command output and the independent harness are preserved outside revision switching under `/job/review-evidence-j-378c4f34c991/`.
