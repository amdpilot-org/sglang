# Independent review of amdpilot-org/sglang#564

Candidate reviewed: `ebba936894d3cd195c0795c40ce6cfca2bf6c8a2`

Base reproduced: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **request changes**. The candidate is a useful partial fix for sufficiently long, supported `no_buffer` prefills, but it does not fully resolve the original issue's contract. The exact three-token scenario from the report still produces a zero-token hit after the candidate.

## Findings

1. The original failure reproduces on the prepared base using the actual source implementation and an assigned AMD Instinct MI355X. Inserting `[10, 20, 30]` with a Mamba state at the leaf and matching the fresh-prefill key `[10, 20]` returns no device indices; matching the full key returns `[0, 1, 2]`.
2. The candidate leaves that minimal failure intact. Its scheduling change creates an interior checkpoint only at a state/page grid boundary. With the tested 64-token grid, an extend of 64 tokens or fewer has no interior boundary, so the candidate still returns a zero-token hit for the original three-token case. The candidate regression suite explicitly expects this zero-hit in `test_n_minus_1_lookup_after_identical_prompt_lands_on_the_safe_boundary`.
3. The candidate does improve a narrower long-prompt case. With a 128-token path and an interior checkpoint at token 64, the candidate returns exactly GPU indices `[0, ..., 63]` for the 127-token lookup. Without that checkpoint, the same lookup returns zero indices. This verifies the new insertion helper on the candidate source, not merely a startup smoke.
4. The added test suite passes, but several tests pin existing tombstone behavior and manually arm an interior checkpoint. Passing those tests does not establish that every fresh prefill gains reuse. The scheduler formula itself deliberately returns no checkpoint below the first grid boundary.
5. Additional counterexamples remain by design: divergence before the deepest retained checkpoint; speculative decoding; ReplaySSM; int8 checkpoints; `extra_buffer`; and unified Mamba+SWA trees. These guards are visible in the candidate source and are also acknowledged in its PR description.
6. The candidate's supplied GPU probe is not proof of the new scheduling/insertion path: it manually inserts nodes at lengths 64 and 94, which exercises pre-existing match semantics. The independent probe instead invokes the candidate's new `_insert_interior_checkpoint` helper and compares the returned GPU indices with an independent `torch.arange` reference.

## Environment and source-path evidence

- Interpreter: `/tmp/amdpilot-repo-j-af0c9c58a42e/venv/bin/python`
- Candidate package import: `/job/repo/python/sglang/__init__.py`
- Candidate cache module import: `/job/repo/python/sglang/srt/mem_cache/mamba_radix_cache.py`
- Torch: `2.11.0+rocm7.2`, HIP `7.2.26015`
- Assigned GPU used: AMD Instinct MI355X
- Runtime AITer module reported: `/tmp/amdpilot-repo-j-af0c9c58a42e/cache/aiter/module_aiter_core.so`
- No C++/HIP/FlyDSL/native source changed in the candidate, and the prepared environment has no task-specific native build entry (`native: null`), so a native rebuild was not applicable.

## Commands and measurements

- Base: `HIP_VISIBLE_DEVICES=0 PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-af0c9c58a42e/venv/bin/python /job/review-evidence-j-af0c9c58a42e/independent_probe.py`
  - Exit 0. Exact minimal n-1 hit: 0; full leaf hit: 3; long n-1 without interior checkpoint: 0.
- Candidate: same command at `ebba936894d3cd195c0795c40ce6cfca2bf6c8a2`
  - Exit 0. Exact minimal n-1 hit remains 0; candidate helper produces a 64-token hit for a 128-token prompt, with indices exactly `[0, ..., 63]` on GPU.
- `PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-af0c9c58a42e/venv/bin/python -m pytest -q test/registered/unit/mem_cache/test_mamba_radix_cache_match.py -vv`
  - Exit 0; 15 passed.
- `PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-af0c9c58a42e/venv/bin/python -m pytest -q test/registered/unit/mem_cache/test_mamba_unittest.py test/registered/unit/mem_cache/test_registry.py`
  - Exit 0; 40 passed.
- `python -m py_compile` on all four changed Python source modules and `git diff --check` against the base
  - Exit 0.

Raw outputs and the independent probe are preserved under `/job/review-evidence-j-af0c9c58a42e/` outside the checkout.

## Limitations

No hybrid-Mamba model weights were prepared, so an end-to-end server run, output-logit comparison, and latency measurement were not possible. The actual cache data path and GPU index tensors were exercised, but the scheduler-to-model state snapshot was not executed with a real model. Therefore the candidate's long-prompt end-to-end correctness remains partially unverified even where the cache helper works.
