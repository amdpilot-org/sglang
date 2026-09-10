# MI300X speculative tree-mask verification

## Result

The existing upstream candidate in `sgl-project/sglang` pull 36201, commit
`f5e6e0bcf5129fe439a4d9c5361a09f26721eb1f`, passes the affected real-GPU
checks on one MI300X (`gfx942`). It passes all 18 upstream case/mode
combinations and two additional synthetic trees with 10 and 20 draft tokens.
No production change is delivered here because the working fix and regression
test already exist upstream; duplicating them would not be useful.

This investigation covers `build_tree_kernel_efficient` mask construction only.
It does not test or change greedy or stochastic verifier acceptance decisions.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942`, driver `6.19.14.31400000`
- Python: `/opt/venv/bin/python`, Python 3.10.12
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Installed Python source: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Installed native Python module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/speculative.py`
- Installed native extension: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Mirror base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`

## Installed-source baseline

The first GPU execution used the preinstalled source and native module, before
cloning or editing:

```bash
start=$(date +%s.%N)
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/spec/utils/test_build_eagle_tree.py -s
status=$?
end=$(date +%s.%N)
printf 'status=%s elapsed_seconds=%.6f\n' "$status" \
  "$(awk -v s="$start" -v e="$end" 'BEGIN {print e-s}')"
```

Result: `2 passed`, process status `0`, pytest-reported `31.82 s`, and
`35.0006 s` wall time measured with GNU `date` around the whole pytest process.
The installed source commit was
`8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.

The first timing attempt used `/usr/bin/time`, which is not installed:

```text
/bin/bash: line 2: /usr/bin/time: No such file or directory
```

That control error is recorded in `/job/baseline-first.json`. The installed
native tree-mask path itself was supported; no unsupported-native-path result is
claimed.

## Upstream context

`sgl-project/sglang` issue 30344 is the DSpark roadmap. Its current
description and five comments discuss DSpark robustness, CUDA-graph tracking,
and related issues; they do not mention this tree-mask construction path. The
directly relevant change is `sgl-project/sglang` pull 36201, which bounds the
ancestor walk, keeps `selected_index` lookups inside one request row, and adds
an independent CPU oracle. No upstream issue, pull request, or comment was
posted or modified.

## Candidate verification

The candidate patch was applied temporarily to the mirror base commit. A
focused extension was built from only `eagle_utils.cu` and a local binding, with
all build output under `/tmp/sglang-cache-j-c7fec84e5779`:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python \
  /opt/venv/bin/python reports/j-c7fec84e5779/build_candidate.py
```

The resulting native extension was
`/tmp/sglang-cache-j-c7fec84e5779/extension/eagle_tree_candidate.so`.
The verification command was:

```bash
cd /job/sglang/reports/j-c7fec84e5779
PYTHONPATH=/job/sglang/python:/job/sglang/reports/j-c7fec84e5779 \
  /opt/venv/bin/python verify_candidate.py
```

The candidate passed:

- 18 upstream case/mode combinations across `FULL_MASK`, `QLEN_ONLY`, and
  `QLEN_ONLY_BITPACKING`
- exact `torch.equal` comparison of positions, tree mask, retrieve index,
  retrieve-next-token, and retrieve-next-sibling against the independent CPU
  traversal
- malformed cases including missing direct parent, missing ancestor, cyclic
  ancestor, maximum valid depth, and cross-row lookup

The measured candidate verification elapsed time was `10.019177546724677 s`.
Raw values are in `results.json`.

## Boundary and padding checks

Two additional valid synthetic trees used batch size 2, `topk=2`, sequence
lengths 7 and 11, and 10 or 20 draft tokens. Each tree has a branch and a long
ancestor chain. The independent CPU reference performed the same row-local
lookup and depth-bounded root walk.

For every mode, the GPU result exactly matched the CPU reference for all five
outputs. Additional gates were:

- `draft_token_num=10`: two-byte packed items; bits 10 through 15 were zero
- `draft_token_num=20`: four-byte packed items; bits 20 through 31 were zero
- ancestor bits spanned the 8-bit byte boundaries at columns 7/8 and 15/16
- every mask row was causally bounded by its token column
- every position stayed within `verified_seq_len + depth`
- the retrieval tree contained a real branch sibling

## Installed-native control

The same valid 10- and 20-token boundary trees were run unchanged through the
installed operator:

```text
torch.ops.sgl_kernel.build_tree_kernel_efficient
```

Both passed all mask, ancestry, padding, causal-bound, and position-bound checks
in `0.13319541327655315 s`. The malformed cyclic case was intentionally not run
on this installed unbounded path. This control shows that valid-tree mask
construction already works; the candidate's relevant improvement is safe
handling of malformed trees and row-local lookup. Raw values are in
`installed-control.json`.

## Reproduction

The report harness intentionally does not duplicate pull 36201. To reproduce
the candidate run:

1. Start from mirror base commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.
2. Apply `sgl-project/sglang` pull 36201 at commit
   `f5e6e0bcf5129fe439a4d9c5361a09f26721eb1f`.
3. Run `reports/j-c7fec84e5779/build_candidate.py`.
4. Run `reports/j-c7fec84e5779/verify_candidate.py`.

The build script copies the source into the job-private cache before invoking
Torch's hipify machinery, so generated files do not pollute the checkout.

## Not done

- No full `sgl-kernel` wheel was rebuilt; only the affected native source was built.
- No ROCm memory-sanitizer run was performed.
- No throughput benchmark was run; this was a bounded correctness investigation.
- No upstream post or review was made.
