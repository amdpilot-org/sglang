# Investigation report: Frozen-KV MTP depth/concurrency corruption

Upstream issue: https://github.com/sgl-project/sglang/issues/32666

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1992

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Outcome

`unsupported_architecture`

The report requires an NVIDIA B200 (SM100), CUDA, `trtllm_mha`, ModelOpt NVFP4,
and two Gemma-4 weight sets. The assigned device is an AMD Instinct MI350X
(`gfx950`). Consequently, this investigation cannot honestly reproduce either
the concurrent output corruption or the depth-4 CUDA-graph illegal access.

No source change was made without issue-specific evidence.

## Current related work

The source issue has no maintainer comments. The only PR found by issue number is
https://github.com/sgl-project/sglang/pull/32691, which remains open. It proposes
removing the `batch_size * draft_token_num**2` component from verify-mask sizing
and enlarging a shared mask buffer.

That candidate was not copied. In the checked-out implementation,
`paged_kernel_lens_sum` is passed separately from `paged_kernel_lens`; only the
tensor is then extended by `draft_token_num` per request. Therefore the mask for
the extended lengths is:

```
(paged_kernel_lens_sum + batch_size * draft_token_num) * draft_token_num
= paged_kernel_lens_sum * draft_token_num
  + batch_size * draft_token_num**2
```

Removing the second term under-sizes the mask. Moreover, the source report says
corruption remains with CUDA graphs disabled, so a graph-only dangling-pointer
theory does not cover the primary serving defect.

Merged PR https://github.com/sgl-project/sglang/pull/25545 was also inspected. It
fixed a distinct `topk > 1` draft CUDA-graph metadata allocation and validated
`trtllm_mha` draft attention on B200. The reported issue pins `topk=1` and fails
in target verify, so that merged change is not evidence that issue #32666 is
fixed.

## Validation

The prepared interpreter ran the independent verify-mask boundary suite:

```
/tmp/amdpilot-repo-j-f551878179e6/venv/bin/python -m pytest -q \
  test/registered/unit/layers/attention/test_verify_mask.py
```

This passed, but is only CPU-side formula coverage. It does not qualify the
Frozen-KV MTP serving path. Raw test and hardware output are retained beside
this report.

## Required follow-up

Faithful resolution still requires the exact target/draft architecture and an
SM100-class CUDA device. The original no-grammar concurrent probe must be run at
depths 3, 4, and 5 with verified server-side overlap, preserving output and
server logs. Any fix must then demonstrate failing-before/passing-after on that
path, including eager execution because the reported corruption survives with
CUDA graphs disabled.
