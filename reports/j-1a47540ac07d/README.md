# Independent review of PR 2445

Reviewed `https://github.com/amdpilot-org/sglang/pull/2445` at exact commit
`c276f578f02392cda2d4766ca4a846f5ab33e3f2` against upstream issue
`https://github.com/sgl-project/sglang/issues/30015` and candidate mirror issue
`https://github.com/amdpilot-org/sglang/issues/2394`.

Recommendation: **accept**. The candidate fully resolves the original issue's
reported contract for PP proxy dictionaries: scheduler traffic only applies the
slice/all-gather optimization to the established TP-replicated keys
`hidden_states` and `residual`; model-specific proxy keys are sent whole. Both
the ordinary and paired send/receive implementations are wired consistently.

## Evidence

- On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the four-process
  Gloo reproducer failed as expected. Rank 2 expected eight `1.0` values and
  rank 3 expected eight `2.0` values, but both received
  `[1,1,1,1,2,2,2,2]`. The replicated control tensor remained correct.
- On candidate `c276f578f02392cda2d4766ca4a846f5ab33e3f2`, the same actual
  `GroupCoordinator` path passed: each receiver got its matching sender's full
  sharded tensor, while the replicated control still reconstructed correctly.
- The candidate's complete `test_parallel_state.py` passed 7/7.
- An independent four-process adversarial exercise of
  `send_recv_tensor_dict` passed on all ranks. It simultaneously exchanged a
  rank-distinct custom shard, replicated `hidden_states`, and a non-tensor tag
  in both PP directions.
- The candidate's GPU selection check ran against source imported from
  `/job/repo/python` on the assigned AMD Instinct MI355X (`gfx950`). It observed
  4 elements sent for replicated `hidden_states` and all 8 elements for the
  custom sharded tensor.
- Import inspection resolved `sglang`, `parallel_state.py`, and
  `scheduler_pp_mixin.py` from `/job/repo/python`, not an installed sglang
  wheel. The exact candidate signature included `all_gather_keys`.
- No C++/HIP/native source changed, and the prepared environment records no
  separate native component for this task, so a native rebuild was not
  applicable.

## Review notes and limitations

The new committed unit test is a mocked send-side boundary test rather than a
registered multi-process end-to-end regression. The candidate nevertheless
includes a four-process reproducer in its report, and this review independently
ran both one-way and paired four-process paths. The API requires callers of the
low-level primitive to provide matching allowlists on sender and receiver;
the scheduler does so at every call site. This is less misuse-resistant than
the metadata-carried design in open upstream PR 30095, but it does not leave a
counterexample in the scheduler contract reported by the issue.

Only one physical GPU was assigned. The four-rank topology therefore used Gloo
and CPU tensors; this review does not claim a four-GPU RCCL/NCCL reproduction,
multi-node coverage, RWKV-7 semantic accuracy, or greedy-decoding validation.
The single-GPU run validates GPU tensor selection only. No RWKV weights or
four-GPU NVIDIA L4 environment were available. The tiny Llama fixture cannot
qualify an RWKV-specific sharded proxy tensor and was not used.

`git diff --check` over the candidate returned exit 2 solely because committed
raw log artifacts contain trailing spaces/newline-at-EOF diagnostics; no source
or test file whitespace error was reported. This is non-functional review
hygiene and does not change the correctness recommendation.
