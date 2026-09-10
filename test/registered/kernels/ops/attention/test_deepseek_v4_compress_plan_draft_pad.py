"""Kernel-level tests for the DSV4 compress write-plan (`plan_prefill`).

`plan_w` decides which tokens' raw KV get persisted into the compress-state ring
for a *future* compression window to read. A speculative verify batch plans from
the optimistic `seq_len = prefix + num_draft_tokens` but rolls back to
`prefix + accept_len`, so every committed token must stay resident whatever the
accept length -- i.e. the plan must write all of `[prefix, seq_len)`.

`c_plan.cuh` used to cap that pad at 4 (`kMaxMTPDraftTokens`), silently
under-writing the ring for larger draft counts -- no IMA, no NaN, just wrong
compressed state. The pad now comes from the ring itself
(`ring_size - window_size + 2`), covering every draft count the ring can serve.
Tests pin the invariant on both planner paths (CPU host loop and GPU
`plan_compress_prefill_kernel0`) and both compress ratios.
"""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.common import make_paged_context, to_seq_extend
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

C4_RING_SIZE = 16  # get_compress_state_ring_size(4, is_speculative=True)
C128_RING_SIZE = 256  # get_compress_state_ring_size(128, is_speculative=True)
C4_RING_SIZE_NO_SPEC = 8  # get_compress_state_ring_size(4, is_speculative=False)
C128_RING_SIZE_NO_SPEC = 128  # get_compress_state_ring_size(128, is_speculative=False)


def _window_size(compress_ratio: int) -> int:
    """Tokens read by one compression: c4 overlaps two chunks, c128 does not."""
    return compress_ratio * (2 if compress_ratio == 4 else 1)


def _max_draft_tokens(compress_ratio: int, ring_size: int) -> int:
    """Largest draft count this ring serves; mirrors `mtp_pad` in c_plan.cuh."""
    window = _window_size(compress_ratio)
    return ring_size - window + 2 if ring_size > window else 0


def _ragged_ids(plan_w: torch.Tensor) -> torch.Tensor:
    words = plan_w.cpu().view(torch.uint32).view(-1, 2)
    return words[:, 0].to(torch.int64)


def _written_positions(plan_w: torch.Tensor, prefix_len: int) -> set[int]:
    """Decode `plan_w` into the set of positions written, for a bs=1 plan.

    `plan_w` is `[n, 8]` uint8 = (uint32 ragged_id, int32 write_loc). Stage 1
    overwrites `write_loc` with the final state slot, but `ragged_id` survives and
    equals the token's index within the ragged layout, so for a single request
    `position = prefix_len + ragged_id`.
    """
    ragged_ids = _ragged_ids(plan_w)
    valid = ragged_ids != 0xFFFFFFFF
    return {prefix_len + int(r) for r in ragged_ids[valid]}


class TestCompressWritePlanDraftPad(CustomTestCase):
    def _make_plan_positions(
        self,
        *,
        compress_ratio: int,
        ring_size: int,
        prefix_len: int,
        num_draft_tokens: int,
        on_gpu: bool = False,
    ) -> set[int]:
        """Build a bs=1 verify plan and return the positions it writes.

        `on_gpu=True` moves the planner inputs to device, which routes
        `plan_prefill` to `plan_compress_prefill_kernel0` instead of the host loop.
        """
        ctx = make_paged_context(
            bs=1, compress_ratio=compress_ratio, ring_size=ring_size
        )
        seq_lens, extend_lens, num_q = to_seq_extend(
            [(prefix_len + num_draft_tokens, num_draft_tokens)]
        )
        if on_gpu:
            seq_lens = seq_lens.to(ctx.req_to_token.device)
            extend_lens = extend_lens.to(ctx.req_to_token.device)
        plan = ctx.make_prefill_plan(seq_lens, extend_lens, num_q)
        return _written_positions(plan.plan_w, prefix_len)

    def _assert_ring_residency(self, compress_ratio: int, ring_size: int):
        """Every committed token must be written, for each (D, sl mod cr) combo.

        This is the sufficient condition, which is why there is no multi-step replay
        test: if a step writes all of `[prefix, prefix + D)`, then whatever the accept
        length, the tokens the next compression window needs are either from this step
        (written here) or older (written by an earlier step, same invariant by
        induction).
        """
        max_d = _max_draft_tokens(compress_ratio, ring_size)
        # Vary `seq_len % compress_ratio`: that residue decides whether the unpadded rule
        # alone would have sufficed. Four is enough -- with the pad in place it dominates
        # `last_c_pos` for every residue, so the rest repeat one branch. Bases are
        # page-aligned so the swa-page-boundary clause does not mask the pad.
        bases = [512 + off for off in range(min(compress_ratio, 4))]
        draft_counts = sorted(
            {1, 2, 3, 4, 5, max_d - 1, max_d} & set(range(1, max_d + 1))
        )
        for num_draft_tokens in draft_counts:
            for prefix_len in bases:
                with self.subTest(
                    cr=compress_ratio,
                    D=num_draft_tokens,
                    prefix=prefix_len,
                ):
                    written = self._make_plan_positions(
                        compress_ratio=compress_ratio,
                        ring_size=ring_size,
                        prefix_len=prefix_len,
                        num_draft_tokens=num_draft_tokens,
                    )
                    seq_len = prefix_len + num_draft_tokens
                    missing = set(range(prefix_len, seq_len)) - written
                    self.assertEqual(
                        missing,
                        set(),
                        f"plan_w skipped committed positions {sorted(missing)}; "
                        f"a later compression would read stale ring slots",
                    )

    def test_c4_ring_residency(self):
        self._assert_ring_residency(4, C4_RING_SIZE)

    def test_c128_ring_residency(self):
        self._assert_ring_residency(128, C128_RING_SIZE)

    def test_cpu_and_gpu_planner_agree(self):
        """Both planner paths must emit the same write set.

        The residency invariants above are checked on the host-loop plan; this
        pins the GPU `plan_compress_prefill_kernel0` plan to it, so the pad fix
        has to hold on both paths.
        """
        for compress_ratio, ring_size in ((4, C4_RING_SIZE), (128, C128_RING_SIZE)):
            max_d = _max_draft_tokens(compress_ratio, ring_size)
            for num_draft_tokens in (1, 4, max_d):
                for prefix_len in (512, 513, 515):
                    with self.subTest(
                        cr=compress_ratio, D=num_draft_tokens, prefix=prefix_len
                    ):
                        kwargs = dict(
                            compress_ratio=compress_ratio,
                            ring_size=ring_size,
                            prefix_len=prefix_len,
                            num_draft_tokens=num_draft_tokens,
                        )
                        self.assertEqual(
                            self._make_plan_positions(**kwargs, on_gpu=False),
                            self._make_plan_positions(**kwargs, on_gpu=True),
                        )

    @unittest.skipIf(not torch.cuda.is_available(), "Requires a CUDA/HIP graph device")
    def test_c128_ragged_plan_replay_and_stream_order(self):
        """Keep a ragged c128 batch out of the uniform-plan fast path.

        The race in issue 32470 can make a ragged batch look uniform and emit
        out-of-bounds ragged IDs. This bounded metadata-only fixture checks the
        producer invariant repeatedly, under graph replay, and after a side-stream
        handoff; it does not claim full-model or distributed correctness.
        """
        batch_size = 96
        extend_lens = [4] * 72 + [3] * 24
        seq_lens, extend_lens_tensor, num_q_tokens = to_seq_extend(
            [(512 + extend_len, extend_len) for extend_len in extend_lens]
        )
        expected_ids = torch.arange(num_q_tokens, dtype=torch.int64)
        context = make_paged_context(
            bs=batch_size,
            compress_ratio=128,
            ring_size=C128_RING_SIZE,
            num_reqs_capacity=128,
        )
        device = context.req_to_token.device
        seq_lens = seq_lens.to(device)
        extend_lens = extend_lens_tensor.to(device)

        def assert_plan(plan):
            ragged_ids = _ragged_ids(plan.plan_w)
            valid_ids = ragged_ids[ragged_ids != 0xFFFFFFFF]
            self.assertEqual(valid_ids.numel(), num_q_tokens)
            self.assertTrue(
                torch.equal(valid_ids.sort().values, expected_ids),
                f"invalid ragged IDs: max={int(valid_ids.max())}, "
                f"expected_max={num_q_tokens - 1}",
            )

        for _ in range(256):
            assert_plan(
                context.make_prefill_plan(seq_lens, extend_lens, num_q_tokens)
            )

        for _ in range(3):
            context.make_prefill_plan(seq_lens, extend_lens, num_q_tokens)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_plan = context.make_prefill_plan(
                seq_lens, extend_lens, num_q_tokens
            )
        for _ in range(32):
            graph.replay()
        torch.cuda.synchronize()
        assert_plan(captured_plan)

        side_stream = torch.cuda.Stream()
        side_event = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            stream_plan = context.make_prefill_plan(
                seq_lens, extend_lens, num_q_tokens
            )
            side_event.record(side_stream)
        torch.cuda.current_stream().wait_event(side_event)
        assert_plan(stream_plan)

    def test_plain_prefill_write_set(self):
        """A non-speculative ring is exactly one window wide, so the pad is 0 and the
        base write rule stands unchanged for both ratios."""
        for compress_ratio, ring_size in (
            (4, C4_RING_SIZE_NO_SPEC),
            (128, C128_RING_SIZE_NO_SPEC),
        ):
            self.assertEqual(_max_draft_tokens(compress_ratio, ring_size), 0)
            is_overlap = compress_ratio == 4
            for seq_len in (512, 600, 777):
                with self.subTest(cr=compress_ratio, sl=seq_len):
                    ctx = make_paged_context(
                        bs=1, compress_ratio=compress_ratio, ring_size=ring_size
                    )
                    seq_lens, extend_lens, num_q = to_seq_extend([(seq_len, seq_len)])
                    plan = ctx.make_prefill_plan(seq_lens, extend_lens, num_q)
                    written = _written_positions(plan.plan_w, 0)

                    last_c_pos = seq_len // compress_ratio * compress_ratio
                    first_w_pos = last_c_pos - (compress_ratio if is_overlap else 0)
                    sps = ctx.swa_page_size
                    expected = {
                        p
                        for p in range(seq_len)
                        if p >= first_w_pos
                        or (is_overlap and p % sps >= sps - compress_ratio)
                    }
                    self.assertEqual(written, expected)

    def test_over_capacity_under_writes(self):
        """Beyond the ring's capacity the plan silently under-writes.

        The planner cannot tell an over-configured verify batch from an ordinary long
        prefill, so it cannot fail loudly -- hence the startup check in
        `DSV4PoolConfigurator._assert_ring_serves_draft_tokens`.
        """
        for compress_ratio, ring_size in ((4, C4_RING_SIZE), (128, C128_RING_SIZE)):
            max_d = _max_draft_tokens(compress_ratio, ring_size)
            # Far enough over that the `last_c_pos` term cannot cover the gap for any
            # residue of `seq_len % compress_ratio`.
            too_many = max_d + compress_ratio + 1
            prefix_len = 512
            with self.subTest(cr=compress_ratio, D=too_many):
                written = self._make_plan_positions(
                    compress_ratio=compress_ratio,
                    ring_size=ring_size,
                    prefix_len=prefix_len,
                    num_draft_tokens=too_many,
                )
                missing = set(range(prefix_len, prefix_len + too_many)) - written
                self.assertNotEqual(
                    missing,
                    set(),
                    "expected the plan to under-write past the ring capacity; if this "
                    "now covers everything, the startup bound can be relaxed",
                )


if __name__ == "__main__":
    unittest.main()
