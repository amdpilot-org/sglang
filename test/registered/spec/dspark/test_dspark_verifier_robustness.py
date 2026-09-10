"""Edge-case coverage for the DSpark greedy verifier pipeline.

The expected values are computed with explicit CPU loops rather than the
production Torch reference implementations.
"""

import unittest

import torch

from sglang.kernels.ops.speculative.dspark import (
    dspark_accept,
    dspark_verify_window,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")


def _cpu_reference(candidates, target_logits, verify_lens):
    predictions = []
    for row in range(candidates.shape[0]):
        prediction = []
        for pos in range(candidates.shape[1]):
            logits_row = target_logits[row * candidates.shape[1] + pos]
            prediction.append(
                max(range(logits_row.shape[0]), key=lambda col: logits_row[col].item())
            )
        predictions.append(prediction)

    raw_correct_len = []
    for row, prediction in enumerate(predictions):
        count = 0
        for pos in range(candidates.shape[1] - 1):
            if int(candidates[row, pos + 1]) != prediction[pos]:
                break
            count += 1
        raw_correct_len.append(count)

    correct_len = [
        min(raw_count, int(verify_len) - 1)
        for raw_count, verify_len in zip(raw_correct_len, verify_lens)
    ]
    cap_trim_len = [raw - capped for raw, capped in zip(raw_correct_len, correct_len)]
    bonus = [
        prediction[capped] if capped >= 0 else prediction[0]
        for prediction, capped in zip(predictions, correct_len)
    ]
    commit_lens = [capped + 1 for capped in correct_len]
    accepted_ids = [
        [int(candidates[row, pos + 1]) for pos in range(correct_len[row])]
        + ([bonus[row]] if correct_len[row] >= 0 else [])
        for row in range(candidates.shape[0])
    ]
    return predictions, correct_len, cap_trim_len, bonus, commit_lens, accepted_ids


class TestDsparkVerifierRobustness(CustomTestCase):
    def test_two_level_bonus_clamps_padded_rows(self):
        accept_index = torch.arange(12, dtype=torch.int64).view(4, 3)
        predicts = torch.arange(100, 112, dtype=torch.int64)
        correct_len = torch.tensor([-1, 0, 1, 2], dtype=torch.int32)

        bonus = dspark_accept.gather_two_level_bonus_triton(
            accept_index=accept_index.to(DEVICE),
            predicts=predicts.to(DEVICE),
            correct_len=correct_len.to(DEVICE),
        ).cpu()

        expected = [
            int(predicts[accept_index[row, max(int(length), 0)]])
            for row, length in enumerate(correct_len)
        ]
        self.assertEqual(bonus.tolist(), expected)

    def test_zero_all_last_mismatch_and_padded_rows(self):
        vocab_size = 24
        stride = 4
        gamma = stride - 1
        draft_trees = [
            [10, 11, 12, 13],
            [10, 11, 12, 13],
            [10, 11, 12, 99],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        target_predictions = [
            [20, 21, 22, 23],
            [11, 12, 13, 14],
            [11, 12, 13, 14],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        verify_lens = [4, 4, 4, 0, 0]

        candidates_cpu = torch.tensor(draft_trees, dtype=torch.int64)
        logits_cpu = torch.zeros(len(draft_trees) * stride, vocab_size)
        for row, prediction in enumerate(target_predictions):
            for pos, token_id in enumerate(prediction):
                logits_cpu[row * stride + pos, token_id] = 1.0

        candidates = candidates_cpu.to(DEVICE)
        target_logits = logits_cpu.to(DEVICE)
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[stride] * 3,
            device=DEVICE,
            grid=(3 * stride,),
        ).padded_to_bucket(padded_bs=len(draft_trees))
        self.assertEqual(layout.verify_lens.cpu().tolist(), verify_lens)

        (
            _,
            expected_correct_len,
            expected_cap_trim_len,
            expected_bonus,
            expected_commit_lens,
            expected_accepted_ids,
        ) = _cpu_reference(candidates_cpu, logits_cpu, verify_lens)

        correct_len, bonus, cap_trim_len = dspark_accept.AcceptGreedy.triton(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=stride,
            cutoff_verify_lens=layout.verify_lens,
        )
        finalized = dspark_accept.FinalizeAcceptLens.triton(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_len,
            prefix_lens=torch.tensor([7, 11, 13, 17, 19], device=DEVICE),
        )
        out_tokens = dspark_verify_window.BuildOutTokens.triton(
            draft_tokens=candidates[:, 1:].contiguous(),
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=stride,
            gamma=gamma,
        )

        self.assertTrue(
            torch.equal(
                correct_len.cpu(), torch.tensor(expected_correct_len, dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(bonus.cpu(), torch.tensor(expected_bonus, dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                cap_trim_len.cpu(),
                torch.tensor(expected_cap_trim_len, dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                finalized.commit_lens.cpu(),
                torch.tensor(expected_commit_lens, dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                finalized.new_seq_lens.cpu(),
                torch.tensor([8, 15, 16, 17, 19], dtype=torch.int64),
            )
        )

        accepted_ids = out_tokens.cpu()
        for row, expected_ids in enumerate(expected_accepted_ids):
            commit_len = expected_commit_lens[row]
            self.assertEqual(
                accepted_ids[row, :commit_len].tolist(),
                expected_ids,
            )

        req_to_token = (torch.arange(5 * 64, dtype=torch.int64).view(5, 64) + 100).to(
            DEVICE
        )
        commit_layout = dspark_verify_window.BuildCommitInjectLayout.triton(
            req_pool_indices=torch.arange(5, dtype=torch.int64, device=DEVICE),
            req_to_token=req_to_token,
            prefix_lens=torch.tensor(
                [7, 11, 13, 17, 19], dtype=torch.int64, device=DEVICE
            ),
            block_pos_offsets=torch.arange(stride, dtype=torch.int64, device=DEVICE),
            full_to_swa_mapping=torch.arange(1000, dtype=torch.int64, device=DEVICE),
            commit_lens=finalized.commit_lens,
            stride=stride,
        )
        actual_mask = commit_layout.swa_loc.view(5, stride).cpu() != -1
        expected_mask = torch.arange(stride).view(1, -1) < torch.tensor(
            expected_commit_lens
        ).view(-1, 1)
        self.assertTrue(torch.equal(actual_mask, expected_mask))


if __name__ == "__main__":
    unittest.main()
