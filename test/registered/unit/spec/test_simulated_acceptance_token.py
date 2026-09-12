import unittest

import torch

from sglang.srt.speculative import spec_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ByteFallbackTokenizer:
    all_special_ids = []

    def __len__(self):
        return 128

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"a": [100], "A": [64], "0": [15], ".": [13]}[text]

    def decode(self, token_ids):
        pieces = {13: ".", 15: "0", 32: "A", 64: "a", 100: "\ufffd"}
        return "".join(pieces.get(token_id, "") for token_id in token_ids)


class TestSimulatedAcceptanceToken(unittest.TestCase):
    def test_resolves_token_that_decodes_to_complete_text(self):
        token_id = spec_utils.resolve_simulated_accept_token_id(
            _ByteFallbackTokenizer(), vocab_size=128
        )
        self.assertEqual(token_id, 64)

    def test_fixed_mode_fills_predict_with_resolved_token(self):
        accept_index = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32)
        predict = torch.full((4,), 100, dtype=torch.int32)
        num_correct_drafts = torch.empty((1,), dtype=torch.int32)

        spec_utils.generate_simulated_accept_index(
            accept_index=accept_index,
            predict=predict,
            num_correct_drafts=num_correct_drafts,
            candidates=torch.zeros((1, 4), dtype=torch.int64),
            target_predict=torch.zeros((1, 4), dtype=torch.int64),
            bs=1,
            spec_steps=3,
            simulate_acc_len=3,
            simulate_acc_method="match-expected",
            simulate_acc_token_mode="fixed",
            simulate_acc_token_id=64,
        )

        self.assertEqual(num_correct_drafts.tolist(), [2])
        self.assertEqual(predict.tolist(), [64, 64, 64, 64])

    def test_fixed_mode_requires_token_id(self):
        with self.assertRaisesRegex(ValueError, "simulate_acc_token_id is required"):
            spec_utils.generate_simulated_accept_index(
                accept_index=torch.tensor([[0, -1]], dtype=torch.int32),
                predict=torch.zeros((2,), dtype=torch.int32),
                num_correct_drafts=torch.empty((1,), dtype=torch.int32),
                candidates=torch.zeros((1, 2), dtype=torch.int64),
                target_predict=torch.zeros((1, 2), dtype=torch.int64),
                bs=1,
                spec_steps=1,
                simulate_acc_len=1,
                simulate_acc_method="match-expected",
                simulate_acc_token_mode="fixed",
            )


if __name__ == "__main__":
    unittest.main()
