import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _RecordingConfidenceHead:
    with_markov = True

    def __init__(self):
        self.hidden_shape = None
        self.markov_shape = None

    def __call__(self, hidden_states, markov_embeddings):
        self.hidden_shape = tuple(hidden_states.shape)
        self.markov_shape = tuple(markov_embeddings.shape)
        return hidden_states[..., 0]

    @staticmethod
    def apply_sts(confidence_raw):
        return confidence_raw


class _RecordingMarkovHead:
    def __init__(self):
        self.prev_seq = None

    def get_prev_embeddings(self, prev_seq):
        self.prev_seq = prev_seq
        return prev_seq.unsqueeze(-1).float()


class TestDeepseekV4DSparkConfidenceRuntimeGamma(CustomTestCase):
    def _compute_confidence(self, *, checkpoint_gamma, runtime_gamma, device="cpu"):
        confidence_head = _RecordingConfidenceHead()
        markov_head = _RecordingMarkovHead()
        model = SimpleNamespace(
            confidence_head=confidence_head,
            gamma=checkpoint_gamma,
            markov_head=markov_head,
        )
        bs, hidden_size = 2, 4
        anchor_tokens = torch.tensor([10, 20], device=device)
        sampled_tokens = torch.arange(
            bs * runtime_gamma, device=device, dtype=torch.int64
        ).view(bs, runtime_gamma)
        x_post_hc = torch.arange(
            bs * runtime_gamma * hidden_size, device=device, dtype=torch.float32
        ).view(bs * runtime_gamma, hidden_size)

        confidence = DeepseekV4ForCausalLMDSpark.compute_confidence(
            model,
            anchor_tokens=anchor_tokens,
            sampled_tokens=sampled_tokens,
            x_post_hc=x_post_hc,
        )
        return confidence, confidence_head, markov_head, sampled_tokens

    def test_uses_larger_runtime_window_instead_of_checkpoint_gamma(self):
        confidence, head, markov, sampled = self._compute_confidence(
            checkpoint_gamma=5, runtime_gamma=7
        )

        self.assertEqual(tuple(confidence.shape), (2, 7))
        self.assertEqual(head.hidden_shape, (2, 7, 4))
        self.assertEqual(head.markov_shape, (2, 7, 1))
        self.assertTrue(torch.equal(markov.prev_seq[:, 0], torch.tensor([10, 20])))
        self.assertTrue(torch.equal(markov.prev_seq[:, 1:], sampled[:, :6]))

    def test_uses_smaller_runtime_window_instead_of_checkpoint_gamma(self):
        confidence, head, markov, sampled = self._compute_confidence(
            checkpoint_gamma=5, runtime_gamma=3
        )

        self.assertEqual(tuple(confidence.shape), (2, 3))
        self.assertEqual(head.hidden_shape, (2, 3, 4))
        self.assertEqual(head.markov_shape, (2, 3, 1))
        self.assertTrue(torch.equal(markov.prev_seq[:, 1:], sampled[:, :2]))

    def test_checkpoint_native_window_remains_unchanged(self):
        confidence, head, markov, _ = self._compute_confidence(
            checkpoint_gamma=5, runtime_gamma=5
        )

        self.assertEqual(tuple(confidence.shape), (2, 5))
        self.assertEqual(head.hidden_shape, (2, 5, 4))
        self.assertEqual(head.markov_shape, (2, 5, 1))


if __name__ == "__main__":
    unittest.main()
