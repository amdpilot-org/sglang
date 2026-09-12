import unittest

import torch

from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkVerifyPlanner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ConfidenceHead:
    with_markov = True

    def __init__(self):
        self.input_shapes = None

    def __call__(self, hidden, markov):
        self.input_shapes = (hidden.shape, markov.shape)
        return torch.zeros(hidden.shape[:2])

    @staticmethod
    def apply_sts(confidence):
        return confidence


class _MarkovHead:
    def __init__(self):
        self.tokens = None

    def get_prev_embeddings(self, tokens):
        self.tokens = tokens
        return torch.zeros(*tokens.shape, 2)


def _model(*, checkpoint_gamma=5):
    model = object.__new__(DeepseekV4ForCausalLMDSpark)
    model.gamma = checkpoint_gamma
    model.confidence_head = _ConfidenceHead()
    model.markov_head = _MarkovHead()
    return model


def _planner(model, *, runtime_gamma):
    planner = object.__new__(DSparkVerifyPlanner)
    planner.draft_model = model
    planner.gamma = runtime_gamma
    planner._confidence_head = model.confidence_head
    return planner


class TestDSparkRuntimeGamma(CustomTestCase):
    def test_planner_override_reaches_confidence_head(self):
        model = _model(checkpoint_gamma=5)
        planner = _planner(model, runtime_gamma=7)
        anchor = torch.tensor([10, 20])
        sampled = torch.arange(14).view(2, 7)

        confidence = planner.compute_confidence_tensor(
            draft_hidden=None,
            anchor_tokens=anchor,
            draft_tokens=sampled,
            confidence_tap=torch.zeros(14, 4),
        )

        self.assertEqual(confidence.shape, (2, 7))
        self.assertEqual(model.confidence_head.input_shapes[0], (2, 7, 4))
        torch.testing.assert_close(
            model.markov_head.tokens,
            torch.cat([anchor[:, None], sampled[:, :6]], dim=1),
        )

    def test_model_hook_defaults_to_checkpoint_gamma(self):
        model = _model(checkpoint_gamma=5)
        anchor = torch.tensor([10, 20])
        sampled = torch.arange(10).view(2, 5)

        confidence = model.compute_confidence(
            anchor_tokens=anchor,
            sampled_tokens=sampled,
            x_post_hc=torch.zeros(10, 4),
        )

        self.assertEqual(confidence.shape, (2, 5))
        self.assertEqual(model.confidence_head.input_shapes[0], (2, 5, 4))

    def test_generic_confidence_path_keeps_runtime_gamma(self):
        class GenericModel:
            confidence_head = _ConfidenceHead()
            markov_head = _MarkovHead()

        model = GenericModel()
        planner = _planner(model, runtime_gamma=3)
        anchor = torch.tensor([10, 20])
        sampled = torch.arange(6).view(2, 3)

        confidence = planner.compute_confidence_tensor(
            draft_hidden=torch.zeros(2, 3, 4),
            anchor_tokens=anchor,
            draft_tokens=sampled,
        )

        self.assertEqual(confidence.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
