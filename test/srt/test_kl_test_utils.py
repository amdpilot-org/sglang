import pytest

from sglang.test.kl_test_utils import compare_kl_divergence

_THRESHOLDS = {"model": {"kl_div": 1e-9}}


def test_exact_comparison_accepts_identical_logprobs():
    compare_kl_divergence(
        [[-1.0, -2.0]],
        [[-1.0, -2.0]],
        _THRESHOLDS,
        "model",
        "identical",
        require_exact=True,
    )


def test_exact_comparison_rejects_nonzero_kl_below_threshold():
    with pytest.raises(AssertionError, match="non-bit-exact logprobs"):
        compare_kl_divergence(
            [[0.0]],
            [[-1e-5]],
            _THRESHOLDS,
            "model",
            "adversarial",
            require_exact=True,
        )


def test_exact_comparison_rejects_difference_rounded_out_of_kl():
    with pytest.raises(AssertionError, match="non-bit-exact logprobs"):
        compare_kl_divergence(
            [[0.0]],
            [[-1e-12]],
            _THRESHOLDS,
            "model",
            "rounded",
            require_exact=True,
        )
