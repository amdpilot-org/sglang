from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.utils import get_dsa_seed_metadata_dim
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _config(*, index_topk, index_share_for_mtp_iteration, index_kpool=1):
    return SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=index_topk,
        index_share_for_mtp_iteration=index_share_for_mtp_iteration,
        index_kpool=index_kpool,
    )


@pytest.mark.parametrize("share", [False, True])
def test_seed_metadata_dim_is_zero_when_dsa_is_disabled(share):
    """A stale index-sharing flag is moot after index_topk disables DSA."""
    config = _config(
        index_topk=None,
        index_share_for_mtp_iteration=share,
    )

    assert get_dsa_seed_metadata_dim(config) == 0


def test_seed_metadata_dim_is_zero_when_index_sharing_is_disabled():
    config = _config(
        index_topk=2048,
        index_share_for_mtp_iteration=False,
    )

    assert get_dsa_seed_metadata_dim(config) == 0


@pytest.mark.parametrize(
    ("index_kpool", "expected_width"),
    [(1, 2048), (128, 2175)],
)
def test_seed_metadata_dim_preserves_enabled_dsa_width(index_kpool, expected_width):
    config = _config(
        index_topk=2048,
        index_share_for_mtp_iteration=True,
        index_kpool=index_kpool,
    )

    assert get_dsa_seed_metadata_dim(config) == expected_width
