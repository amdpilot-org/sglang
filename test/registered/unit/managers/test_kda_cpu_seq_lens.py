import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.managers.overlap_utils import decide_needs_cpu_seq_lens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKDACPUSeqLensContract(CustomTestCase):
    def _decide(self, backends, *, algorithm="DSPARK", tbo=False):
        exec_config = SimpleNamespace(
            overlap=SimpleNamespace(enable_two_batch_overlap=tbo)
        )
        spec_config = SimpleNamespace(speculative_algorithm=algorithm)
        with (
            patch(
                "sglang.srt.managers.overlap_utils.get_exec",
                return_value=exec_config,
            ),
            patch(
                "sglang.srt.managers.overlap_utils.get_spec",
                return_value=spec_config,
            ),
        ):
            return decide_needs_cpu_seq_lens(backends)

    def test_kda_dspark_does_not_request_blocking_host_mirror(self):
        # Regression for https://github.com/sgl-project/sglang/issues/33846:
        # inheriting the base-class True here forced a per-step seq_lens D2H
        # synchronization in the c=1 KDA speculative path.
        self.assertFalse(KDAAttnBackend.needs_cpu_seq_lens)
        self.assertFalse(self._decide([KDAAttnBackend]))

    def test_backend_that_needs_host_lengths_keeps_legacy_path(self):
        host_consumer = SimpleNamespace(needs_cpu_seq_lens=True)
        self.assertTrue(self._decide([KDAAttnBackend, host_consumer]))

    def test_unset_backend_slots_do_not_force_host_mirror(self):
        self.assertFalse(self._decide([None, KDAAttnBackend, None]))

    def test_external_host_consumers_still_force_host_mirror(self):
        self.assertTrue(self._decide([KDAAttnBackend], tbo=True))
        self.assertTrue(self._decide([KDAAttnBackend], algorithm="NGRAM"))


if __name__ == "__main__":
    unittest.main()
