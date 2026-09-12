# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reject DFLASH before an unsupported PD launch reaches the scheduler."""

import unittest

from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _run_handler(*, mode: str, algorithm: str | None) -> None:
    args = ServerArgs(
        model_path="missing-model-must-not-be-loaded",
        disaggregation_mode=mode,
        speculative_algorithm=algorithm,
    )
    handle_pd_disaggregation(args)


class TestDFlashPDDisaggregationGate(unittest.TestCase):
    def test_dflash_is_rejected_for_both_pd_roles(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(
                    ValueError, "DFLASH.*not supported.*draft KV.*auxiliary hidden states"
                ):
                    _run_handler(mode=mode, algorithm="DFLASH")

    def test_dflash_without_disaggregation_is_unchanged(self):
        _run_handler(mode="null", algorithm="DFLASH")

    def test_supported_pd_speculative_algorithms_are_unchanged(self):
        for algorithm in ("EAGLE", "DSPARK"):
            for mode in ("prefill", "decode"):
                with self.subTest(mode=mode, algorithm=algorithm):
                    _run_handler(mode=mode, algorithm=algorithm)

    def test_guard_is_case_insensitive_before_algorithm_normalization(self):
        with self.assertRaisesRegex(ValueError, "DFLASH.*not supported"):
            _run_handler(mode="decode", algorithm="dflash")


if __name__ == "__main__":
    unittest.main()
