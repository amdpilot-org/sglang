# Independent review of PR 1547

Reviewed exact candidate `ccabba009ecd391d7f97d50a7c1ca52b7490ac95` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.

Recommendation: **request changes**. The added unit regression is useful and passes on the actual GPU, but the candidate is test-only hardening, not a production fix. The recorded base already contains the runtime guard. The candidate does not run an end-to-end NEXTN TARGET_VERIFY workload and therefore cannot independently establish complete resolution of the original hybrid-mamba serving report.

The candidate evidence also needs factual corrections: its PR body references mirror issue 1484 instead of 1585, and its result names the device MI350X although the interpreter reports AMD Instinct MI355X (`gfx950`). Its `outcome: fixed` should be narrowed to the test-only contribution and its limitations.

No native files changed, so no rebuild was applicable. Source imports were confirmed from `/job/repo/python/sglang/srt/managers/schedule_batch.py`; Torch came from the prepared ROCm interpreter environment.

Raw logs preserve import paths, GPU identity, the exact candidate tests, adjacent lifecycle tests, the legacy TypeError reproduction, and independent adversarial numerical checks.
