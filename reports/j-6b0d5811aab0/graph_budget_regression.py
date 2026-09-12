import runpy
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.environ import envs
if len(sys.argv) == 2:
    utils = runpy.run_path(sys.argv[1])
    mqa_logits_budget_bytes = utils["mqa_logits_budget_bytes"]
    mqa_logits_row_bytes = utils["mqa_logits_row_bytes"]
    mqa_logits_rows_per_chunk = utils["mqa_logits_rows_per_chunk"]
else:
    from sglang.srt.layers.attention.dsa.utils import (
        mqa_logits_budget_bytes,
        mqa_logits_row_bytes,
        mqa_logits_rows_per_chunk,
    )


total = 80 << 30
device_module = SimpleNamespace(
    get_device_properties=MagicMock(
        return_value=SimpleNamespace(total_memory=total)
    )
)
schedule = SimpleNamespace(mem_fraction_static=0.9)
with envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.override(0.2):
    if len(sys.argv) == 2:
        mqa_logits_budget_bytes.__globals__["get_device_module"] = lambda: device_module
        mqa_logits_budget_bytes.__globals__["get_schedule"] = lambda: schedule
        graph_budget = mqa_logits_budget_bytes(device_index=0, allow_sync=False)
    else:
        with (
            patch(
                "sglang.srt.layers.attention.dsa.utils.get_device_module",
                return_value=device_module,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.utils.get_schedule",
                return_value=schedule,
            ),
        ):
            graph_budget = mqa_logits_budget_bytes(device_index=0, allow_sync=False)

row_bytes = mqa_logits_row_bytes(92992)
rows = mqa_logits_rows_per_chunk(
    num_rows=4096, row_bytes=row_bytes, budget_bytes=graph_budget
)
issue_allocation = 4096 * row_bytes
print(
    f"graph_budget={graph_budget} issue_allocation={issue_allocation} "
    f"rows_per_chunk={rows}"
)
assert issue_allocation == 1526726656
assert graph_budget == 512 << 20
assert rows == graph_budget // row_bytes
assert rows * row_bytes < int(1.17 * (1 << 30))
