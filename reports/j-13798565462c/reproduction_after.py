import json

from sglang.srt.utils.request_logger import (
    _dataclass_to_string_truncated,
    _transform_data_for_logging,
)

big = "x" * 1_000_000
short = _transform_data_for_logging([big] * 10, 2048)
elided = _transform_data_for_logging([big] * 2049, 2048)
text = _dataclass_to_string_truncated([big] * 10, 2048)

print(f"json_short={len(json.dumps(short))}")
print(f"json_elided={len(json.dumps(elided))}")
print(
    f"json_elided_first_element={len(elided[0])} marker_count={elided.count('...')}"
)
print(f"text_short={len(text)} contains_full_element={big in text}")

assert len(elided[0]) == 2051
assert elided.count("...") == 1
assert big not in text
