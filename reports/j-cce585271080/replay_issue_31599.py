"""Deterministic replay for upstream issue 31599 against the real Req method."""

from array import array

from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams


STOP = 1


class Tokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def decode(self, ids):
        return "".join(
            "STOP" if int(token) == STOP else chr(97 + int(token) % 26)
            for token in ids
        )

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def make_req(output_ids, max_new_tokens, stop=("STOP",)):
    params = SamplingParams(max_new_tokens=max_new_tokens, stop=list(stop))
    params.normalize(tokenizer=Tokenizer())
    req = Req(
        rid="issue-31599",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=params,
        eos_token_ids=set(),
        vocab_size=10_000,
    )
    req.tokenizer = Tokenizer()
    req.output_ids = array("q", output_ids)
    return req


def check(
    name,
    output_ids,
    cap,
    expected_type,
    expected_len,
    expected_output,
    stop=("STOP",),
):
    req = make_req(output_ids, cap, stop=stop)
    req.update_finish_state(new_accepted_len=len(output_ids))
    actual_output = list(req.output_ids_through_stop)
    assert isinstance(req.finished_reason, expected_type), (name, req.finished_reason)
    assert req.finished_len == expected_len, (name, req.finished_len)
    assert actual_output == expected_output, (name, actual_output)
    print(
        f"PASS {name}: reason={type(req.finished_reason).__name__} "
        f"finished_len={req.finished_len} emitted={actual_output}"
    )


check(
    "reported_in_budget_stop",
    [10, 11, STOP, 20, 21, 22],
    5,
    FINISH_MATCHED_STR,
    3,
    [10, 11, STOP],
)
check(
    "stop_exactly_at_cap",
    [10, 11, 12, 20, STOP, 22],
    5,
    FINISH_MATCHED_STR,
    5,
    [10, 11, 12, 20, STOP],
)
check(
    "stop_beyond_cap",
    [10, 11, 12, 20, 21, STOP],
    5,
    FINISH_LENGTH,
    5,
    [10, 11, 12, 20, 21],
)
check(
    "no_stop_overshoot",
    [10, 11, 12, 20, 21, 22],
    5,
    FINISH_LENGTH,
    5,
    [10, 11, 12, 20, 21],
    stop=(),
)
