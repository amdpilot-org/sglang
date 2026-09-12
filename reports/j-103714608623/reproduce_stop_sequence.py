"""Issue-specific stop-sequence probe using the real SGLang request path.

This intentionally drives ``Req.update_finish_state`` rather than reimplementing
the matcher.  Pass a local snapshot of the DeepSeek-V4-Flash-0731 tokenizer so
the test remains independent of model weights and GPU availability.
"""

import argparse
import json
from array import array

from transformers import AutoTokenizer

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams


def make_req(tokenizer, stop="<END>"):
    params = SamplingParams(max_new_tokens=64, stop=[stop])
    params.normalize(tokenizer=tokenizer)
    req = Req(
        rid="issue-36698",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=params,
        eos_token_ids=frozenset(),
        vocab_size=len(tokenizer),
    )
    req.tokenizer = tokenizer
    return req


def run_single_commit(tokenizer, text, expected):
    req = make_req(tokenizer)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    req.output_ids.extend(token_ids)
    req.update_finish_state(new_accepted_len=len(token_ids))
    through_stop = tokenizer.decode(req.output_ids_through_stop)
    trimmed = through_stop.split("<END>", 1)[0]
    assert req.finished()
    assert req.finished_reason.to_json() == {"type": "stop", "matched": "<END>"}
    assert trimmed == expected
    return {
        "case": text,
        "mode": "single_multi_token_commit",
        "token_ids": token_ids,
        "finished_len": req.finished_len,
        "through_stop": through_stop,
        "trimmed": trimmed,
    }


def run_incremental(tokenizer, text, expected):
    req = make_req(tokenizer)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    matched_at = None
    for index, token_id in enumerate(token_ids, start=1):
        req.output_ids.append(token_id)
        req.update_finish_state(new_accepted_len=1)
        if req.finished():
            matched_at = index
            break
    assert matched_at is not None
    through_stop = tokenizer.decode(req.output_ids_through_stop)
    trimmed = through_stop.split("<END>", 1)[0]
    assert trimmed == expected
    return {
        "case": text,
        "mode": "one_token_per_step",
        "token_ids": token_ids,
        "matched_at_token": matched_at,
        "finished_len": req.finished_len,
        "through_stop": through_stop,
        "trimmed": trimmed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_path")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    # SGLang's tokenizer wrapper supplies this runtime attribute. The direct
    # transformers load used by this standalone probe does not.
    tokenizer.additional_stop_token_ids = None

    results = []
    # Reported English control and Chinese failure case.
    results.append(run_single_commit(tokenizer, "banana<END>apple", "banana"))
    results.append(run_single_commit(tokenizer, "香蕉<END>苹果", "香蕉"))
    # Independent boundary: a multibyte prefix whose first token alone decodes
    # with a replacement character in this tokenizer.
    results.append(run_single_commit(tokenizer, "🙂香蕉<END>苹果", "🙂香蕉"))
    # Independent boundary: stop string completed over three decode steps.
    results.append(run_incremental(tokenizer, "香蕉<END>苹果", "香蕉"))

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
