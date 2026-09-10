#!/usr/bin/env python3
import json
import re
from pathlib import Path


RESULT_PATH = Path("/job/raw/dsv4_generation_validation.json")
OUTPUT_PATH = Path("/job/raw/dsv4_acceptance_summary.json")
EOS_TOKEN_ID = 1


def extract_text(mode: str, response: dict) -> str:
    if mode == "chat":
        return response["choices"][0]["message"].get("content") or ""
    return response.get("text") or ""


def extract_ids(mode: str, response: dict) -> list[int]:
    if mode == "chat":
        return response["choices"][0].get("response_token_ids") or []
    return response.get("output_ids") or []


def finish_reason(mode: str, response: dict):
    if mode == "chat":
        return response["choices"][0].get("finish_reason")
    return (response.get("meta_info") or {}).get("finish_reason", {}).get("type")


def completion_tokens(mode: str, response: dict):
    if mode == "chat":
        return response.get("usage", {}).get("completion_tokens")
    return (response.get("meta_info") or {}).get("completion_tokens")


def semantic_checks(case: str, text: str) -> dict:
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", text))
    checks = {
        "arithmetic": {
            "contains_46": "46" in text,
            "at_least_two_sentences": sentence_count >= 2,
        },
        "day_night": {
            "mentions_earth": "Earth" in text,
            "mentions_rotation": "rotation" in text,
            "at_least_three_sentences": sentence_count >= 3,
        },
        "france": {
            "contains_paris": "Paris" in text,
            "at_least_two_sentences": sentence_count >= 2,
        },
        "sum_squares": {
            "defines_function": bool(re.search(r"\bdef\s+sum_squares\s*\(", text)),
            "contains_return_or_sum": bool(
                re.search(r"\b(return|sum)\s*\(", text)
            ),
            "mentions_expected_values": all(
                value in text for value in ("0", "14", "13")
            ),
        },
    }
    return checks[case]


def bounded_code_execution(text: str) -> dict:
    blocks = re.findall(r"```python\s*(.*?)\s*```", text, flags=re.DOTALL)
    if not blocks:
        return {"executed": False, "reason": "no Python code block"}
    source = blocks[0]
    try:
        code = compile(source, "<generated-sum-squares>", "exec")
    except SyntaxError as error:
        return {
            "executed": False,
            "reason": f"SyntaxError: {error.msg}",
            "source": source,
        }
    namespace = {}
    exec(code, namespace, namespace)
    function = namespace.get("sum_squares")
    if not callable(function):
        return {
            "executed": True,
            "reason": "sum_squares is not callable",
            "source": source,
        }
    try:
        observed = {
            "empty": function([]),
            "one_two_three": function([1, 2, 3]),
            "minus_two_three": function([-2, 3]),
        }
    except Exception as error:
        return {
            "executed": True,
            "reason": f"execution raised {type(error).__name__}: {error}",
            "source": source,
        }
    expected = {"empty": 0, "one_two_three": 14, "minus_two_three": 13}
    return {
        "executed": True,
        "observed": observed,
        "expected": expected,
        "correct": observed == expected,
        "source": source,
    }


def main() -> None:
    data = json.loads(RESULT_PATH.read_text())
    summary = {"metadata": data["metadata"], "modes": {}}
    for mode in ("chat", "raw_generate"):
        summary["modes"][mode] = {}
        for case, runs in data[mode].items():
            evaluated_runs = []
            for run in runs:
                response = run["response"]
                text = extract_text(mode, response)
                ids = extract_ids(mode, response)
                finish = finish_reason(mode, response)
                checks = semantic_checks(case, text)
                code_result = (
                    bounded_code_execution(text) if case == "sum_squares" else None
                )
                evaluated_runs.append({
                    "repeat": run["repeat"],
                    "status": run["status"],
                    "elapsed_seconds": run["elapsed_seconds"],
                    "finish_reason": finish,
                    "completion_tokens": completion_tokens(mode, response),
                    "output_token_count": len(ids),
                    "eos_token_present": EOS_TOKEN_ID in ids,
                    "last_output_token_ids": ids[-12:],
                    "semantic_checks": checks,
                    "code_execution": code_result,
                    "text": text,
                })
            repeats_equal = evaluated_runs[0]["text"] == evaluated_runs[1]["text"]
            all_status_200 = all(run["status"] == 200 for run in evaluated_runs)
            all_stop = all(run["finish_reason"] == "stop" for run in evaluated_runs)
            all_eos = all(run["eos_token_present"] for run in evaluated_runs)
            all_semantic = all(
                all(run["semantic_checks"].values()) for run in evaluated_runs
            )
            code_correct = (
                all(
                    run["code_execution"].get("correct") is True
                    for run in evaluated_runs
                )
                if case == "sum_squares"
                else True
            )
            summary["modes"][mode][case] = {
                "runs": evaluated_runs,
                "repeats_equal": repeats_equal,
                "all_status_200": all_status_200,
                "all_finish_stop": all_stop,
                "all_eos_token_present": all_eos,
                "all_semantic_checks": all_semantic,
                "code_correct": code_correct,
                "accepted": (
                    all_status_200
                    and all_stop
                    and all_eos
                    and repeats_equal
                    and all_semantic
                    and code_correct
                ),
            }
    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
