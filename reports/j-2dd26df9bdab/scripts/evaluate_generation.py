#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


EOS_TOKEN_ID = 1


def extract_text(mode: str, response: dict) -> str:
    if response is None:
        return ""
    if mode == "chat":
        return ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return response.get("text") or ""


def extract_ids(mode: str, response: dict) -> list[int]:
    if response is None:
        return []
    if mode == "chat":
        choice = (response.get("choices") or [{}])[0]
        return choice.get("response_token_ids") or choice.get("token_ids") or []
    return response.get("output_ids") or []


def finish_reason(mode: str, response: dict):
    if response is None:
        return None
    if mode == "chat":
        return (response.get("choices") or [{}])[0].get("finish_reason")
    return ((response.get("meta_info") or {}).get("finish_reason") or {}).get("type")


def matched_stop(mode: str, response: dict):
    if response is None:
        return None
    if mode == "chat":
        return (response.get("choices") or [{}])[0].get("matched_stop")
    return ((response.get("meta_info") or {}).get("finish_reason") or {}).get("matched_stop")


def completion_tokens(mode: str, response: dict):
    if response is None:
        return None
    if mode == "chat":
        return (response.get("usage") or {}).get("completion_tokens")
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
            "contains_return_or_sum": bool(re.search(r"\b(return|sum)\s*\(", text)),
            "mentions_expected_values": all(value in text for value in ("0", "14", "13")),
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
        return {"executed": False, "reason": f"SyntaxError: {error.msg}", "source": source}
    namespace = {}
    exec(code, namespace, namespace)
    function = namespace.get("sum_squares")
    if not callable(function):
        return {"executed": True, "reason": "sum_squares is not callable", "source": source}
    observed = {
        "empty": function([]),
        "one_two_three": function([1, 2, 3]),
        "minus_two_three": function([-2, 3]),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute-code", action="store_true")
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
    summary = {"metadata": data["metadata"], "modes": {}}
    for mode in ("chat", "raw_generate"):
        summary["modes"][mode] = {}
        grouped = {}
        for record in data["records"]:
            if record["mode"] != mode:
                continue
            grouped.setdefault(record["case"], []).append(record)
        for case, runs in grouped.items():
            evaluated_runs = []
            for run in runs:
                response = run["response"]
                text = extract_text(mode, response)
                ids = extract_ids(mode, response)
                checks = semantic_checks(case, text)
                code_result = (
                    bounded_code_execution(text)
                    if case == "sum_squares" and args.execute_code
                    else None
                )
                evaluated_runs.append({
                    "repeat": run["repeat"],
                    "status": run["status"],
                    "elapsed_seconds": run["elapsed_seconds"],
                    "finish_reason": finish_reason(mode, response),
                    "matched_stop": matched_stop(mode, response),
                    "completion_tokens": completion_tokens(mode, response),
                    "output_token_count": len(ids),
                    "eos_token_present": EOS_TOKEN_ID in ids,
                    "last_output_token_ids": ids[-12:],
                    "semantic_checks": checks,
                    "code_execution": code_result,
                    "text": text,
                    "output_token_ids": ids,
                })
            texts_equal = all(run["text"] == evaluated_runs[0]["text"] for run in evaluated_runs)
            ids_equal = all(run["output_token_ids"] == evaluated_runs[0]["output_token_ids"] for run in evaluated_runs)
            all_status_200 = all(run["status"] == 200 for run in evaluated_runs)
            all_stop = all(run["finish_reason"] == "stop" for run in evaluated_runs)
            all_eos = all(run["eos_token_present"] for run in evaluated_runs)
            all_semantic = all(all(run["semantic_checks"].values()) for run in evaluated_runs)
            code_correct = (
                all(run["code_execution"].get("correct") is True for run in evaluated_runs)
                if case == "sum_squares" and args.execute_code
                else None
            )
            accepted = (
                all_status_200
                and all_stop
                and all_eos
                and texts_equal
                and ids_equal
                and all_semantic
                and (code_correct is not False)
            )
            summary["modes"][mode][case] = {
                "runs": evaluated_runs,
                "repeated_text_equal": texts_equal,
                "repeated_token_ids_equal": ids_equal,
                "all_status_200": all_status_200,
                "all_finish_stop": all_stop,
                "all_eos_token_present": all_eos,
                "all_semantic_checks": all_semantic,
                "code_correct": code_correct,
                "accepted": accepted,
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
