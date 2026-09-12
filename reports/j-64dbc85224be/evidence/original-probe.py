#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backends.sglang.tools.probe_runtime_structures import run_probe
from backends.sglang.tools.run_live_action_plan_once import (
    _load_probe_records,
    _summarize_probe_result,
    _write_json,
)
from backends.sglang.poc.action_plan_recipes import (
    build_issue30294_chunked_prefill_cancel_action_plan,
)


DEFAULT_OUTPUT_ROOT = (
    Path.home()
    / "code"
    / "llm"
    / "InferFuzz-data"
    / "shared"
    / "sglang_poc_issue30294_chunked_prefill_cancel"
)


def _load_action_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"action plan is not a JSON object: {path}")
    return payload


def _negative_output_id_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    first_step = None
    min_value = None
    for step_index, row in enumerate(records, 1):
        tag = str(row.get("tag") or "")
        if tag not in {"schedule_batch", "schedule_batch_post_run"}:
            continue
        payload = dict(row.get("payload") or {})
        output_ids = payload.get("output_ids")
        if not isinstance(output_ids, dict):
            continue
        values = output_ids.get("values")
        if not isinstance(values, list):
            continue
        negatives = [int(value) for value in values if int(value) < 0]
        if not negatives:
            continue
        count += 1
        if first_step is None:
            first_step = {
                "record_index": step_index,
                "tag": tag,
                "forward_mode": payload.get("forward_mode"),
                "values": negatives[:8],
            }
        current_min = min(negatives)
        min_value = current_min if min_value is None else min(min_value, current_min)
    return {
        "negative_schedule_output_id_steps": count,
        "first_negative_schedule_output_id_step": first_step,
        "min_negative_schedule_output_id": min_value,
    }


def _finish_type(summary: dict[str, Any]) -> str:
    reason = dict(summary.get("last_finish_reason") or {})
    return str(reason.get("type") or "").strip().lower()


def _infer_cancelled_alias(action_plan: dict[str, Any]) -> str:
    for action in list(action_plan.get("actions") or []):
        if str(action.get("type") or "").strip().lower() != "cancel":
            continue
        aliases = list(action.get("request_aliases") or [])
        if aliases:
            return str(aliases[0])
        alias = str(action.get("request_alias") or "").strip()
        if alias:
            return alias
    raise RuntimeError("action plan does not contain a cancel action")


def _is_reproduced(row: dict[str, Any], *, cancelled_alias: str) -> tuple[bool, dict[str, Any]]:
    cancelled = dict(dict(row.get("per_alias") or {}).get(cancelled_alias) or {})
    output_tokens = int(cancelled.get("last_output_token_count") or 0)
    emitted_text = " ".join(str(cancelled.get("last_chunk_text") or "").split())
    reproduced = (
        bool(row.get("cancelled_alias_completed"))
        and cancelled_alias in list(row.get("completed_cancelled_aliases") or [])
        and _finish_type(cancelled) == "abort"
        and output_tokens >= 1
        and int(row.get("negative_schedule_output_id_steps") or 0) >= 32
    )
    evidence = {
        "cancelled_alias": cancelled_alias,
        "cancelled_alias_completed": row.get("cancelled_alias_completed"),
        "completed_cancelled_aliases": list(row.get("completed_cancelled_aliases") or []),
        "cancelled_text": emitted_text[:240],
        "cancelled_output_tokens": output_tokens,
        "cancelled_finish_reason": cancelled.get("last_finish_reason"),
        "negative_schedule_output_id_steps": row.get("negative_schedule_output_id_steps"),
        "min_negative_schedule_output_id": row.get("min_negative_schedule_output_id"),
        "first_negative_schedule_output_id_step": row.get(
            "first_negative_schedule_output_id_step"
        ),
    }
    return reproduced, evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the issue30294-style chunked-prefill overlap anomaly: a "
            "cancelled third request still emits one token before aborting, while "
            "the scheduler walks a long negative output-id ladder."
        )
    )
    parser.add_argument("--action-plan-file", default="")
    parser.add_argument("--variant", choices=["a", "b"], default="a")
    parser.add_argument("--stagger-ms", type=int, default=35)
    parser.add_argument("--cancel-delay-ms", type=int, default=120)
    parser.add_argument(
        "--model-path",
        default="/home/neil/code/llm/Llama-3.2-1B-Instruct",
    )
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.30)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--probe-limit", type=int, default=96)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if str(args.action_plan_file or "").strip():
        action_plan_path = Path(args.action_plan_file).expanduser().resolve()
        action_plan = _load_action_plan(action_plan_path)
    else:
        action_plan_path = None
        action_plan = build_issue30294_chunked_prefill_cancel_action_plan(
            variant=args.variant,
            stagger_ms=args.stagger_ms,
            cancel_delay_ms=args.cancel_delay_ms,
        )
    cancelled_alias = _infer_cancelled_alias(action_plan)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser().resolve() / (
        f"issue30294_chunked_prefill_cancel_{run_stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "action_plan.json", action_plan)

    rows: list[dict[str, Any]] = []
    reproduced_count = 0
    for attempt in range(max(1, int(args.repeat))):
        attempt_dir = output_dir / f"run_{attempt + 1:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        probe_path = attempt_dir / "probe.jsonl"
        response = run_probe(
            model_path=args.model_path,
            output=probe_path,
            prompt="unused",
            batch_size=1,
            max_new_tokens=8,
            temperature=0.0,
            gpu=0,
            disable_overlap_schedule=False,
            probe_limit=max(args.probe_limit, len(list(action_plan.get("actions") or [])) * 4),
            return_logprob=False,
            logprob_start_len=None,
            top_logprobs_num=None,
            json_schema=None,
            regex=None,
            ebnf=None,
            generate_kwargs_override=None,
            action_plan_override=action_plan,
            engine_kwargs_override=None,
            mem_fraction_static=args.mem_fraction_static,
        )
        records = _load_probe_records(probe_path)
        summary = _summarize_probe_result(
            response=response,
            records=records,
            sleep_ms=None,
        )
        summary.update(_negative_output_id_stats(records))
        reproduced, evidence = _is_reproduced(summary, cancelled_alias=cancelled_alias)
        if reproduced:
            reproduced_count += 1
        row = {
            "attempt": attempt + 1,
            "reproduced": reproduced,
            "evidence": evidence,
            **summary,
        }
        rows.append(row)
        _write_json(attempt_dir / "result.json", row)

    aggregate = {
        "case_name": "issue30294_chunked_prefill_cancel_poc",
        "action_plan_file": str(action_plan_path) if action_plan_path is not None else None,
        "variant": args.variant,
        "stagger_ms": int(args.stagger_ms),
        "cancel_delay_ms": int(args.cancel_delay_ms),
        "model_path": str(Path(args.model_path).expanduser()),
        "gpu": args.gpu,
        "mem_fraction_static": args.mem_fraction_static,
        "repeat": max(1, int(args.repeat)),
        "reproduced_count": reproduced_count,
        "stable_reproduction": reproduced_count == max(1, int(args.repeat)),
        "cancelled_alias": cancelled_alias,
        "rows": rows,
    }
    _write_json(output_dir / "summary.json", aggregate)
    print(json.dumps({"output_dir": str(output_dir), "summary": aggregate}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
