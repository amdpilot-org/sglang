#!/usr/bin/env python3
"""Fail when the independently reviewed quantization prefixes are absent."""

import ast
import sys
from pathlib import Path

EXPECTED = {
    ("persimmon.py", "ColumnParallelLinear", "dense_h_to_4h"),
    ("persimmon.py", "RowParallelLinear", "dense_4h_to_h"),
    ("persimmon.py", "QKVParallelLinear", "query_key_value"),
    ("persimmon.py", "RowParallelLinear", "dense"),
    ("persimmon.py", "ParallelLMHead", "lm_head"),
    ("voxtral.py", "QKVParallelLinear", "qkv_proj"),
    ("voxtral.py", "RowParallelLinear", "out_proj"),
    ("falcon_h1.py", "QKVParallelLinear", "qkv_proj"),
    ("falcon_h1.py", "RowParallelLinear", "o_proj"),
    ("hunyuan.py", "FusedMoE", "experts"),
    ("hunyuan.py", "ParallelLMHead", "lm_head"),
    ("gpt_j.py", "RadixAttention", "attn"),
    ("gpt_j.py", "ParallelLMHead", "lm_head"),
}


def main() -> int:
    root = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parents[2] / "python/sglang/srt/models"
    )
    found = set()
    for filename, constructor, suffix in EXPECTED:
        tree = ast.parse((root / filename).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != constructor:
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            prefix = keywords.get("prefix")
            if prefix is not None and suffix in ast.unparse(prefix):
                found.add((filename, constructor, suffix))

    missing = EXPECTED - found
    for item in sorted(missing):
        print("missing:", item)
    print(f"qualified_review_counterexamples={len(found)}/{len(EXPECTED)}")
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
