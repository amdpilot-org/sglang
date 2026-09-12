#!/usr/bin/env python3
"""Check the independently reviewed missing-prefix counterexamples."""

import ast
from pathlib import Path

EXPECTED = {
    ("bailing_moe_v3.py", "ParallelLMHead", "lm_head"),
    ("exaone.py", "RadixAttention", "attn"),
    ("grok.py", "FusedMoE", "experts"),
    ("whisper.py", "ParallelLMHead", "proj_out"),
}


def main() -> int:
    root = Path(__file__).parents[2] / "python/sglang/srt/models"
    found = set()
    for filename, constructor, suffix in EXPECTED:
        tree = ast.parse((root / filename).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != constructor:
                continue
            keywords = {item.arg: item.value for item in node.keywords}
            prefix = keywords.get("prefix")
            if prefix is not None and suffix in ast.unparse(prefix):
                found.add((filename, constructor, suffix))

    missing = EXPECTED - found
    for item in sorted(missing):
        print("missing:", item)
    print(f"qualified_review_counterexamples={len(found)}/{len(EXPECTED)}")
    return bool(missing)


if __name__ == "__main__":
    raise SystemExit(main())
