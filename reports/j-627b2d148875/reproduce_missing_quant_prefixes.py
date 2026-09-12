#!/usr/bin/env python3
"""List quantization-aware model constructors that omit an explicit prefix."""

import ast
from pathlib import Path


QUANT_AWARE = {
    "ParallelLMHead",
    "ReplicatedLinear",
    "RowParallelLinear",
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "VocabParallelEmbedding",
    "RadixAttention",
    "FusedMoE",
}


def scan(root: Path):
    hits = []
    for source in sorted(root.rglob("*.py")):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            keywords = {keyword.arg for keyword in node.keywords}
            if (
                name in QUANT_AWARE
                and "quant_config" in keywords
                and "prefix" not in keywords
            ):
                hits.append((source.relative_to(root.parent), node.lineno, name))
    return hits


if __name__ == "__main__":
    model_root = Path(__file__).parents[2] / "python" / "sglang" / "srt" / "models"
    missing = scan(model_root)
    for path, line, constructor in missing:
        print(f"{path}:{line}: {constructor}")
    print(f"missing_prefix_sites={len(missing)}")
