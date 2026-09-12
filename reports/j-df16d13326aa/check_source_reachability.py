import ast
from pathlib import Path


SOURCE = Path(
    "python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py"
)


def main() -> None:
    text = SOURCE.read_text()
    tree = ast.parse(text)
    calls = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            "mqa_logits" in ast.unparse(node.func)
            or "_should_chunk_mqa_logits" in ast.unparse(node.func)
        )
    ]
    print(f"source={SOURCE}")
    print(f"has_fp8_wrapper={'def _fp8_mqa_logits' in text}")
    print(f"has_aiter_import={'from aiter' in text}")
    print(f"chunk_helper_call_count={text.count('self._should_chunk_mqa_logits(')}")
    print(f"relevant_calls={calls}")


if __name__ == "__main__":
    main()
