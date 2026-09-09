#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from sglang.srt.entrypoints.openai import encoding_dsv4
from transformers import AutoTokenizer


CASES = {
    "arithmetic": "Explain 12 + 34 in two sentences.",
    "day_night": "Explain why day and night occur in three sentences.",
    "france": "What is the capital of France? Answer in a complete sentence, then add one relevant factual sentence.",
    "sum_squares": "Implement a Python function sum_squares(nums) that returns the sum of squares. Verify it returns 0 for [], 14 for [1, 2, 3], and 13 for [-2, 3].",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_path = Path("/models/DeepSeek-V4-Flash-0731")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    result = {
        "source": args.source,
        "encoding_dsv4_file": encoding_dsv4.__file__,
        "model_path": str(model_path),
        "thinking_mode": "chat",
        "reasoning_effort_profile": "official",
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "eos_token": tokenizer.eos_token,
        "cases": {},
    }
    for name, user_content in CASES.items():
        messages = [{"role": "user", "content": user_content}]
        prompt = encoding_dsv4.encode_messages(
            messages,
            thinking_mode="chat",
            reasoning_effort_profile="official",
        )
        token_ids = tokenizer.encode(prompt)
        result["cases"][name] = {
            "messages": messages,
            "formatted_prompt": prompt,
            "prompt_token_ids": token_ids,
            "prompt_token_count": len(token_ids),
            "round_trip": tokenizer.decode(token_ids),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
