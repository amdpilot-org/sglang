#!/usr/bin/env python3
"""Create the deterministic tiny Llama fixture used by the sampling probe."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


VOCAB = {
    "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "user": 4,
    "assistant": 5, "hello": 6, "world": 7, "alpha": 8, "beta": 9,
    "gamma": 10, "delta": 11, "one": 12, "two": 13, "three": 14,
    "four": 15, "five": 16, "six": 17, "seven": 18, "eight": 19,
    "nine": 20, "ten": 21, ".": 22, ":": 23, "A": 24, "B": 25,
    "C": 26, "D": 27, "E": 28, "F": 29, "G": 30, "H": 31,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw = Tokenizer(WordLevel(vocab=VOCAB, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} : "
        "{{ message['content'] }} . {% endfor %}"
        "{% if add_generation_prompt %}assistant :{% endif %}"
    )
    tokenizer.save_pretrained(args.output)

    torch.manual_seed(20260912)
    config = LlamaConfig(
        vocab_size=len(VOCAB), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=2,
        pad_token_id=0, tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).save_pretrained(args.output, safe_serialization=True)

    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.iterdir()) if path.is_file()
    }
    (args.output / "SHA256SUMS.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(hashes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
