#!/usr/bin/env python3
"""Create the deterministic local model used by this qualification."""

import hashlib
import json
import os
import platform
import random
import sys
from importlib import metadata
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

SEED = 20260912
SPECIAL = ["<pad>", "<s>", "</s>", "<unk>"]
WORDS = [
    "hello",
    "world",
    "tiny",
    "llama",
    "rocm",
    "gpu",
    "test",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "red",
    "blue",
    "green",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "the",
    "a",
    "is",
    "on",
    "and",
    "response",
    "prompt",
    "batch",
    "stream",
]
VOCAB_TOKENS = SPECIAL + WORDS + [f"tok{i:03d}" for i in range(92)]


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    output = Path(
        os.environ.get(
            "SGLANG_QUAL_FIXTURE",
            "/job/cache/j-1fc9870820b9/tiny-random-llama",
        )
    )
    output.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)

    vocab = {token: index for index, token in enumerate(VOCAB_TOKENS)}
    tokenizer_impl = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_impl.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=256,
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant:"
    tokenizer.save_pretrained(output)

    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        bos_token_id=vocab["<s>"],
        eos_token_id=vocab["</s>"],
        pad_token_id=vocab["<pad>"],
        torch_dtype="float16",
    )
    model = LlamaForCausalLM(config).half().eval()
    model.save_pretrained(output, safe_serialization=True)

    weights = output / "model.safetensors"
    manifest = {
        "identity": "job-local deterministic tiny random Llama; no checkpoint download",
        "seed": SEED,
        "configuration": config.to_dict(),
        "tokenizer_vocabulary": vocab,
        "weight_file": weights.name,
        "weight_sha256": sha256(weights),
        "dependencies": {
            name: version(name)
            for name in (
                "sglang",
                "torch",
                "transformers",
                "tokenizers",
                "safetensors",
                "triton",
            )
        },
        "python": sys.version,
        "platform": platform.platform(),
    }
    (output / "fixture-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps({"fixture": str(output), "weight_sha256": manifest["weight_sha256"]})
    )


if __name__ == "__main__":
    main()
