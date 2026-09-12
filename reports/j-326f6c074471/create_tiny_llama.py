#!/usr/bin/env python3
"""Create the deterministic tiny Llama fixture qualified in mirror PR 649."""

import hashlib
import json
import os
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

SEED = 20260912
SPECIAL = ["<pad>", "<s>", "</s>", "<unk>"]
WORDS = "hello world tiny llama rocm gpu test one two three four five six seven eight nine red blue green alpha beta gamma delta the a is on and response prompt batch stream".split()


def main():
    output = Path(os.environ["SGLANG_QUAL_FIXTURE"])
    output.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    tokens = SPECIAL + WORDS + [f"tok{i:03d}" for i in range(92)]
    vocab = {token: index for index, token in enumerate(tokens)}
    tokenizer_impl = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_impl.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
        model_max_length=256,
    )
    tokenizer.save_pretrained(output)
    config = LlamaConfig(
        vocab_size=len(vocab), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, bos_token_id=vocab["<s>"],
        eos_token_id=vocab["</s>"], pad_token_id=vocab["<pad>"],
        torch_dtype="float16",
    )
    LlamaForCausalLM(config).half().eval().save_pretrained(
        output, safe_serialization=True
    )
    weights = output / "model.safetensors"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest = {
        "identity": "deterministic tiny random Llama from amdpilot-org/sglang PR 649 commit f1d603677ca76a9ea21124a544e405c5b0cbd315",
        "seed": SEED,
        "weight_sha256": digest,
        "path": str(output),
    }
    (output / "fixture-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
