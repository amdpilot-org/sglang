"""Deterministic reproducer for sgl-project/sglang#37707."""

import json
import string

import xgrammar as xg

CHARS = list(string.printable[:95])
VOCAB = CHARS + ["<eos>"]
INFO = xg.TokenizerInfo(
    VOCAB, vocab_type=xg.VocabType.RAW, stop_token_ids=[len(VOCAB) - 1]
)
COMPILER = xg.GrammarCompiler(INFO)


def accepts(schema, document):
    matcher = xg.GrammarMatcher(COMPILER.compile_json_schema(json.dumps(schema)))
    for char in document:
        if char not in CHARS or not matcher.accept_string(char):
            return False
    return matcher.is_terminated() or matcher.accept_token(len(VOCAB) - 1)


def main():
    cases = [
        (
            "pattern+minLength rejects too short",
            {"type": "string", "pattern": "^[a-z]+$", "minLength": 5},
            '"ab"',
            False,
        ),
        (
            "pattern+minLength accepts boundary",
            {"type": "string", "pattern": "^[a-z]+$", "minLength": 5},
            '"abcde"',
            True,
        ),
        (
            "pattern+minLength rejects pattern violation",
            {"type": "string", "pattern": "^[a-z]+$", "minLength": 5},
            '"ABCDEF"',
            False,
        ),
        (
            "pattern+maxLength rejects too long",
            {"type": "string", "pattern": "^[a-z]+$", "maxLength": 5},
            '"abcdef"',
            False,
        ),
        (
            "pattern+maxLength accepts boundary",
            {"type": "string", "pattern": "^[a-z]+$", "maxLength": 5},
            '"abcde"',
            True,
        ),
        (
            "minLength alone rejects too short",
            {"type": "string", "minLength": 5},
            '"ab"',
            False,
        ),
        (
            "pattern alone rejects uppercase",
            {"type": "string", "pattern": "^[a-z]+$"},
            '"ABCDEF"',
            False,
        ),
        (
            "separate allOf branches",
            {
                "allOf": [
                    {"type": "string", "pattern": "^[a-z]+$"},
                    {"type": "string", "minLength": 5},
                ]
            },
            '"ab"',
            False,
        ),
    ]
    failed = []
    for name, schema, document, expected in cases:
        actual = accepts(schema, document)
        print(f"{name}: actual={actual} expected={expected}")
        if actual != expected:
            failed.append(name)
    if failed:
        raise AssertionError("schema constraints violated: " + ", ".join(failed))


if __name__ == "__main__":
    main()
