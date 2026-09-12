import json
import string

import xgrammar as xg

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.xgrammar_backend import (
    XGrammarGrammarBackend,
    has_xgrammar_unsupported_pattern_length_combination,
)

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


schemas = {
    "object_allof": {
        "type": "object",
        "allOf": [
            {"properties": {"v": {"pattern": "^[a-z]+$"}}},
            {"properties": {"v": {"minLength": 5}}},
        ],
    },
    "object_allof_refs": {
        "type": "object",
        "$defs": {
            "patterned": {"properties": {"v": {"pattern": "^[a-z]+$"}}},
            "long": {"properties": {"v": {"minLength": 5}}},
        },
        "allOf": [
            {"$ref": "#/$defs/patterned"},
            {"$ref": "#/$defs/long"},
        ],
    },
}

backend = object.__new__(XGrammarGrammarBackend)
backend.grammar_compiler = COMPILER
backend.any_whitespace = True
backend.override_stop_tokens = None
backend.vocab_size = len(VOCAB)

for name, schema in schemas.items():
    detected = has_xgrammar_unsupported_pattern_length_combination(schema)
    rejected = isinstance(
        backend.dispatch_json(json.dumps(schema)), InvalidGrammarObject
    )
    raw_accepted = accepts(schema, '{"v": "ab"}')
    print(
        f"{name}: detected={detected} admission_rejected={rejected} "
        f"raw_xgrammar_accepts_invalid={raw_accepted}"
    )
    assert detected and rejected and raw_accepted
