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
    "split_all_of": {
        "type": "object",
        "properties": {
            "v": {
                "allOf": [
                    {"type": "string", "pattern": "^[a-z]+$"},
                    {"type": "string", "minLength": 5},
                ]
            }
        },
    },
    "local_refs_split_all_of": {
        "$defs": {
            "letters": {"type": "string", "pattern": "^[a-z]+$"},
            "long": {"type": "string", "minLength": 5},
        },
        "type": "object",
        "properties": {
            "v": {
                "allOf": [
                    {"$ref": "#/$defs/letters"},
                    {"$ref": "#/$defs/long"},
                ]
            }
        },
    },
}

document = '{"v": "ab"}'
backend = object.__new__(XGrammarGrammarBackend)
backend.grammar_compiler = COMPILER
backend.any_whitespace = True
backend.override_stop_tokens = None
backend.vocab_size = len(VOCAB)

failures = 0
for name, schema in schemas.items():
    detected = has_xgrammar_unsupported_pattern_length_combination(schema)
    result = backend.dispatch_json(json.dumps(schema))
    rejected = isinstance(result, InvalidGrammarObject)
    invalid_accepted = accepts(schema, document)
    print(
        f"{name}: detected={detected} admission_rejected={rejected} "
        f"invalid_document_accepted={invalid_accepted}"
    )
    if not detected and not rejected and invalid_accepted:
        failures += 1

assert failures == 0, f"candidate misses {failures} lossy schema compositions"
