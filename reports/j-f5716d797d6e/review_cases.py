import importlib
import importlib.metadata
import json
import string
import sys

import xgrammar as xg


chars = list(string.printable[:95])
vocab = chars + ["<eos>"]
info = xg.TokenizerInfo(
    vocab, vocab_type=xg.VocabType.RAW, stop_token_ids=[len(vocab) - 1]
)
compiler = xg.GrammarCompiler(info)


def accepts(schema, doc):
    matcher = xg.GrammarMatcher(compiler.compile_json_schema(json.dumps(schema)))
    for ch in doc:
        if ch not in chars or not matcher.accept_string(ch):
            return False
    return matcher.is_terminated() or matcher.accept_token(len(vocab) - 1)


schemas = {
    "original_colocated": {
        "type": "object",
        "properties": {
            "v": {"type": "string", "pattern": "^[a-z]+$", "minLength": 5}
        },
    },
    "split_string_allof": {
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
    "split_ref_allof": {
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
    "split_object_allof": {
        "type": "object",
        "allOf": [
            {
                "properties": {
                    "v": {"type": "string", "pattern": "^[a-z]+$"}
                }
            },
            {"properties": {"v": {"type": "string", "minLength": 5}}},
        ],
    },
    "split_object_ref_allof": {
        "$defs": {
            "patterned_object": {
                "type": "object",
                "properties": {
                    "v": {"type": "string", "pattern": "^[a-z]+$"}
                },
            },
            "long_object": {
                "type": "object",
                "properties": {"v": {"type": "string", "minLength": 5}},
            },
        },
        "allOf": [
            {"$ref": "#/$defs/patterned_object"},
            {"$ref": "#/$defs/long_object"},
        ],
    },
}

print("python", sys.executable)
print("xgrammar", importlib.metadata.version("xgrammar"), xg.__file__)
try:
    module = importlib.import_module("sglang.srt.constrained.xgrammar_backend")
    detector = module.has_xgrammar_unsupported_pattern_length_combination
    print("backend", module.__file__)
except (ImportError, AttributeError) as exc:
    detector = None
    print("backend_detector_unavailable", repr(exc))

bad_doc = '{"v": "ab"}'
pattern_bad_doc = '{"v": "ABCDEF"}'
failed = False
for name, schema in schemas.items():
    invalid_accepted = accepts(schema, bad_doc)
    pattern_invalid_accepted = accepts(schema, pattern_bad_doc)
    detected = detector(schema) if detector else None
    print(
        name,
        "short_accepted=", invalid_accepted,
        "uppercase_accepted=", pattern_invalid_accepted,
        "detected=", detected,
    )
    if name in {"split_object_allof", "split_object_ref_allof"} and detector:
        failed |= invalid_accepted and not detected

sys.exit(1 if failed else 0)
