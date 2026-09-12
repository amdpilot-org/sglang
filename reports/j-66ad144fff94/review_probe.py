import json
import string

import xgrammar as xg

CHARS = list(string.printable[:95])
VOCAB = CHARS + ["<eos>"]
INFO = xg.TokenizerInfo(VOCAB, vocab_type=xg.VocabType.RAW, stop_token_ids=[len(VOCAB)-1])
COMPILER = xg.GrammarCompiler(INFO)

def accepts(schema, doc):
    matcher = xg.GrammarMatcher(COMPILER.compile_json_schema(json.dumps(schema)))
    for char in doc:
        if char not in CHARS or not matcher.accept_string(char):
            return False
    return matcher.is_terminated() or matcher.accept_token(len(VOCAB)-1)

CASES = {
    "original": ({"type":"object","properties":{"v":{"type":"string","pattern":"^[a-z]+$","minLength":5}}}, '{"v": "ab"}'),
    "object_allof": ({"type":"object","allOf":[{"properties":{"v":{"pattern":"^[a-z]+$"}}},{"properties":{"v":{"minLength":5}}}]}, '{"v": "ab"}'),
    "object_allof_refs": ({"type":"object","$defs":{"p":{"properties":{"v":{"pattern":"^[a-z]+$"}}},"l":{"properties":{"v":{"minLength":5}}}},"allOf":[{"$ref":"#/$defs/p"},{"$ref":"#/$defs/l"}]}, '{"v": "ab"}'),
    "nested_object_allof": ({"type":"object","allOf":[{"properties":{"v":{"type":"object","properties":{"w":{"pattern":"^[a-z]+$"}}}}},{"properties":{"v":{"type":"object","properties":{"w":{"minLength":5}}}}}]}, '{"v":{"w":"ab"}}'),
    "nested_object_allof_refs": ({"type":"object","$defs":{"p":{"properties":{"v":{"type":"object","properties":{"w":{"pattern":"^[a-z]+$"}}}}},"l":{"properties":{"v":{"type":"object","properties":{"w":{"minLength":5}}}}}},"allOf":[{"$ref":"#/$defs/p"},{"$ref":"#/$defs/l"}]}, '{"v":{"w":"ab"}}'),
}

for name, (schema, doc) in CASES.items():
    print(name, "raw_xgrammar_accepts_invalid=", accepts(schema, doc))
    try:
        from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
        backend = object.__new__(XGrammarGrammarBackend)
        backend.grammar_compiler = COMPILER
        backend.any_whitespace = True
        backend.override_stop_tokens = False
        backend.vocab_size = len(VOCAB)
        result = backend.dispatch_json(json.dumps(schema))
        print(name, "admission_result=", type(result).__name__, getattr(result, "error_message", None))
    except Exception as exc:
        print(name, "admission_probe_error=", repr(exc))
