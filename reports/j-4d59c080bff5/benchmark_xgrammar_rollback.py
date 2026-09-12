import statistics
import time

import xgrammar as xgr

from sglang.srt.constrained.xgrammar_backend import (
    MAX_ROLLBACK_TOKENS,
    XGrammarGrammar,
)


def make_grammar(compiled_grammar):
    matcher = xgr.GrammarMatcher(
        compiled_grammar, max_rollback_tokens=MAX_ROLLBACK_TOKENS
    )
    return XGrammarGrammar(matcher, 1, compiled_grammar, None)


def measure(compiled_grammar, length, repeats=5):
    samples = []
    for _ in range(repeats):
        grammar = make_grammar(compiled_grammar)
        start = time.perf_counter()
        for _ in range(length):
            grammar.accept_token(0)  # committed token grows the debug history
            grammar.accept_token(0)  # speculative tree node
            grammar.rollback(1)
        samples.append(time.perf_counter() - start)
    return samples


tokenizer_info = xgr.TokenizerInfo(["a"], vocab_size=1)
compiler = xgr.GrammarCompiler(tokenizer_info)
compiled_grammar = compiler.compile_grammar(xgr.Grammar.from_ebnf('root ::= "a"*'))

print("backend=xgrammar; operation=actual XGrammarGrammar accept/rollback; unit=seconds")
previous = None
for length in (2000, 4000, 8000, 16000):
    samples = measure(compiled_grammar, length)
    median = statistics.median(samples)
    ratio = median / previous if previous is not None else float("nan")
    print(
        f"N={length} median={median:.6f} doubling_ratio={ratio:.3f} "
        f"samples={','.join(f'{sample:.6f}' for sample in samples)}"
    )
    previous = median
