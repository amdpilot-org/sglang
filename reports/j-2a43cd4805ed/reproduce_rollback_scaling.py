import time

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar


class Matcher:
    def rollback(self, _count):
        pass


def measure_legacy(history_length, rounds):
    history = list(range(history_length))
    begin = time.perf_counter()
    for _ in range(rounds):
        history.append(-1)
        history = history[:-1]
    return (time.perf_counter() - begin) * 1e6 / rounds


def measure_fixed(history_length, rounds):
    grammar = XGrammarGrammar.__new__(XGrammarGrammar)
    grammar.matcher = Matcher()
    grammar.accepted_tokens = list(range(history_length))
    begin = time.perf_counter()
    for _ in range(rounds):
        grammar.accepted_tokens.append(-1)
        grammar.rollback(1)
    return (time.perf_counter() - begin) * 1e6 / rounds


for length, rounds in [(1_000, 5_000), (10_000, 2_000), (100_000, 500)]:
    print(
        f"length={length} legacy_us={measure_legacy(length, rounds):.3f} "
        f"fixed_us={measure_fixed(length, rounds):.3f}"
    )
