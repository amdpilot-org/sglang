import unittest

import xgrammar as xgr

from sglang.srt.constrained.xgrammar_backend import (
    MAX_ROLLBACK_TOKENS,
    XGrammarGrammar,
)


class TestXGrammarRollback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer_info = xgr.TokenizerInfo(["a"], vocab_size=1)
        compiler = xgr.GrammarCompiler(tokenizer_info)
        cls.compiled_grammar = compiler.compile_grammar(
            xgr.Grammar.from_ebnf('root ::= "a"*')
        )

    def make_grammar(self):
        matcher = xgr.GrammarMatcher(
            self.compiled_grammar, max_rollback_tokens=MAX_ROLLBACK_TOKENS
        )
        return XGrammarGrammar(matcher, 1, self.compiled_grammar, None)

    def test_rollback_updates_history_in_place(self):
        grammar = self.make_grammar()
        history = grammar.accepted_tokens

        for _ in range(4):
            grammar.accept_token(0)
        grammar.rollback(1)

        self.assertIs(grammar.accepted_tokens, history)
        self.assertEqual(grammar.accepted_tokens, [0, 0, 0])

    def test_rollback_boundary_histories(self):
        grammar = self.make_grammar()

        grammar.rollback(0)
        self.assertEqual(grammar.accepted_tokens, [])

        for _ in range(4):
            grammar.accept_token(0)
        grammar.rollback(0)
        self.assertEqual(grammar.accepted_tokens, [0, 0, 0, 0])

        grammar.rollback(4)
        self.assertEqual(grammar.accepted_tokens, [])


if __name__ == "__main__":
    unittest.main()
