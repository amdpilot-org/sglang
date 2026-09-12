from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that pull in sgl_kernel

from sglang.srt.entrypoints.openai.streaming_asr import StreamingASRState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestStreamingASRState(CustomTestCase):
    @staticmethod
    def _state() -> StreamingASRState:
        return StreamingASRState(
            chunk_size_sec=2.0,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
        )

    def test_word_extension_is_treated_as_revision(self):
        state = self._state()
        self.assertEqual(
            state.update("the cat one two three four five"),
            "the cat",
        )

        delta = state.update("the caterpillar one two three four five")

        self.assertEqual(delta, "caterpillar")
        self.assertEqual(state.emitted_text, "the cat caterpillar")

    def test_punctuation_suffix_stays_attached_in_prompt(self):
        state = self._state()
        self.assertEqual(
            state.update("hello world one two three four five"),
            "hello world",
        )

        delta = state.update("hello world, one two three four five")

        self.assertEqual(delta, ",")
        self.assertEqual(state.emitted_text, "hello world,")

    def test_unicode_punctuation_suffixes_stay_attached_in_prompt(self):
        for punctuation in ('"', "…", "—", "?!"):
            with self.subTest(punctuation=punctuation):
                state = self._state()
                self.assertEqual(
                    state.update("hello world one two three four five"),
                    "hello world",
                )

                delta = state.update(
                    f"hello world{punctuation} one two three four five"
                )

                self.assertEqual(delta, punctuation)
                self.assertEqual(state.emitted_text, f"hello world{punctuation}")

    def test_punctuation_rollback_does_not_repeat_text(self):
        state = self._state()
        self.assertEqual(
            state.update('hello world" one two three four five'),
            'hello world"',
        )

        self.assertEqual(
            state.update("hello world one two three four five"),
            "",
        )
        self.assertEqual(state.emitted_text, 'hello world"')

    def test_normal_word_append_keeps_separator(self):
        state = self._state()
        self.assertEqual(state.update("hello one two three four five"), "hello")

        delta = state.update("hello world one two three four five")

        self.assertEqual(delta, "world")
        self.assertEqual(state.emitted_text, "hello world")

    def test_temporary_word_boundary_rollback_does_not_repeat_text(self):
        state = self._state()
        self.assertEqual(
            state.update("alpha beta gamma one two three four five"),
            "alpha beta gamma",
        )
        self.assertEqual(
            state.update("alpha beta one two three four five"),
            "",
        )

        delta = state.update("alpha beta gamma delta one two three four five")

        self.assertEqual(delta, "delta")
        self.assertEqual(state.emitted_text, "alpha beta gamma delta")


if __name__ == "__main__":
    import unittest

    unittest.main()
