# Diffusion lifecycle correction generation 2

This change preserves the lifecycle API from candidate PR
https://github.com/amdpilot-org/sglang/pull/3065 at exact commit
`d2019b162a993952a2c4728ccad0a1b2b940fc4d` and corrects the concrete
malformed-response counterexample reported by independent review PR
https://github.com/amdpilot-org/sglang/pull/3120.

At the exact candidate commit, a scheduler response of `None` raised
`AttributeError: 'NoneType' object has no attribute 'error'`, and an object with
`error` but no `output` raised `AttributeError: ... has no attribute 'output'`.
`raw/failing_before.txt` retains that reproduction. The corrected implementation
raises lifecycle-specific `RuntimeError` messages for both cases.

`raw/passing_after.txt` retains the focused lifecycle and compatibility run:
15 tests passed. No real diffusion checkpoint was available, so model-specific
generation, memory behavior, disk refit, FSDP operation, and the requested
sleep+wake+refit versus kill-and-relaunch timing comparison remain unverified.
The available tiny Llama fixture cannot qualify those diffusion-specific paths.
No native code changed, and the correction itself did not require GPU execution.
