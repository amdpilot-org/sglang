# DeepSeek-V4 system-only correction

The fix from https://github.com/amdpilot-org/sglang/pull/1451 at exact commit
`7a71a1546b5b27eb420f37fe167ecbe2d76919d4` correctly rejects a system message
after conversation turns, including across `context` and new messages. That
valid behavior is preserved.

The counterexample from https://github.com/amdpilot-org/sglang/pull/1534 was
independently reproduced. A request containing only one or more leading system
messages passed the candidate validator and rendered no assistant generation
boundary. The OpenAI DSV4 path does not normalize that request into a user turn;
it passes the rendered string to the tokenizer. Therefore it could proceed to
model execution instead of producing a client error.

The consolidated correction rejects a conversation with no non-system message.
It retains leading system plus user/developer conversations and the candidate's
rejection of inline/trailing systems. A serving-path regression verifies the
error is raised before tokenizer execution; the common request handler maps the
`ValueError` to HTTP 400.

DeepSeek-V4-Flash-0731 weights were unavailable, so the reported long-context
first-token EOS behavior was not replayed. The deterministic validation defect
and pre-tokenization correction do not require model weights. No GPU execution
or native rebuild was applicable.
