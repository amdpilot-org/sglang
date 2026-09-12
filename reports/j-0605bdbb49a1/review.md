# Independent review of PR 1339

Candidate: https://github.com/amdpilot-org/sglang/pull/1339 at `d0bd05d4171be2654eaf62984012f50181ca65c0`

Upstream issue: https://github.com/sgl-project/sglang/issues/35736

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1373

## Verdict

Recommendation: **accept**. The candidate fully resolves the original source-level defect. No remaining counterexample was found within the reported prompt-construction contract.

## Evidence

The prepared checkout exactly matched the requested recorded base, `358c163250ad3b1f62939b01ce1314a0a31a0365`; the candidate's parent is that same commit.

On the base, an issue-shaped `ChatCompletionRequest` was passed through the real `OpenAIServingChat._process_messages` DSV4 path. With a non-empty `tools` array and `tool_choice="none"`, the tokenizer input contained all three forbidden artifacts:

```text
HAS_TOOLS_HEADER= True
HAS_TOOL_NAME= True
HAS_DSML_TOOL_CALLS= True
```

The prompt included the Bash schema and the full DSV4 instructions beginning with `## Tools` and `<｜DSML｜tool_calls>`. The negative assertion failed, reproducing the reported causal failure before switching to the candidate.

At the exact candidate commit, source imports resolved to:

```text
/job/repo/python/sglang/__init__.py
/job/repo/python/sglang/srt/entrypoints/openai/serving_chat.py
/job/repo/python/sglang/srt/entrypoints/openai/encoding_dsv4.py
```

The candidate regression passed (2 tests, 4 subtests). The entire `test_serving_chat.py` file also passed (137 tests, 72 subtests). Independent cases verified:

- DSV4 with `tool_choice="none"` and `reasoning_effort` unset, `high`, and `max` contains no tools header, tool schema/name, or DSML tool-call instruction.
- Request-level tools and tools already embedded in a system message are both absent from the encoded prompt.
- The original Pydantic request remains unchanged after processing.
- DSV3.2's corresponding `none` path also omits tool instructions.
- `auto` and `required` retain tools for both encoders. DSV3.2 correctly uses `<｜DSML｜function_calls>` while DSV4 uses `<｜DSML｜tool_calls>`.

The code change is correctly located after message copying and before encoder rendering. It therefore prevents schema/instruction exposure without mutating the caller's request. The existing serving logic already disables constraint/output parsing for `none`; removing the causal prompt injection closes the reported path while preserving enabled-tool behavior.

## Scope and limitations

No native or C++ source changed, so no native rebuild was applicable. The prepared interpreter is `/tmp/amdpilot-repo-j-0605bdbb49a1/venv/bin/python`, with PyTorch `2.11.0+rocm7.2`. The host exposes one AMD Instinct MI350X (`gfx950`), whereas the original report used two NVIDIA B300 GPUs. DeepSeek-V4-Flash weights were not available, so this review did not claim a full HTTP/model-generation reproduction, semantic accuracy validation, or multi-GPU validation. GPU execution was unnecessary for the deterministic prompt-construction contract.
