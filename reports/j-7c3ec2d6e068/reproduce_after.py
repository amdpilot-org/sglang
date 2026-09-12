from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.utils import to_openai_style_logprobs


out = to_openai_style_logprobs(
    output_token_logprobs=[(-0.1, 1, "A")],
    output_top_logprobs=[
        [(-0.1, 1, "A"), (-0.5, 77, "�"), (-2.5, 88, "�")]
    ],
)
content = OpenAIServingChat._process_logprobs_tokens(
    None, out, use_token_index=True
)
candidates = [(item.token, item.logprob) for item in content[0].top_logprobs]

print(f"candidate_count={len(candidates)}")
print(f"candidates={candidates}")
assert candidates == [("A", -0.1), ("�", -0.5), ("�", -2.5)]
