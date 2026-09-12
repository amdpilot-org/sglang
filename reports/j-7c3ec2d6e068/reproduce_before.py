from sglang.srt.entrypoints.openai.utils import to_openai_style_logprobs


out = to_openai_style_logprobs(
    output_token_logprobs=[(-0.1, 1, "A")],
    output_top_logprobs=[
        [(-0.1, 1, "A"), (-0.5, 77, "�"), (-2.5, 88, "�")]
    ],
)

print(f"candidate_count={len(out.top_logprobs[0])}")
print(f"replacement_logprob={out.top_logprobs[0]['�']}")
assert len(out.top_logprobs[0]) == 3
assert out.top_logprobs[0]["�"] == -0.5
