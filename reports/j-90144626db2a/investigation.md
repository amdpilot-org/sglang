# Consolidated GDN mixed-dtype correction

Upstream issue: https://github.com/sgl-project/sglang/issues/31719

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2521

Candidate parent: https://github.com/amdpilot-org/sglang/pull/2401 at exact commit
`8926f3ebb5219cba35b8fcd68916bde4bb8fe74c`.

Independent review parent: https://github.com/amdpilot-org/sglang/pull/2487

## Result

Both review counterexamples were independently reproduced against the exact
candidate on the assigned AMD Instinct MI350X (`gfx950`). The correction keeps
the candidate's tracked-state cast and ordinary-prefill mixed-dtype helper,
then:

- routes the MIS convolution boundary through that helper; and
- gives padded sequences detached temporary state, scattering back only cache
  indices that are not `PAD_SLOT_ID=-1`.

## Failing before

`raw/candidate_counterexamples.txt` records the exact candidate source path and
GPU. The MIS call boundary failed Triton compilation for BF16 activations with
both FP32 and FP16 cache storage. Triton reported mismatched
`new_conv_state` branch types. The direct same-dtype kernel preserved the
padded row, while the candidate helper changed `conv_states[-1]`.

## Passing after

`reproduce_corrected_boundaries.py` uses a separate grouped
`torch.nn.functional.conv1d` reference. On gfx950, both formerly failing dtype
pairs compiled and matched output and active final-state references, while the
padded cache row remained exactly unchanged. The candidate's original four-way
GPU regression also remains passing. The focused policy suite passes 19 tests
and 27 subtests, and the existing causal-convolution GPU regression passes.

## Limitations

The reported QuantTrio/Qwen3.6-27B-AWQ weights, NVIDIA RTX 5090, CUDA 13.3, and
SM120 were unavailable. Full-model serving, model semantic accuracy, NVIDIA
behavior, and multi-node behavior were not tested. The tiny Llama fixture
cannot qualify the qwen3_5 hybrid-GDN architecture and was not substituted.
No native source changed, so a native rebuild was not applicable.
