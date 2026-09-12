# cuDNN attention backend intake report

Upstream issue: https://github.com/sgl-project/sglang/issues/2272

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3033

## Result

Outcome: `unsupported_architecture`.

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, SGLang has neither a
`cudnn` entry in `ATTENTION_BACKEND_CHOICES` nor a corresponding factory in
`attention_registry.py`. The requested command therefore fails during argument
parsing:

```text
sglang serve: error: argument --attention-backend: invalid choice: 'cudnn'
EXIT_CODE=2
```

The pinned interpreter reports:

```text
torch: 2.11.0+rocm7.2
torch.version.cuda: null
torch.version.hip: 7.2.26015
device: AMD Instinct MI355X
cudnn frontend module: absent
```

The assigned GPU is therefore incapable of executing NVIDIA cuDNN. Installing
or replacing Torch was explicitly out of scope and would not make an AMD GPU a
valid target.

## Existing work reviewed

- Upstream PR 4650, “Preliminary support for the CuDNN attention backend,” was
  closed after reporting correctness and performance limitations.
- Upstream PR 5505, “Implement CuDNN Attention Backend with Graph Caching,” was
  also closed. Its discussion records a decode correctness fix requiring cuDNN
  sequence-length tensors to be `int32`, and says CUDA Graph support was still
  being completed.
- The head of PR 5505 (`407286248663946eea06bcc83ca31897013d4232`) overrides
  `init_forward_metadata_capture_cuda_graph` and
  `init_forward_metadata_replay_cuda_graph`. The current
  `AttentionBackend` contract says those hooks were removed and requires the
  split `init_forward_metadata_out_graph` / `init_forward_metadata_in_graph`
  lifecycle. Copying the closed implementation would therefore not preserve
  the current graph-capture contract.

## Required qualification not performed

No backend code was added because it could not be honestly qualified on this
node. A complete follow-up needs supported NVIDIA hardware and a pinned cuDNN
frontend, then must cover all of the following:

1. Current registry, CLI, platform, and dependency wiring.
2. Current KV-pool/index-translation contracts and page size 1.
3. BF16 and FP16 prefill plus decode against an independent numerical
   reference, including mixed sequence/prefix lengths and non-power-of-two
   batches.
4. Strict `int32` query/KV sequence metadata, with adversarial tests that catch
   the previously reported alternating-batch corruption.
5. Eager execution and CUDA Graph capture/replay using the current metadata
   lifecycle, including replay at smaller padded batch sizes.
6. Full Llama-3.1-8B-Instruct serving validation. The tiny Llama fixture named
   in the task cannot qualify cuDNN semantics or that model architecture.

No startup smoke or neighboring backend test is claimed as evidence that the
feature is implemented.
