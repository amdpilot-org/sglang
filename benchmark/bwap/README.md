# Batch-wise Adaptive Pruning (BWAP)

BWAP (`--enable-bwap`, arXiv:2608.14003) is an opt-in, default-off,
training-free FFN-neuron-pruning mode for gated-MLP models. It scores each
layer's `SiluAndMul` output, periodically builds a shared top-k mask from the
maximum score across exploring batch rows, and applies that mask during each
request's prune phase.

The default schedule is eight initial dense decode steps, then cycles of 16
prune steps followed by four dense exploration steps. Scheduling is per request,
so a request joining an active batch receives its own warmup. The neuron mask is
shared across the batch.

## Execution modes

- `--enable-bwap`: always-correct activation masking. This is useful for
  correctness and quality evaluation but does not reduce GEMM work.
- `--enable-bwap --bwap-fused`: on eligible all-prune decode steps, gather the
  retained gate/up rows and down-projection columns and execute reduced-width
  GEMMs. Eligibility is currently unquantized floating-point, bias-free gated
  MLPs with tensor parallel size 1. Other layers use activation masking.
- Decode CUDA graphs are supported with fused BWAP. Exploration or mixed-phase
  steps execute eagerly to update scores; all-prune steps replay a fixed-width
  graph. Version-gated `post_fill` callbacks update fixed-address gathered-weight
  buffers between replays when the adaptive mask changes.
- `--bwap-probe` captures a frozen dummy mask to measure the throughput ceiling.
  It deliberately produces invalid model output and requires both
  `--enable-bwap` and `--bwap-fused`.

The current limitations are deliberate: no quantized fused path, no fused
bias handling, no TP>1 reduced-width all-reduce path, and no global cross-TP
top-k. The masked fallback remains available, but does not provide a speedup.
The gathered weights and graph scratch buffers also add substantial memory
overhead (approximately another copy of the pruned FFN weights at 50% sparsity).

## Validation

Run the focused unit suite:

```bash
python -m pytest test/registered/unit/bwap/test_bwap_manager.py -v
```

It covers scoring, exact top-k cardinality, per-request scheduling, late joiners,
mixed batches, hook behavior, gathered-weight layout, fused-vs-masked numerical
equivalence, persistent graph buffers, adaptive graph gating, and invalid CLI
combinations.

For model-quality evaluation, compare a dense server with a BWAP server on the
same model and request order. For example:

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 30000 \
  --enable-bwap --bwap-fused --bwap-sparsity 0.5
python benchmark/gsm8k/bench_sglang.py --num-questions 200 --parallel 8
```

Accuracy is model- and sparsity-dependent; a transport/startup smoke test does
not establish semantic quality or throughput improvement. Benchmark throughput
against the same dense graph configuration and report model, hardware, batch
shape, schedule, sparsity, and accuracy together.
