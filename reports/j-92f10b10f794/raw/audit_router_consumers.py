from pathlib import Path

root = Path("python/sglang/srt/models")
models = (
    "hunyuan_v3.py",
    "minimax_m3.py",
    "sarvam_moe.py",
    "step3p5.py",
    "nemotron_h.py",
)
for name in models:
    text = (root / name).read_text()
    print(
        name,
        "RouterGate",
        "RouterGate(" in text,
        "local_gate_tuple",
        "router_logits, _ = self.gate" in text,
        "direct_gate_mm",
        "self.gate.weight.t()" in text,
    )

neutral = Path("python/sglang/kernels/ops/gemm/bf16_fp32.py").read_text()
legacy = Path("python/sglang/kernels/ops/attention/dsv4/gemm.py").read_text()
print("neutral_owns_dispatch", "def linear_bf16_fp32(" in neutral)
print("legacy_is_compatibility_import", "from sglang.kernels.ops.gemm" in legacy)
