#!/usr/bin/env python3
"""
Benchmark: paged GQA flash-decoding attention (the core RadixAttention decode op).

Operation under test
--------------------
SGLang's decode path is dominated by a single kernel: ``decode_attention_fwd``
(``python/sglang/kernels/ops/attention/decode_attention.py``), a Triton
"flash-decoding" kernel adapted from lightllm.  For each request in a decode
batch it takes one query token and attends it over that request's *paged* KV
cache (page size 1), described by:

    q            : [num_tokens, num_q_heads,   head_dim]   (1 token / request)
    k/v_buffer   : [num_pages, num_kv_heads, head_dim]     (paged KV store)
    req_to_token : [batch, seq_len]  int32  (page table: req -> buffer rows)
    b_seq_len    : [batch]           int32  (cached length per request)

and produces  o : [num_tokens, num_q_heads, head_dim].

This is exactly the operation RadixAttention reuses a prefix cache for, and it
is the latency-critical path of autoregressive serving -- SGLang ships a
``benchmark/kernels/decoding_attention_triton/`` harness for it.

``sglang`` is not importable without a build (no installed wheel), so this script
reimplements the *operation* faithfully in plain PyTorch: gather each request's
KV rows out of the paged buffer via ``req_to_token`` and run fused
``scaled_dot_product_attention`` (the same math, different parallelization).
This is a faithful reference, NOT the shipped Triton kernel -- see REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict

import os
import subprocess

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Workload
# --------------------------------------------------------------------------- #
@dataclass
class Workload:
    batch: int
    seq_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype

    @property
    def dtype_bytes(self) -> int:
        return torch.finfo(self.dtype).bits // 8


def build_paged_kv(wl: Workload, device: str, seed: int = 0):
    """Create q, paged k/v buffers, and the req_to_token page table.

    Mirrors the layout the shipped kernel consumes: page size 1, so
    req_to_token[i, j] is the buffer row holding request i's j-th cached token.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = wl.batch * wl.seq_len
    k_buf = torch.randn(n, wl.num_kv_heads, wl.head_dim, dtype=wl.dtype, generator=g)
    v_buf = torch.randn(n, wl.num_kv_heads, wl.head_dim, dtype=wl.dtype, generator=g)
    k_buf = k_buf.to(device)
    v_buf = v_buf.to(device)
    # contiguous, page-size-1 mapping (identity over the buffer for a clean test)
    req_to_token = torch.arange(n, device=device, dtype=torch.int32).view(
        wl.batch, wl.seq_len
    )
    b_seq_len = torch.full((wl.batch,), wl.seq_len, dtype=torch.int32, device=device)
    q = torch.randn(
        wl.batch, wl.num_q_heads, 1, wl.head_dim, dtype=wl.dtype, device=device
    )
    return q, k_buf, v_buf, req_to_token, b_seq_len


def paged_decode_attn(
    q: torch.Tensor,          # [B, Hq, 1, D]
    k_buf: torch.Tensor,      # [N, Hkv, D]
    v_buf: torch.Tensor,      # [N, Hkv, D]
    req_to_token: torch.Tensor,  # [B, L]
    b_seq_len: torch.Tensor,   # [B]
    sm_scale: float,
):
    """Faithful PyTorch reference of paged GQA decode attention.

    Gathers each request's KV rows via the page table, forms [B, L, Hkv, D],
    then runs fused scaled_dot_product_attention with GQA head broadcasting.
    """
    B, Hq, _, D = q.shape
    Hkv = k_buf.shape[1]
    L = req_to_token.shape[1]
    group = Hq // Hkv

    # paged gather: [B, L] rows -> [B, L, Hkv, D]  (one copy of KV; intrinsic to paging)
    idx = req_to_token.reshape(-1)            # [B*L]
    k = k_buf.index_select(0, idx.long()).view(B, L, Hkv, D)
    v = v_buf.index_select(0, idx.long()).view(B, L, Hkv, D)

    # Move heads next to batch so GQA can fold the group (Hq/Hkv) in via stride-0
    # expand with NO KV materialization. permute+contiguous is one KV copy (the only
    # extra traffic vs the fused shipped kernel); repeat_interleave would copy
    # group-x KV. enable_gqa=True is not honored on this ROCm SDPA path.
    k = k.permute(0, 2, 1, 3).contiguous()        # [B, Hkv, L, D]
    v = v.permute(0, 2, 1, 3).contiguous()        # [B, Hkv, L, D]
    q = q.view(B, Hkv, group, 1, D).reshape(B * Hkv, group, 1, D)       # [B*Hkv, group, 1, D]
    k = k.view(B, Hkv, 1, L, D).reshape(B * Hkv, 1, L, D).expand(B * Hkv, group, L, D)
    v = v.view(B, Hkv, 1, L, D).reshape(B * Hkv, 1, L, D).expand(B * Hkv, group, L, D)

    o = F.scaled_dot_product_attention(
        q, k, v, scale=sm_scale, is_causal=False
    )                                              # [B*Hkv, group, 1, D]
    return o.view(B, Hkv, group, 1, D).reshape(B, Hq, 1, D)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def time_op(fn, warmup: int, repeats: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))  # ms
    return times


def summarize(ms):
    ms = sorted(ms)
    n = len(ms)
    def pct(p):
        if n == 1:
            return ms[0]
        k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
        return ms[k]
    return {
        "n": n,
        "min_ms": ms[0],
        "max_ms": ms[-1],
        "mean_ms": statistics.fmean(ms),
        "median_ms": statistics.median(ms),
        "p99_ms": pct(99),
        "std_ms": statistics.pstdev(ms) if n > 1 else 0.0,
        "spread_pct": 100.0 * (ms[-1] - ms[0]) / ms[0] if ms[0] > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Theoretical traffic / FLOPs (decode, memory-bound on KV read)
# --------------------------------------------------------------------------- #
def kv_bytes_read(wl: Workload) -> int:
    # K + V, each [B, L, Hkv, D]
    return 2 * wl.batch * wl.seq_len * wl.num_kv_heads * wl.head_dim * wl.dtype_bytes


def attn_flops(wl: Workload) -> int:
    # QK^T (2 B Hq L D) + softmax.scale (negligible) + PV (2 B Hq L D)
    return 4 * wl.batch * wl.num_q_heads * wl.seq_len * wl.head_dim


# --------------------------------------------------------------------------- #
# Env
# --------------------------------------------------------------------------- #
def env_info():
    info = {}
    info["torch_version"] = torch.__version__
    info["torch_hip_version"] = getattr(torch.version, "hip", None)
    info["cuda_available"] = torch.cuda.is_available()
    info["device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        p = torch.cuda.get_device_properties(0)
        info["device_props"] = {
            "name": p.name,
            "total_memory_GB": round(p.total_memory / 1024**3, 3),
            "major": p.major,
            "minor": p.minor,
            "multi_processor_count": getattr(p, "multi_processor_count", None),
        }
        info["device_capability"] = tuple(torch.cuda.get_device_capability(0))
    return info


def rocm_env():
    """Read ROCm/HIP + GPU identity from the machine (best-effort, never assumes)."""
    out = {}
    for path in ("/opt/rocm/.info/version", "/opt/rocm-7.2.0/.info/version"):
        try:
            with open(path) as f:
                out["rocm_version_file"] = f.read().strip()
                out["rocm_version_path"] = path
                break
        except OSError:
            continue
    def _run(cmd):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            return (r.stdout + r.stderr).strip()
        except Exception as e:
            return f"<unavailable: {e}>"
    out["rocm_smi_product"] = _run(["rocm-smi", "--showproductname"])
    out["rocm_smi_clocks"] = _run(["rocm-smi", "--showclocks"])
    out["hipcc_version"] = _run(["hipcc", "--version"])
    # keep rocminfo terse: just the GPU agent's name/marketing/isa lines
    rmin = _run(["rocminfo"])
    keep = []
    for line in rmin.splitlines():
        ls = line.strip()
        if ls.startswith(("Name:", "Marketing Name:", "Vendor Name:")) and (
            "gfx" in ls or "Instinct" in ls
        ):
            keep.append(ls)
        if ls.startswith("Name:") and "gfx9" in ls:
            keep.append(ls)
    out["rocminfo_gpu"] = keep
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _self_check(device: str):
    """Verify paged_decode_attn against a manual per-head reference on a tiny shape."""
    dt = torch.float32
    B, Hq, Hkv, L, D = 2, 8, 2, 16, 4
    group = Hq // Hkv
    g = torch.Generator(device="cpu").manual_seed(1)
    q = torch.randn(B, Hq, 1, D, dtype=dt, generator=g).to(device)
    k_buf = torch.randn(B * L, Hkv, D, dtype=dt, generator=g).to(device)
    v_buf = torch.randn(B * L, Hkv, D, dtype=dt, generator=g).to(device)
    r2t = torch.arange(B * L, device=device, dtype=torch.int32).view(B, L)
    bsl = torch.full((B,), L, dtype=torch.int32, device=device)
    sc = 1.0 / (D ** 0.5)
    with torch.inference_mode():
        o = paged_decode_attn(q, k_buf, v_buf, r2t, bsl, sc)
    ref = torch.empty_like(o)
    for b in range(B):
        for h in range(Hq):
            hk = h // group
            kk = k_buf[r2t[b].long()]            # [L, Hkv, D]
            vv = v_buf[r2t[b].long()]
            scores = (q[b, h, 0] * kk[:, hk]) * sc   # [L, D]
            scores = scores.sum(-1)                  # [L]
            p = scores.softmax(-1)
            ref[b, h, 0] = (p[:, None] * vv[:, hk]).sum(0)
    md = (o - ref).abs().max().item()
    ok = torch.allclose(o, ref, atol=1e-5)
    print(f"self-check: max_abs_diff={md:.2e} allclose={ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batches", default="1,4,16,64,256")
    ap.add_argument("--seq-lens", default="1024,4096,16384")
    ap.add_argument("--num-q-heads", type=int, default=32)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--json", default=None, help="write results json here")
    ap.add_argument("--check", action="store_true", help="run a correctness self-check vs a manual reference, then exit")
    args = ap.parse_args()

    if args.check:
        _self_check(args.device)
        return

    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]

    env = env_info()
    env["rocm"] = rocm_env()
    print("=== environment ===")
    print(json.dumps(env, indent=2))
    print("=== workload sweep ===")
    print(f"q_heads={args.num_q_heads} kv_heads={args.num_kv_heads} "
          f"head_dim={args.head_dim} dtype={args.dtype} "
          f"warmup={args.warmup} repeats={args.repeats}")

    results = {"env": env, "config": vars(args), "cases": []}
    sm_scale = 1.0 / (args.head_dim ** 0.5)

    header = f"{'B':>4} {'L':>6} {'min_ms':>9} {'mean_ms':>9} {'med_ms':>9} {'p99_ms':>9} {'std_ms':>8} {'spread%':>8} {'kv_GB/s':>10} {'TFLOPS':>8}"
    print(header)
    print("-" * len(header))

    for B in batches:
        for L in seq_lens:
            wl = Workload(B, L, args.num_q_heads, args.num_kv_heads, args.head_dim, dt)
            q, k_buf, v_buf, r2t, bsl = build_paged_kv(wl, args.device)

            def fn():
                paged_decode_attn(q, k_buf, v_buf, r2t, bsl, sm_scale)

            times = time_op(fn, args.warmup, args.repeats)
            s = summarize(times)
            kv_b = kv_bytes_read(wl)
            fl = attn_flops(wl)
            kv_gbs = kv_b / (s["min_ms"] / 1e3) / 1e9
            tflops = fl / (s["min_ms"] / 1e3) / 1e12
            print(f"{B:>4} {L:>6} {s['min_ms']:>9.3f} {s['mean_ms']:>9.3f} "
                  f"{s['median_ms']:>9.3f} {s['p99_ms']:>9.3f} {s['std_ms']:>8.3f} "
                  f"{s['spread_pct']:>8.2f} {kv_gbs:>10.1f} {tflops:>8.3f}")
            results["cases"].append({
                "batch": B, "seq_len": L,
                "stats": s,
                "kv_bytes_read": kv_b,
                "attn_flops": fl,
                "kv_bandwidth_GBps_min": kv_gbs,
                "tflops_min": tflops,
            })
            del q, k_buf, v_buf, r2t, bsl
            torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
