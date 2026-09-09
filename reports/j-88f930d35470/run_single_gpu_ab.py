from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
from typing import Any

import torch


SENTINEL = 0xA5
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
GROUP_COUNT = NOPE_DIM // 64


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tensor(path: pathlib.Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.detach().cpu().contiguous().view(torch.uint16)
    else:
        tensor = tensor.detach().cpu().contiguous()
    path.write_bytes(tensor.numpy().tobytes())


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def loaded_private_jit_paths(cache_root: pathlib.Path) -> list[str]:
    paths: set[str] = set()
    for map_line in pathlib.Path("/proc/self/maps").read_text().splitlines():
        fields = map_line.split()
        if fields and fields[-1].endswith(".so") and cache_root.as_posix() in fields[-1]:
            paths.add(fields[-1])
    return sorted(paths)


def compiler_identity() -> dict[str, str]:
    hipcc = pathlib.Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin/hipcc"
    completed = subprocess.run(
        [str(hipcc), "--version"], check=True, capture_output=True, text=True
    )
    return {"hipcc": completed.stdout.strip()}


def make_gamma() -> tuple[torch.Tensor, torch.Tensor]:
    nope_pattern = torch.tensor(
        [64, 128, 160, 224, -64, -128, -160, -224],
        dtype=torch.float32,
    )
    rope_pattern = torch.tensor(
        [1, -2, 0.5, -0.75], dtype=torch.float32
    ).repeat(16)
    gamma_reference = torch.empty(HEAD_DIM, dtype=torch.float32)
    for group_index in range(GROUP_COUNT):
        gamma_reference[group_index * 64 : (group_index + 1) * 64] = (
            (nope_pattern * (2.0 ** -group_index)).repeat(8)
        )
    gamma_reference[NOPE_DIM:] = rope_pattern
    return gamma_reference.to(torch.bfloat16), gamma_reference


def expected_nope_bytes(sign: int) -> torch.Tensor:
    nope_pattern = torch.tensor(
        [64, 128, 160, 224, -64, -128, -160, -224],
        dtype=torch.float32,
    )
    if sign == -1:
        nope_pattern = -nope_pattern
    group_bytes = nope_pattern.to(torch.float8_e4m3fnuz).view(torch.uint8).repeat(8)
    return group_bytes.repeat(GROUP_COUNT)


def expected_cache(
    page_size: int,
    ratio: int,
    out_loc: torch.Tensor,
    gamma_reference: torch.Tensor,
) -> torch.Tensor:
    page_bytes = ((584 * page_size + 575) // 576) * 576
    cache = torch.full((3, page_bytes), SENTINEL, dtype=torch.uint8)
    positive_nope_bytes = expected_nope_bytes(1)
    negative_nope_bytes = expected_nope_bytes(-1)
    positive_rope = gamma_reference[NOPE_DIM:].to(torch.bfloat16).view(torch.uint8)
    negative_rope = (-gamma_reference[NOPE_DIM:]).to(torch.bfloat16).view(torch.uint8)

    for row_index, sign in enumerate((1, -1)):
        location = int(out_loc[row_index].item())
        page = location // page_size
        slot = location % page_size
        value_start = slot * 576
        cache[page, value_start : value_start + NOPE_DIM] = (
            positive_nope_bytes if sign == 1 else negative_nope_bytes
        )
        cache[page, value_start + NOPE_DIM : value_start + 576] = (
            positive_rope if sign == 1 else negative_rope
        )
        scale_start = page_size * 576 + slot * 8
        expected_scales = torch.tensor(
            [127, 126, 125, 124, 123, 122, 121], dtype=torch.uint8
        )
        cache[page, scale_start : scale_start + 7] = expected_scales
    return cache


def run_producer_page(
    module: Any,
    page_size: int,
    ratio: int,
    gamma: torch.Tensor,
    gamma_reference: torch.Tensor,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    input_tensor = torch.ones((3, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    input_tensor[1] = -1
    freqs_cis = torch.zeros((ratio + 1, ROPE_DIM), dtype=torch.float32, device="cuda")
    freqs_cis[:, 0::2] = 1
    out_loc = torch.tensor(
        [page_size - 1, page_size + 1, 2 * page_size],
        dtype=torch.int64,
        device="cuda",
    )
    plan_int32 = torch.zeros((3, 4), dtype=torch.int32, device="cuda")
    plan_int32[:, 0] = torch.tensor(
        [ratio, 2 * ratio, ratio + 1], dtype=torch.int32, device="cuda"
    )
    plan = plan_int32.view(torch.uint8)
    page_bytes = ((584 * page_size + 575) // 576) * 576
    cache = torch.full((3, page_bytes), SENTINEL, dtype=torch.uint8, device="cuda")

    module.forward(
        input_tensor,
        plan,
        gamma,
        0.0,
        freqs_cis,
        out_loc,
        cache,
        True,
        ratio,
    )
    torch.cuda.synchronize()

    expected = expected_cache(page_size, ratio, out_loc.cpu(), gamma_reference.cpu()).cuda()
    full_equal = torch.equal(cache, expected)
    mismatch = (cache != expected).nonzero(as_tuple=False)
    mismatch_indices = (
        mismatch[:32, 0] * page_bytes + mismatch[:32, 1]
    ).tolist()
    mismatch_actual = [int(cache.flatten()[index].item()) for index in mismatch_indices]
    mismatch_expected = [int(expected.flatten()[index].item()) for index in mismatch_indices]

    payload_equal = True
    scale_equal = True
    rope_equal = True
    decoded_equal = True
    positive_nope_bytes = expected_nope_bytes(1).cuda()
    negative_nope_bytes = expected_nope_bytes(-1).cuda()
    for row_index, sign in enumerate((1, -1)):
        location = int(out_loc[row_index].item())
        page = location // page_size
        slot = location % page_size
        value_start = slot * 576
        scale_start = page_size * 576 + slot * 8
        payload_equal &= bool(
            torch.equal(
                cache[page, value_start : value_start + NOPE_DIM],
                positive_nope_bytes if sign == 1 else negative_nope_bytes,
            )
        )
        scale_equal &= bool(
            torch.equal(
                cache[page, scale_start : scale_start + 7],
                torch.tensor([127, 126, 125, 124, 123, 122, 121], dtype=torch.uint8, device="cuda"),
            )
        )
        expected_rope = (
            gamma_reference[NOPE_DIM:]
            if sign == 1
            else -gamma_reference[NOPE_DIM:]
        ).to(torch.bfloat16).view(torch.uint8).cuda()
        rope_equal &= bool(
            torch.equal(
                cache[page, value_start + NOPE_DIM : value_start + 576], expected_rope
            )
        )
        for group_index in range(GROUP_COUNT):
            actual_bytes = cache[
                page,
                value_start + group_index * 64 : value_start + (group_index + 1) * 64,
            ].view(torch.float8_e4m3fnuz)
            decoded = actual_bytes.to(torch.float32) * (2.0 ** -group_index)
            reference = (
                gamma_reference[
                    group_index * 64 : (group_index + 1) * 64
                ]
                if sign == 1
                else -gamma_reference[
                    group_index * 64 : (group_index + 1) * 64
                ]
            )
            decoded_equal &= bool(torch.equal(decoded, reference.cuda()))

    stem = output_dir / f"producer_p{page_size}_r{ratio}"
    write_tensor(stem.with_suffix(".actual.bin"), cache)
    write_tensor(stem.with_suffix(".expected.bin"), expected)
    write_tensor(stem.with_suffix(".input.bin"), input_tensor)
    write_tensor(stem.with_suffix(".gamma.bin"), gamma)
    write_tensor(stem.with_suffix(".freqs_cis.bin"), freqs_cis)
    write_tensor(stem.with_suffix(".out_loc.bin"), out_loc)
    write_tensor(stem.with_suffix(".plan.bin"), plan)

    return {
        "page_size": page_size,
        "compression_ratio": ratio,
        "page_bytes": page_bytes,
        "out_loc": out_loc.cpu().tolist(),
        "plan_seq_len": [ratio, 2 * ratio, ratio + 1],
        "full_cache_equal": bool(full_equal),
        "payload_equal": bool(payload_equal),
        "scale_equal": bool(scale_equal),
        "rope_equal": bool(rope_equal),
        "decoded_nope_equal": bool(decoded_equal),
        "mismatch_count": int(mismatch.numel()),
        "first_mismatch_flat_indices": mismatch_indices,
        "first_mismatch_actual_hex": [f"{value:02X}" for value in mismatch_actual],
        "first_mismatch_expected_hex": [f"{value:02X}" for value in mismatch_expected],
    }


def pack_values() -> tuple[list[tuple[str, list[float]]], dict[str, float]]:
    half_subnormal = 2.0 ** -11
    positive_below = torch.nextafter(
        torch.tensor(half_subnormal), torch.tensor(0.0)
    ).item()
    positive_above = torch.nextafter(
        torch.tensor(half_subnormal), torch.tensor(float("inf"))
    ).item()
    negative_above_zero = torch.nextafter(
        torch.tensor(-half_subnormal), torch.tensor(0.0)
    ).item()
    negative_below = torch.nextafter(
        torch.tensor(-half_subnormal), torch.tensor(float("-inf"))
    ).item()

    base_pairs: list[tuple[str, list[float]]] = [
        ("first-six-0", [1.0, -1.0]),
        ("first-six-1", [128.0, 160.0]),
        ("first-six-2", [224.0, -1e-8]),
        ("signed-zero", [0.0, -0.0]),
        ("half-subnormal-positive-below", [half_subnormal, positive_below]),
        ("half-subnormal-positive-above", [half_subnormal, positive_above]),
        ("half-subnormal-negative-above-zero", [-half_subnormal, negative_above_zero]),
        ("half-subnormal-negative-below", [-half_subnormal, negative_below]),
        (
            "subnormal-boundaries",
            [
                0.75 * 2.0 ** -10,
                -0.75 * 2.0 ** -10,
                2.0 ** -10,
                -2.0 ** -10,
                1.5 * 2.0 ** -10,
                -1.5 * 2.0 ** -10,
                7.5 * 2.0 ** -10,
                -7.5 * 2.0 ** -10,
                2.0 ** -7,
                -2.0 ** -7,
            ],
        ),
        ("rne-ties", [1.0625, -1.0625, 1.1875, -1.1875]),
        (
            "top-segment",
            [
                120.0,
                -120.0,
                124.0,
                -124.0,
                128.0,
                -128.0,
                144.0,
                -144.0,
                160.0,
                -160.0,
                192.0,
                -192.0,
                224.0,
                -224.0,
                240.0,
                -240.0,
                256.0,
                -256.0,
                448.0,
                -448.0,
            ],
        ),
    ]

    cases: list[tuple[str, list[float]]] = []
    for case_name, values in base_pairs:
        if len(values) % 2:
            values = values + [0.0]
        cases.append((case_name, values))
        cases.append((case_name + "-swapped", list(reversed(values))))

    neighbors = {
        "half_subnormal": half_subnormal,
        "positive_below": positive_below,
        "positive_above": positive_above,
        "negative_above_zero": negative_above_zero,
        "negative_below": negative_below,
    }
    return cases, neighbors


def run_pack_probe(
    probe_module: Any,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    cases, neighbors = pack_values()
    results: list[dict[str, Any]] = []
    all_equal = True
    for case_name, values in cases:
        input_tensor = torch.tensor(values, dtype=torch.float32, device="cuda")
        output_tensor = torch.empty_like(input_tensor, dtype=torch.uint8)
        metadata_tensor = torch.zeros(4, dtype=torch.float32, device="cuda")
        probe_module.run(input_tensor, output_tensor, metadata_tensor)
        torch.cuda.synchronize()
        expected = input_tensor.clamp(min=-224.0, max=224.0).to(
            torch.float8_e4m3fnuz
        ).view(torch.uint8)
        case_equal = bool(torch.equal(output_tensor, expected))
        all_equal &= case_equal
        mismatch = (output_tensor != expected).nonzero(as_tuple=False).flatten().tolist()
        stem = output_dir / f"pack_{case_name}"
        write_tensor(stem.with_suffix(".input.bin"), input_tensor)
        write_tensor(stem.with_suffix(".input_bits.bin"), input_tensor.view(torch.uint32))
        write_tensor(stem.with_suffix(".actual.bin"), output_tensor)
        write_tensor(stem.with_suffix(".expected.bin"), expected)
        results.append(
            {
                "case": case_name,
                "values": values,
                "equal": case_equal,
                "mismatch_indices": mismatch,
                "actual_hex": [f"{value:02X}" for value in output_tensor.cpu().tolist()],
                "expected_hex": [f"{value:02X}" for value in expected.cpu().tolist()],
            }
        )

    metadata = metadata_tensor.cpu()
    return {
        "all_equal": bool(all_equal),
        "policy_max": float(metadata[0].item()),
        "fnuz_macro": float(metadata[1].item()),
        "hardware_cvt": float(metadata[2].item()),
        "neighbors": neighbors,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--probe-source", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    cache_root = pathlib.Path(os.environ["SGLANG_JIT_CACHE_DIR"]).expanduser().resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import sglang
    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.ops.attention.dsv4.compress import (
        _jit_compress_norm_rope_module,
    )

    torch.cuda.set_device(0)
    device_properties = torch.cuda.get_device_properties(0)
    source_root = args.source_root.resolve()
    fused_source = (
        source_root
        / "python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh"
    )
    fp8_header = (
        source_root
        / "python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp8_utils.cuh"
    )

    gamma, gamma_reference = make_gamma()
    producer_results = []
    modules = {}
    for page_size, ratio in ((64, 4), (2, 128)):
        module = _jit_compress_norm_rope_module(
            dtype=torch.bfloat16,
            head_dim=HEAD_DIM,
            rope_dim=ROPE_DIM,
            page_size=page_size,
            bf16_store=False,
        )
        modules[page_size] = module
        producer_results.append(
            run_producer_page(
                module,
                page_size,
                ratio,
                gamma.cuda(),
                gamma_reference,
                output_dir,
            )
        )

    probe_module = load_jit(
        "dsv4_fp8_pack_probe_regression",
        cuda_files=[str(args.probe_source.resolve())],
        cuda_wrappers=[("run", "fp8_pack_probe::run")],
    )
    pack_result = run_pack_probe(probe_module, output_dir)

    build_evidence = []
    for page_size in (64, 2):
        matching_paths = [
            pathlib.Path(path)
            for path in loaded_private_jit_paths(cache_root)
            if f"_512_64_{page_size}_" in pathlib.Path(path).name
        ]
        if len(matching_paths) != 1:
            raise RuntimeError(
                f"Expected one loaded producer .so for page size {page_size}, got {matching_paths}"
            )
        loaded_so = matching_paths[0]
        build_dir = loaded_so.parent
        generated_source = build_dir / "cuda.cu"
        build_ninja = build_dir / "build.ninja"
        generated_text = generated_source.read_text()
        expected_symbol = (
            f"FusedNormRopeKernel<bf16_t, 512, 64, {page_size}, false, 0, false>::forward"
        )
        if expected_symbol not in generated_text:
            raise RuntimeError(
                f"Generated wrapper did not select {expected_symbol}: {generated_text}"
            )
        destination = output_dir / f"producer_p{page_size}"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build_ninja, destination / "build.ninja")
        shutil.copy2(generated_source, destination / "cuda.cu")
        build_evidence.append(
            {
                "page_size": page_size,
                "selected_symbol": expected_symbol,
                "generated_source": str(generated_source),
                "generated_source_sha256": sha256_file(generated_source),
                "build_ninja": str(build_ninja),
                "build_ninja_sha256": sha256_file(build_ninja),
                "loaded_so": str(loaded_so),
                "loaded_so_sha256": sha256_file(loaded_so),
                "offload_gfx942": "--offload-arch=gfx942:sramecc+:xnack-" in build_ninja.read_text(),
                "fnuz_macro": "-DHIP_FP8_TYPE_FNUZ=1" in build_ninja.read_text(),
                "use_rocm": "-DUSE_ROCM" in build_ninja.read_text(),
                "hardware_cvt_macro": "SGL_ROCM_FP8_HW_CVT" in build_ninja.read_text(),
            }
        )

    pack_so_paths = [
        pathlib.Path(path)
        for path in loaded_private_jit_paths(cache_root)
        if "dsv4_fp8_pack_probe_regression" in pathlib.Path(path).name
    ]
    if len(pack_so_paths) != 1:
        raise RuntimeError(f"Expected one loaded pack probe .so, got {pack_so_paths}")
    pack_build_dir = pack_so_paths[0].parent
    shutil.copy2(pack_build_dir / "build.ninja", output_dir / "pack_build.ninja")
    shutil.copy2(pack_build_dir / "cuda.cu", output_dir / "pack_cuda.cu")

    report = {
        "arm": args.arm,
        "runtime_source_root": str(source_root),
        "runtime_sglang_file": str(sglang.__file__),
        "git_head": subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_status": subprocess.run(
            ["git", "-C", str(source_root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "device_count": torch.cuda.device_count(),
        "device_name": device_properties.name,
        "gcn_arch_name": device_properties.gcnArchName,
        "compiler_identity": compiler_identity(),
        "jit_cache_root": str(cache_root),
        "fused_source": str(fused_source),
        "fused_source_sha256": sha256_file(fused_source),
        "fp8_header": str(fp8_header),
        "fp8_header_sha256": sha256_file(fp8_header),
        "probe_source": str(args.probe_source.resolve()),
        "probe_source_sha256": sha256_file(args.probe_source.resolve()),
        "producer": producer_results,
        "producer_build_evidence": build_evidence,
        "pack": pack_result,
        "pack_loaded_so": str(pack_so_paths[0]),
        "pack_loaded_so_sha256": sha256_file(pack_so_paths[0]),
    }
    write_json(output_dir / "result.json", report)

    producer_pass = all(
        result["full_cache_equal"]
        and result["payload_equal"]
        and result["scale_equal"]
        and result["rope_equal"]
        and result["decoded_nope_equal"]
        for result in producer_results
    )
    pack_pass = (
        pack_result["all_equal"]
        and pack_result["policy_max"] == 224.0
        and pack_result["fnuz_macro"] == 1.0
        and pack_result["hardware_cvt"] == 0.0
    )
    print(json.dumps({"producer_pass": producer_pass, "pack_pass": pack_pass}, indent=2))
    if args.arm == "candidate" and not (producer_pass and pack_pass):
        raise SystemExit(1)
    if args.arm == "control" and producer_pass:
        raise SystemExit(
            "Control unexpectedly passed the strict full-cache producer comparison."
        )


if __name__ == "__main__":
    main()
