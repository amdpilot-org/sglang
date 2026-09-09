#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_tree(root_pid: int) -> list[int]:
    parents = {}
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            tail = stat[stat.rfind(")") + 2 :].split()
            parents[int(entry.name)] = int(tail[1])
            pids.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid in pids:
            if pid in tree:
                continue
            if parents.get(pid) in tree:
                tree.add(pid)
                changed = True
    return sorted(tree)


def read_text(path: Path, limit: int = 2_000_000) -> str | None:
    try:
        if path.stat().st_size > limit:
            return None
        return path.read_text(errors="replace")
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache_root = Path(args.cache_root).resolve()
    pid = int(Path(f"/job/evidence/{args.arm}/server.pid").read_text().strip())
    pids = process_tree(pid)
    processes = []
    loaded_paths = set()
    for current_pid in pids:
        try:
            cmdline = Path(f"/proc/{current_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmdline = None
        mapped = []
        try:
            for line in Path(f"/proc/{current_pid}/maps").read_text().splitlines():
                if " " not in line:
                    continue
                path_text = line.rsplit(" ", 1)[-1]
                candidate = Path(path_text)
                try:
                    resolved = candidate.resolve()
                except OSError:
                    resolved = candidate
                if cache_root in resolved.parents or resolved == cache_root:
                    mapped.append({"raw_path": path_text, "resolved_path": str(resolved)})
                    loaded_paths.add(resolved)
        except OSError:
            pass
        processes.append({"pid": current_pid, "cmdline": cmdline, "private_cache_maps": mapped})

    cache_files = []
    for path in cache_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
            record = {
                "path": str(path),
                "size": stat.st_size,
                "sha256": sha256(path),
                "loaded": path.resolve() in loaded_paths,
            }
            if path.name in {"build.ninja", "compile_commands.json"} or path.suffix in {".cu", ".cpp", ".hip", ".log", ".json"}:
                record["text"] = read_text(path)
            cache_files.append(record)
        except OSError:
            continue

    loaded = []
    for path in sorted(loaded_paths):
        try:
            loaded.append({
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            })
        except OSError:
            loaded.append({"path": str(path), "error": "unreadable"})

    producer_matches = [
        record for record in cache_files
        if any(token in record["path"] for token in (
            "main_k_norm_rope_flashmla",
            "main_q_norm_rope",
            "fused_norm_rope_v2",
            "compress_4_v2",
            "compress_128_online_v2",
            "c4_v2",
            "c128_online_v2",
        ))
    ]

    result = {
        "arm": args.arm,
        "cache_root": str(cache_root),
        "root_pid": pid,
        "process_tree": processes,
        "loaded_private_native_paths": loaded,
        "cache_files": cache_files,
        "producer_matches": producer_matches,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "arm": args.arm,
        "root_pid": pid,
        "process_count": len(processes),
        "loaded_count": len(loaded),
        "cache_file_count": len(cache_files),
        "producer_match_count": len(producer_matches),
    }, indent=2))


if __name__ == "__main__":
    main()
