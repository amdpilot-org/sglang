#!/usr/bin/env python3
import argparse
import hashlib
import json
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
            if pid not in tree and parents.get(pid) in tree:
                tree.add(pid)
                changed = True
    return sorted(tree)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pid = int(Path(f"/job/evidence/{args.arm}/server.pid").read_text().strip())
    paths = set()
    processes = []
    for current_pid in process_tree(pid):
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
                if path_text.startswith("/sgl-workspace/aiter/") and path_text.endswith(".so"):
                    path = Path(path_text).resolve()
                    mapped.append(str(path))
                    paths.add(path)
        except OSError:
            pass
        processes.append({"pid": current_pid, "cmdline": cmdline, "aiter_maps": sorted(set(mapped))})

    libraries = []
    for path in sorted(paths):
        try:
            libraries.append({
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            })
        except OSError:
            libraries.append({"path": str(path), "error": "unreadable"})

    result = {
        "arm": args.arm,
        "root_pid": pid,
        "processes": processes,
        "shared_aiter_libraries": libraries,
        "provenance": "Immutable image-installed AITER libraries shared across arms; not consumers of the changed SGLang fp8_utils.cuh.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"arm": args.arm, "library_count": len(libraries)}, indent=2))


if __name__ == "__main__":
    main()
