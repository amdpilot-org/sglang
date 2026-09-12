#!/usr/bin/env python3
"""Audit repository-owned GitHub automation for sunset Gemini Code Assist references."""

from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB_ROOT = REPO_ROOT / ".github"
SUNSET_REFERENCES = (
    re.compile(r"gemini-code-assist", re.IGNORECASE),
    re.compile(r"gemini\s+code\s+assist", re.IGNORECASE),
)


def find_sunset_references(root: Path) -> list[str]:
    matches = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in SUNSET_REFERENCES):
            matches.append(str(path.relative_to(root)))
    return matches


def check_boundary_cases() -> None:
    assert any(pattern.search("uses: google/gemini-code-assist@v1") for pattern in SUNSET_REFERENCES)
    assert any(pattern.search("Gemini Code Assist review") for pattern in SUNSET_REFERENCES)
    assert not any(pattern.search("Route requests to a Gemini model") for pattern in SUNSET_REFERENCES)


def main() -> int:
    check_boundary_cases()
    matches = find_sunset_references(GITHUB_ROOT)
    if matches:
        print("sunset Gemini Code Assist references found in .github:")
        print("\n".join(matches))
        return 1
    print("PASS: no sunset Gemini Code Assist references in tracked .github files")
    print("PASS: exact-slug, display-name, and unrelated-Gemini boundary cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
