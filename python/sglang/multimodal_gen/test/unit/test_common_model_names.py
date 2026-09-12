import ast
from pathlib import Path
from textwrap import dedent

TEST_ROOT = Path(__file__).resolve().parents[1]
TEST_UTILS = TEST_ROOT / "test_utils.py"


def _common_model_names() -> set[str]:
    tree = ast.parse(TEST_UTILS.read_text(encoding="utf-8"), filename=str(TEST_UTILS))
    names = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id.startswith("DEFAULT_")
            and target.id.endswith("_MODEL_NAME_FOR_TEST")
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            names.add(node.value.value)
    return names


def _find_hard_coded_model_names(
    paths: list[Path], common_names: set[str]
) -> list[str]:
    duplicates = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in common_names
                and node not in docstrings
            ):
                duplicates.append(f"{path}:{node.lineno}")
    return duplicates


def test_runnable_test_entries_use_common_model_name_constants():
    """Keep shared model IDs out of executable diffusion test entries."""
    paths = [
        path
        for path in sorted(TEST_ROOT.rglob("*.py"))
        if path != TEST_UTILS and TEST_ROOT / "unit" not in path.parents
    ]
    duplicates = _find_hard_coded_model_names(paths, _common_model_names())

    assert not duplicates, (
        "Runnable diffusion tests must import shared model IDs from test_utils.py; "
        f"found hard-coded duplicates: {', '.join(duplicates)}"
    )


def test_model_name_guard_rejects_code_but_ignores_docstrings(tmp_path: Path):
    source = tmp_path / "test_entry.py"
    source.write_text(
        dedent(
            '''
            """Qwen/Qwen-Image is documented here and is not executable."""

            MODEL = "Qwen/Qwen-Image"
            '''
        ),
        encoding="utf-8",
    )

    assert _find_hard_coded_model_names([source], _common_model_names()) == [
        f"{source}:4"
    ]
