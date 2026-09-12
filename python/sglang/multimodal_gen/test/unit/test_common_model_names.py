import ast
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

TEST_ROOT = Path(__file__).resolve().parents[1]
TEST_UTILS = TEST_ROOT / "test_utils.py"
LITERAL_VECTOR_PATHS = {
    Path(__file__).resolve(),
    TEST_ROOT / "unit" / "test_server_args.py",
}


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


def _find_repeated_uncentralized_model_names(
    paths: list[Path], common_names: set[str]
) -> list[str]:
    """Find repeated literals used in model-selection positions.

    This complements the exact shared-constant check: a newly repeated model ID
    cannot evade the policy merely because nobody added its constant yet.
    """
    occurrences: dict[str, list[str]] = defaultdict(list)
    model_resolvers = {"get_model_info", "hf_cached_model", "use_modelscope"}

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            values = []
            if isinstance(node, ast.Call):
                values.extend(
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg in {"model_id", "model_path"}
                )
                function_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else None
                )
                if function_name in model_resolvers and node.args:
                    values.append(node.args[0])

            for value in values:
                if (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and "/" in value.value
                    and value.value not in common_names
                ):
                    location = f"{path}:{value.lineno}"
                    if location not in occurrences[value.value]:
                        occurrences[value.value].append(location)

    return [
        f"{model_name}: {', '.join(locations)}"
        for model_name, locations in sorted(occurrences.items())
        if len(locations) > 1
    ]


def test_runnable_test_entries_use_common_model_name_constants():
    """Keep shared model IDs out of executable diffusion test entries."""
    paths = [
        path
        for path in sorted(TEST_ROOT.rglob("*.py"))
        if path != TEST_UTILS and path not in LITERAL_VECTOR_PATHS
    ]
    duplicates = _find_hard_coded_model_names(paths, _common_model_names())
    uncentralized = _find_repeated_uncentralized_model_names(
        [path for path in paths if TEST_ROOT / "unit" not in path.parents],
        _common_model_names(),
    )

    assert not duplicates, (
        "Runnable diffusion tests must import shared model IDs from test_utils.py; "
        f"found hard-coded duplicates: {', '.join(duplicates)}"
    )
    assert not uncentralized, (
        "Repeated model IDs must first be defined in test_utils.py; "
        f"found uncentralized values: {'; '.join(uncentralized)}"
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


def test_model_name_guard_rejects_repeated_uncentralized_ids(tmp_path: Path):
    paths = []
    for index in range(2):
        path = tmp_path / f"test_entry_{index}.py"
        path.write_text(
            'case = DiffusionServerArgs(model_path="example-org/new-model")\n',
            encoding="utf-8",
        )
        paths.append(path)

    assert _find_repeated_uncentralized_model_names(paths, _common_model_names()) == [
        f"example-org/new-model: {paths[0]}:1, {paths[1]}:1"
    ]
