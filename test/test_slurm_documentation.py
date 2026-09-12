import ast
import re
import subprocess
import sys
from pathlib import Path


DOC = Path(__file__).parents[1] / "docs/docs/references/multi_node_deployment/multi_node.mdx"


def _bash_blocks():
    text = DOC.read_text()
    return re.findall(r"```bash(?: Command)?\n(.*?)```", text, re.DOTALL)


def test_slurm_bash_examples_have_valid_syntax(tmp_path):
    blocks = _bash_blocks()
    assert len(blocks) >= 5
    for index, block in enumerate(blocks):
        script = tmp_path / f"example-{index}.sh"
        script.write_text(block.replace("ENV_FOLDER", "/tmp/sglang-env"))
        subprocess.run(["bash", "-n", script], check=True)


def test_multinode_rank_is_expanded_by_each_srun_task():
    text = DOC.read_text()
    assert '--node-rank "$SLURM_PROCID"' in text
    assert '--node-rank "$SLURM_NODEID"' not in text
    assert 'bash -c \'\n        exec python3 -m sglang.launch_server' in text
    assert "node%n.out" in text


def test_documented_server_flags_match_current_cli():
    help_result = subprocess.run(
        [sys.executable, "-m", "sglang.launch_server", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    help_text = help_result.stdout
    for flag in (
        "--model-path",
        "--host",
        "--port",
        "--tp-size",
        "--dist-init-addr",
        "--nnodes",
        "--node-rank",
    ):
        assert flag in help_text

    assert not re.search(r"(?<![-\w])--tp(?:\s|=)", DOC.read_text())


def test_offline_example_has_spawn_safe_entrypoint():
    text = DOC.read_text()
    match = re.search(r"cat >\"\$script_path\" <<'PY'\n(.*?)\nPY", text, re.DOTALL)
    assert match is not None
    tree = ast.parse(match.group(1))
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(isinstance(value, ast.Constant) and value.value == "__main__" for value in node.test.comparators)
    ]
    assert guards


def test_container_example_keeps_rank_expansion_inside_task_shell():
    text = DOC.read_text()
    container = text.split("### Apptainer or Singularity", 1)[1]
    assert "apptainer exec --nv" in container
    assert '--node-rank "$SLURM_PROCID"' in container
    assert "--rocm" in container
