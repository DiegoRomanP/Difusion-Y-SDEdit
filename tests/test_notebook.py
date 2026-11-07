import ast
from pathlib import Path

import nbformat


def test_notebook_is_valid_unexecuted_and_plain_python():
    notebook = nbformat.read(Path("anime_diffusion.ipynb"), as_version=4)
    nbformat.validate(notebook)
    assert len(notebook.cells) >= 35
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
            ast.parse(cell.source)


def test_notebook_expensive_phases_are_disabled_by_default():
    notebook = nbformat.read("anime_diffusion.ipynb", as_version=4)
    flags = {}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("RUN_"):
                        assert isinstance(node.value, ast.Constant)
                        flags[target.id] = node.value.value
    assert {"RUN_DOWNLOAD", "RUN_TRAINING", "RUN_RESULTS", "RUN_ABLATION"} <= flags.keys()
    assert all(value is False for value in flags.values())
