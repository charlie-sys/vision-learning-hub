"""Validate notebook schema, clean outputs, Python syntax and launch mapping."""
from pathlib import Path
import ast
import json
import nbformat
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
site = json.loads((ROOT / "data/site.json").read_text(encoding="utf-8"))
files = sorted((ROOT / "notebooks").glob("*.ipynb"))
assert len(files) == 7
code_cells = 0
for path in files:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells), path.name
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{path.name}:cell{index}")
            assert cell.execution_count is None and not cell.outputs, f"Unclean output: {path.name}"
            code_cells += 1
    generated = ROOT / "docs/notebooks" / path.name
    assert generated.read_bytes() == path.read_bytes(), f"Stale notebook: {path.name}"
page = BeautifulSoup((ROOT / "docs/playground.html").read_text(encoding="utf-8"), "html.parser")
assert len(page.select('.python-runner')) == 8
for path in files:
    url = f'https://colab.research.google.com/github/{site["repository"]}/blob/{site["branch"]}/notebooks/{path.name}'
    assert page.find('a', href=url), path.name
    assert page.find('a', href=f'notebooks/{path.name}', download=path.name), path.name
examples = json.loads((ROOT / "data/runner_examples.json").read_text(encoding="utf-8"))
for example in examples:
    ast.parse(example['code'])
    node = page.select_one(f'[data-example="{example["id"]}"] textarea')
    assert node.get_text() == example['code'], f'Altered editable code: {example["id"]}'
print(f"{len(files)} clean notebooks, {code_cells} code cells, launch/download mapping and 8 editable examples: OK")
