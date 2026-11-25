"""Convert percent-format notebook sources to .ipynb and execute them in order.

    python scripts/build_notebooks.py            # build + execute all
    python scripts/build_notebooks.py 02 03      # only notebooks whose name starts with 02 / 03
    python scripts/build_notebooks.py --no-exec  # convert only

Sources live in scripts/notebook_sources/*.py; cells are separated by '# %%' lines and
markdown cells by '# %% [markdown]' (their lines are '# '-prefixed).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "notebook_sources"
OUT = ROOT / "notebooks"


def parse(text: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            return
        if kind == "md":
            lines = [ln[2:] if ln.startswith("# ") else ln[1:] if ln.startswith("#") else ln for ln in body.splitlines()]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "md" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    nb.cells = cells
    return nb


def build(src: Path, execute: bool = True, timeout: int = 3600) -> Path:
    nb = parse(src.read_text(encoding="utf-8"))
    out = OUT / (src.stem + ".ipynb")
    if execute:
        t = time.time()
        NotebookClient(nb, timeout=timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(OUT)}}).execute()
        print(f"  executed {out.name} in {time.time() - t:.0f} s")
    nbformat.write(nb, out)
    return out


def main(argv: list[str]) -> None:
    execute = "--no-exec" not in argv
    prefixes = [a for a in argv if not a.startswith("--")]
    OUT.mkdir(exist_ok=True)
    for src in sorted(SRC.glob("*.py")):
        if prefixes and not any(src.stem.startswith(p) for p in prefixes):
            continue
        print(f"building {src.stem}")
        build(src, execute)


if __name__ == "__main__":
    main(sys.argv[1:])
