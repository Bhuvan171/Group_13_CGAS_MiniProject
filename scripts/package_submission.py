"""Create a complete, self-contained project archive next to the project folder.

    python scripts/package_submission.py

Includes code, declared inputs, raw snapshots, processed data, executed notebooks,
figures, tables, the report, the presentation (PDF/PPTX/source) and the reference
PDFs, so the project can be moved to another machine and rebuilt offline.
Excludes only caches and editor/scratch files.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "Sustainome_Group13_complete"
INCLUDE = ["README.md", "CLAUDE.md", "requirements.txt", "run_all.py", "verify_submission.py", ".gitignore",
           "config", "data", "docs", "notebooks", "presentation", "references", "reports", "scripts", "src", "tests"]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def main() -> int:
    out = ROOT.parent / f"{NAME}.zip"
    n = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for item in INCLUDE:
            p = ROOT / item
            if not p.exists():
                continue
            files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
            for f in files:
                if EXCLUDE_PARTS & set(f.parts) or f.suffix == ".pyc":
                    continue
                z.write(f, Path(NAME) / f.relative_to(ROOT))
                n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
