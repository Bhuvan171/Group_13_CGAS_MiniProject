"""Reproduce every Sustainome deliverable from the frozen raw snapshots.

    python run_all.py              # verify data, run tests, execute notebooks 00-05, build report, verify
    python run_all.py --skip-tests

Steps
  1. scripts/download_data.py   - fetch raw files if absent; verify SHA-256 against the manifest
  2. pytest                     - unit + integration tests (parser, Pareto, graph rules, optimiser)
  3. scripts/build_notebooks.py - execute notebooks 00-05 in order (writes data/processed, reports/)
  4. scripts/build_report.py    - assemble reports/Sustainome_Report.pdf from figures and tables
  5. verify_submission.py       - acceptance checks -> reports/verification.json
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(name: str, cmd: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    t = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"--- {name}: done in {time.time() - t:.0f} s", flush=True)


def main(argv: list[str]) -> None:
    py = sys.executable
    step("1/5 data snapshots", [py, "scripts/download_data.py"])
    if "--skip-tests" not in argv:
        step("2/5 tests", [py, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    step("3/5 notebooks", [py, "scripts/build_notebooks.py"])
    step("4/5 report", [py, "scripts/build_report.py"])
    step("5/5 verification", [py, "verify_submission.py"])


if __name__ == "__main__":
    main(sys.argv[1:])
