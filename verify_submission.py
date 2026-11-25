"""Check that every deliverable artefact exists and every acceptance criterion is met.

    python verify_submission.py      -> prints a summary and writes reports/verification.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import nbformat  # noqa: E402
import pandas as pd  # noqa: E402

from sustainome import config as C, io  # noqa: E402

NOTEBOOKS = ["00_data_audit", "01_sustainable_food_matrix", "02_tradeoff_analysis",
             "03_substitution_network", "04_basket_optimization", "05_robustness_counterexamples"]
MIN_FIGURES = {1: 4, 2: 7, 3: 6, 4: 6, 5: 6}


def main() -> int:
    res: dict[str, object] = {}
    res["raw_checksums_verified"] = all(io.verify_manifest().values())
    m = pd.read_csv(C.MATRIX_FILE, dtype={"ciqual_code": str})
    stages = pd.read_csv(C.TABLES / "D1_stage_counts.csv").set_index("stage")["count"]
    res["matrix_rows"] = len(m)
    res["matrix_rows_match_join"] = int(stages["Exact matches (inner join on normalised CIQUAL code)"]) == len(m)

    nb_status = {}
    for name in NOTEBOOKS:
        p = ROOT / "notebooks" / f"{name}.ipynb"
        if not p.exists():
            nb_status[name] = "missing"
            continue
        nb = nbformat.read(p, as_version=4)
        code = [c for c in nb.cells if c.cell_type == "code"]
        errors = sum(o.get("output_type") == "error" for c in code for o in c.get("outputs", []))
        unexecuted = sum(c.get("execution_count") is None for c in code)
        nb_status[name] = "ok" if errors == 0 and unexecuted == 0 else f"errors={errors}, unexecuted={unexecuted}"
    res["notebooks"] = nb_status

    acc = {}
    for d in range(1, 6):
        t = C.TABLES / f"D{d}_acceptance.csv"
        if t.exists():
            a = pd.read_csv(t)
            acc[f"D{d}"] = {"criteria": len(a), "met": int(a["met"].astype(bool).sum())}
        else:
            acc[f"D{d}"] = "missing"
    res["acceptance"] = acc

    figs = sorted(p.name for p in C.FIGURES.glob("D*_*.png"))
    res["figures"] = {f"D{d}": sum(f.startswith(f"D{d}_") for f in figs) for d in range(1, 6)}
    res["figures_meet_minimum"] = all(res["figures"][f"D{d}"] >= n for d, n in MIN_FIGURES.items())
    res["tables"] = len(list(C.TABLES.glob("D*_*.csv")))

    reg = C.TABLES / "D5_claims_register.csv"
    if reg.exists():
        r = pd.read_csv(reg)
        res["claims"] = r["label"].value_counts().to_dict()

    report = C.REPORTS / "Sustainome_Report.pdf"
    if report.exists():
        from pypdf import PdfReader
        res["report_pages"] = len(PdfReader(str(report)).pages)
    else:
        res["report_pages"] = 0

    ok = (res["raw_checksums_verified"] and res["matrix_rows_match_join"]
          and all(v == "ok" for v in nb_status.values())
          and all(isinstance(v, dict) and v["criteria"] == v["met"] for v in acc.values())
          and res["figures_meet_minimum"] and res["report_pages"] > 0)
    res["ALL_CHECKS_PASSED"] = bool(ok)
    (C.REPORTS / "verification.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
