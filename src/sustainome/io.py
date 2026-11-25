"""Raw-data access and provenance.

Raw files are immutable snapshots: these functions only read them. Every value is read
as text so that the explicit parser in :mod:`sustainome.clean` sees exactly what the
publisher wrote (decimal commas, '-', 'traces', '< x').
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

from . import config as C


def norm_text(s: object) -> str:
    """Accent-, case- and whitespace-insensitive key used to match published headers."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return " ".join(s.split()).casefold()


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# CIQUAL
# ---------------------------------------------------------------------------
def load_ciqual_raw(path: Path = C.CIQUAL_FILE) -> pd.DataFrame:
    """Return the CIQUAL composition sheet with every cell as a string (or NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # openpyxl header/footer warning is cosmetic
        df = pd.read_excel(path, sheet_name=C.SOURCES["ciqual_2025"]["sheet"], dtype=str)
    return df


# ---------------------------------------------------------------------------
# AGRIBALYSE 3.2
# ---------------------------------------------------------------------------
AGB_ID_COLUMNS = {  # normalised header prefix -> clean name (row 3 of the 'Synthese' sheet)
    "code agb": "agb_code",
    "code ciqual": "ciqual_code_raw",
    "groupe d'aliment": "agb_group_fr",
    "sous-groupe d'aliment": "agb_subgroup_fr",
    "nom du produit en francais": "agb_name_fr",
    "lci name": "agb_lci_name",
    "code saison": "season_code",
    "code avion": "plane_code",
    "livraison": "delivery",
    "approche emballage": "packaging_approach",
    "preparation": "preparation",
    "dqr": "dqr",
}


def load_agribalyse_raw(path: Path = C.AGB_FILE) -> pd.DataFrame:
    """Read the 3.2 'Synthese' sheet with explicit header mapping and row font colours.

    The publisher marks rows with known errors in red font (see the workbook 'Notice'):
    uncorrected packaging rows and the over-estimated quinoa rows. Font colour is
    therefore data and is preserved in ``row_font_red``.
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[C.SOURCES["agribalyse_3.2"]["sheet"]]
    head = list(ws.iter_rows(min_row=1, max_row=3, values_only=True))
    indicator_row, header_row = list(head[1]), list(head[2])

    colmap: dict[int, str] = {}
    for j, h in enumerate(header_row):
        key = norm_text(h)
        for prefix, clean in AGB_ID_COLUMNS.items():
            if key.startswith(prefix) and clean not in colmap.values():
                colmap[j] = clean
                break
    env_lookup = {norm_text(fr): k for k, fr, _, _ in C.ENV}
    for j, ind in enumerate(indicator_row):
        k = env_lookup.get(norm_text(ind))
        if k is not None:
            colmap[j] = f"{k}_per_kg"
    missing = set(AGB_ID_COLUMNS.values()) | {f"{k}_per_kg" for k in C.ENV_KEYS}
    missing -= set(colmap.values())
    if missing:
        raise ValueError(f"AGRIBALYSE columns not found: {sorted(missing)}")

    records = []
    for r, row in enumerate(ws.iter_rows(min_row=4), start=4):
        if len(row) < 2 or (row[0].value is None and row[1].value is None):
            continue
        rec = {clean: row[j].value for j, clean in colmap.items()}
        font = getattr(row[4], "font", None)
        color = None if font is None else font.color
        rec["row_font_red"] = bool(color is not None and color.type == "rgb" and str(color.rgb).upper().endswith("FF0000"))
        rec["source_row"] = r
        records.append(rec)
    wb.close()
    df = pd.DataFrame.from_records(records)
    for k in C.ENV_KEYS:
        df[f"{k}_per_kg"] = pd.to_numeric(df[f"{k}_per_kg"], errors="raise")
    df["dqr"] = pd.to_numeric(df["dqr"], errors="raise")
    df["agb_code"] = df["agb_code"].astype(str).str.strip()
    return df


def load_agribalyse_notice(path: Path = C.AGB_FILE) -> list[str]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True)
    out = []
    for sheet in ("Notice", "Environnement de calcul"):
        for row in wb[sheet].iter_rows(values_only=True):
            out.extend(str(v) for v in row if v is not None)
    return out


def load_agribalyse31_api(path: Path = C.AGB31_API_FILE) -> pd.DataFrame:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    df = pd.DataFrame(payload["results"])
    rename = {}
    for col in df.columns:
        key = norm_text(col.replace("_", " "))
        if key == "code ciqual":
            rename[col] = "ciqual_code_raw"
        elif key == "code agb":
            rename[col] = "agb_code"
        elif key == "lci name":
            rename[col] = "agb_lci_name"
        elif key == "dqr":
            rename[col] = "dqr"
        else:
            for k, fr, _, _ in C.ENV:
                if key == norm_text(fr) or (k == "ef_score" and key == "score unique ef"):
                    rename[col] = f"{k}_per_kg"
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Snapshot manifest
# ---------------------------------------------------------------------------
def build_manifest(access_date: str | None = None) -> dict:
    """Describe every raw snapshot (size, checksum, version, licence, sheet)."""
    existing = read_manifest() if C.MANIFEST_FILE.exists() else {}
    entries = {}
    for key, meta in C.SOURCES.items():
        p: Path = meta["path"]
        prev = existing.get("files", {}).get(key, {})
        entries[key] = {
            "organization": meta["organization"],
            "title": meta["title"],
            "version": meta["version"],
            "download_url": meta["url"],
            "record": meta["record"],
            "license": meta["license"],
            "sheet_or_table": meta["sheet"],
            "local_path": p.relative_to(C.ROOT).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
            "access_date": prev.get("access_date", access_date or date.today().isoformat()),
            "local_transformation": "none (immutable snapshot; renamed only)",
        }
    return {"project": "Sustainome", "files": entries}


def write_manifest(manifest: dict) -> None:
    C.MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def read_manifest() -> dict:
    return json.loads(C.MANIFEST_FILE.read_text(encoding="utf-8"))


def verify_manifest() -> dict[str, bool]:
    """Re-hash raw files and compare with the frozen manifest."""
    man = read_manifest()
    return {k: sha256(C.ROOT / v["local_path"]) == v["sha256"] for k, v in man["files"].items()}
