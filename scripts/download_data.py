"""Download the raw snapshots (if absent) and verify them against the frozen manifest.

    python scripts/download_data.py

The manifest (data/raw/SNAPSHOT_MANIFEST.json) pins SHA-256 checksums. If a re-download no
longer matches (the publisher replaced the file), the script stops: analyses must then be
re-frozen deliberately, never silently updated.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sustainome import config as C  # noqa: E402
from sustainome.io import sha256  # noqa: E402


def main() -> int:
    manifest = json.loads(C.MANIFEST_FILE.read_text(encoding="utf-8")) if C.MANIFEST_FILE.exists() else {"files": {}}
    ok = True
    for key, meta in C.SOURCES.items():
        path: Path = meta["path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {key} ...")
            req = urllib.request.Request(meta["url"], headers={"User-Agent": "sustainome-research-script"})
            with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as fh:
                fh.write(r.read())
        expected = manifest["files"].get(key, {}).get("sha256")
        actual = sha256(path)
        status = "OK" if expected in (None, actual) else "CHECKSUM MISMATCH"
        ok &= status == "OK"
        print(f"{key:22s} {path.stat().st_size:>10,d} bytes  sha256 {actual[:16]}…  {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
