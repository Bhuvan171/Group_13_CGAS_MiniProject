# Data directory

Raw data are intentionally not committed in this context bundle. Download instructions and version rules are in `docs/DATA_SOURCES.md`.

Expected structure:

```text
data/
  raw/
    ciqual/2025/
    agribalyse/3.2/
  interim/
  processed/
```

Rules:

- Raw files are immutable snapshots.
- Record access date, URL, version, byte size, and SHA-256 checksum.
- All processed files must be reproducible from code.
- Do not commit large raw or processed data files unless the team explicitly decides to use Git LFS or another data-versioning mechanism.
- Never mix AGRIBALYSE 3.1 and 3.2 values in one table without explicit version labels.

