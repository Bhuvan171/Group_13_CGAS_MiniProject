import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def matrix_build():
    """Full D1 build from the frozen raw snapshots (integration fixture, ~15 s)."""
    from sustainome import matrix

    return matrix.build_matrix()


@pytest.fixture(scope="session")
def m(matrix_build):
    return matrix_build.matrix
