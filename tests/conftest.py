"""Shared pytest fixtures."""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator so tests are reproducible."""
    return np.random.default_rng(12345)
