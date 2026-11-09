"""Shared test fixtures for HydroArray tests."""
import sys
from pathlib import Path

import numpy as np
import pytest


# Make the project importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_grid_nodes():
    """Create a small linear chain of grid nodes (head -> mid -> outlet)."""
    from HydroArray.domain.models import GridNode

    nodes = [
        GridNode(node_id=0, x=0.0, y=0.0, area=1.0, elevation=100.0,
                 downstream_id=1, flow_accumulation=1, stream_order=1),
        GridNode(node_id=1, x=1.0, y=0.0, area=1.0, elevation=90.0,
                 downstream_id=2, flow_accumulation=2, stream_order=2),
        GridNode(node_id=2, x=2.0, y=0.0, area=1.0, elevation=80.0,
                 downstream_id=-1, flow_accumulation=3, stream_order=3),
    ]
    return nodes


@pytest.fixture
def sample_precip(rng):
    """Synthetic precipitation: a few storm events over 30 days."""
    p = np.zeros(30)
    # Event 1: 5-day storm
    p[5:10] = np.array([5.0, 15.0, 30.0, 20.0, 5.0])
    # Event 2: 3-day storm
    p[20:23] = np.array([10.0, 25.0, 8.0])
    return p


@pytest.fixture
def sample_pet():
    """Synthetic PET: constant 3 mm/day."""
    return np.full(30, 3.0)
