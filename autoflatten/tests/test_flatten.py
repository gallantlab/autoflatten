"""
Tests for the flatten module (pyflatten algorithm).
"""

import json
import os
import tempfile

import numpy as np
import pytest

from autoflatten.flatten.config import (
    FlattenConfig,
    KRingConfig,
    PhaseConfig,
    ConvergenceConfig,
    LineSearchConfig,
    NegativeAreaRemovalConfig,
    SpringSmoothingConfig,
    get_kring_cache_filename,
)
from autoflatten.flatten import count_flipped_triangles


class TestKRingConfig:
    """Tests for KRingConfig dataclass."""

    def test_default_values(self):
        """Test default values for KRingConfig."""
        config = KRingConfig()
        assert config.k_ring == 20
        assert config.n_neighbors_per_ring == 30

    def test_custom_values(self):
        """Test custom values for KRingConfig."""
        config = KRingConfig(k_ring=15, n_neighbors_per_ring=25)
        assert config.k_ring == 15
        assert config.n_neighbors_per_ring == 25


class TestConvergenceConfig:
    """Tests for ConvergenceConfig dataclass."""

    def test_default_values(self):
        """Test default values for ConvergenceConfig."""
        config = ConvergenceConfig()
        assert config.base_tol == 0.2
        assert config.max_small == 50000
        assert config.total_small == 15000

    def test_custom_values(self):
        """Test custom values for ConvergenceConfig."""
        config = ConvergenceConfig(base_tol=0.5, max_small=10000, total_small=5000)
        assert config.base_tol == 0.5
        assert config.max_small == 10000
        assert config.total_small == 5000


class TestLineSearchConfig:
    """Tests for LineSearchConfig dataclass."""

    def test_default_values(self):
        """Test default values for LineSearchConfig."""
        config = LineSearchConfig()
        assert config.n_coarse_steps == 15
        assert config.max_mm == 1000.0
        assert config.min_mm == 0.001


class TestPhaseConfig:
    """Tests for PhaseConfig dataclass."""

    def test_default_values(self):
        """Test default values for PhaseConfig (requires name and area_ratio)."""
        config = PhaseConfig(name="test", area_ratio=1.0)
        assert config.name == "test"
        assert config.area_ratio == 1.0
        assert config.enabled is True
        assert config.iters_per_level == 200
        assert config.base_tol is None
        assert len(config.smoothing_schedule) == 7

    def test_custom_phase(self):
        """Test custom phase configuration."""
        config = PhaseConfig(
            name="test_phase",
            area_ratio=10.0,
            enabled=True,
            iters_per_level=100,
            base_tol=0.5,
        )
        assert config.name == "test_phase"
        assert config.area_ratio == 10.0
        assert config.iters_per_level == 100
        assert config.base_tol == 0.5


class TestNegativeAreaRemovalConfig:
    """Tests for NegativeAreaRemovalConfig dataclass."""

    def test_default_values(self):
        """Test default values for NegativeAreaRemovalConfig."""
        config = NegativeAreaRemovalConfig()
        assert config.enabled is True
        assert config.base_averages == 256
        assert config.min_area_pct == 0.5
        assert config.max_passes == 5
        assert config.iters_per_level == 200
        assert config.base_tol == 0.5


class TestSpringSmoothing:
    """Tests for SpringSmoothingConfig dataclass."""

    def test_default_values(self):
        """Test default values for SpringSmoothingConfig."""
        config = SpringSmoothingConfig()
        assert config.enabled is True
        assert config.n_iterations == 5
        assert config.dt == 0.5
        assert config.max_step_mm == 1.0


class TestFlattenConfig:
    """Tests for FlattenConfig dataclass."""

    def test_default_values(self):
        """Test default values for FlattenConfig."""
        config = FlattenConfig()
        assert isinstance(config.kring, KRingConfig)
        assert isinstance(config.negative_area_removal, NegativeAreaRemovalConfig)
        assert isinstance(config.spring_smoothing, SpringSmoothingConfig)
        assert config.verbose is True
        assert config.n_jobs == -1
        assert len(config.phases) == 4  # 4 default phases

    def test_default_phases(self):
        """Test that default phases are created correctly."""
        config = FlattenConfig()
        phase_names = [p.name for p in config.phases]
        assert "area_dominant" in phase_names
        assert "balanced" in phase_names
        assert "distance_dominant" in phase_names
        assert "distance_refinement" in phase_names

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = FlattenConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "kring" in d
        assert "phases" in d
        assert "negative_area_removal" in d
        assert "spring_smoothing" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "kring": {"k_ring": 15, "n_neighbors_per_ring": 20},
            "verbose": False,
            "n_jobs": 4,
            # Need to provide phases with required fields or use default
            "phases": [
                {"name": "test_phase", "area_ratio": 1.0},
            ],
        }
        config = FlattenConfig.from_dict(d)
        assert config.kring.k_ring == 15
        assert config.kring.n_neighbors_per_ring == 20
        assert config.verbose is False
        assert config.n_jobs == 4

    def test_json_roundtrip(self):
        """Test JSON save/load roundtrip."""
        config = FlattenConfig()
        config.kring.k_ring = 25
        config.verbose = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Write JSON using to_json method
            with open(temp_path, "w") as f:
                f.write(config.to_json())
            loaded = FlattenConfig.from_json_file(temp_path)
            assert loaded.kring.k_ring == 25
            assert loaded.verbose is False
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestGetKringCacheFilename:
    """Tests for get_kring_cache_filename function."""

    def test_basic_filename(self):
        """Test basic cache filename generation."""
        output_path = "/path/to/output.patch.3d"
        kring = KRingConfig(k_ring=20, n_neighbors_per_ring=30)
        result = get_kring_cache_filename(output_path, kring)
        assert "k20_n30" in result
        assert result.endswith(".npz")

    def test_different_params(self):
        """Test that different params produce different filenames."""
        output_path = "/path/to/output.patch.3d"
        kring1 = KRingConfig(k_ring=20, n_neighbors_per_ring=30)
        kring2 = KRingConfig(k_ring=25, n_neighbors_per_ring=40)
        result1 = get_kring_cache_filename(output_path, kring1)
        result2 = get_kring_cache_filename(output_path, kring2)
        assert result1 != result2
        assert "k20" in result1
        assert "k25" in result2


class TestFlippedTriangles:
    """Tests for flipped triangle counting."""

    def test_no_flipped_triangles(self):
        """Test mesh with no flipped triangles."""
        # Counter-clockwise triangles (normal orientation)
        uv = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [1.5, 1.0],
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],  # Counter-clockwise
                [1, 3, 2],  # Counter-clockwise
            ]
        )
        n_flipped = count_flipped_triangles(uv, faces)
        assert n_flipped == 0

    def test_one_flipped_triangle(self):
        """Test mesh with one flipped triangle."""
        uv = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.5, -1.0],
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],  # Counter-clockwise (normal)
                [0, 2, 1],  # Clockwise (flipped - reversed order)
            ]
        )
        n_flipped = count_flipped_triangles(uv, faces)
        assert n_flipped == 1

    def test_all_flipped_triangles(self):
        """Test mesh with all triangles flipped."""
        # Clockwise triangles
        uv = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ]
        )
        faces = np.array(
            [
                [0, 2, 1],  # Clockwise (flipped)
            ]
        )
        n_flipped = count_flipped_triangles(uv, faces)
        assert n_flipped == 1
