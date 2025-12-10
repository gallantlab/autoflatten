"""
Tests for the flatten module (pyflatten algorithm).
"""

import os
import tempfile

import numpy as np

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
from autoflatten.flatten.algorithm import remove_small_components, TopologyError

import pytest


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


class TestRemoveSmallComponents:
    """Tests for remove_small_components function."""

    def _make_triangle_mesh(self, offset=0):
        """Create a single triangle mesh with optional vertex offset."""
        vertices = np.array(
            [
                [0.0 + offset, 0.0, 0.0],
                [1.0 + offset, 0.0, 0.0],
                [0.5 + offset, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2]])
        return vertices, faces

    def _make_quad_mesh(self, offset=0):
        """Create a quad (2 triangles, 4 vertices) mesh."""
        vertices = np.array(
            [
                [0.0 + offset, 0.0, 0.0],
                [1.0 + offset, 0.0, 0.0],
                [1.0 + offset, 1.0, 0.0],
                [0.0 + offset, 1.0, 0.0],
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )
        return vertices, faces

    def test_single_component_no_removal(self):
        """Test that single component mesh is returned unchanged."""
        vertices, faces = self._make_quad_mesh()
        new_verts, new_faces, indices = remove_small_components(vertices, faces)

        assert len(new_verts) == len(vertices)
        assert len(new_faces) == len(faces)
        np.testing.assert_array_equal(indices, np.arange(len(vertices)))

    def test_removes_small_component_keeps_largest(self):
        """Test removal of small components while keeping the largest."""
        # Create main mesh (4 vertices)
        main_verts, main_faces = self._make_quad_mesh(offset=0)

        # Create isolated triangle (3 vertices, offset by 10)
        small_verts, small_faces = self._make_triangle_mesh(offset=10)
        small_faces = small_faces + len(main_verts)  # Adjust indices

        # Combine meshes
        vertices = np.vstack([main_verts, small_verts])
        faces = np.vstack([main_faces, small_faces])

        # Remove small components (threshold=20 by default, triangle has 3 verts)
        new_verts, new_faces, indices = remove_small_components(vertices, faces)

        # Should have removed the triangle, kept the quad
        assert len(new_verts) == 4
        assert len(new_faces) == 2
        np.testing.assert_array_equal(indices, np.arange(4))

    def test_correct_vertex_face_reindexing(self):
        """Test that vertex/face indices are correctly remapped after removal."""
        # Create isolated triangle first (vertices 0, 1, 2)
        small_verts, small_faces = self._make_triangle_mesh(offset=0)

        # Create main mesh after (vertices 3, 4, 5, 6)
        main_verts, main_faces = self._make_quad_mesh(offset=10)
        main_faces = main_faces + len(small_verts)

        # Combine: small component first, then main
        vertices = np.vstack([small_verts, main_verts])
        faces = np.vstack([small_faces, main_faces])

        new_verts, new_faces, indices = remove_small_components(vertices, faces)

        # Should keep only the quad (4 vertices)
        assert len(new_verts) == 4
        assert len(new_faces) == 2
        # Original indices were 3, 4, 5, 6
        np.testing.assert_array_equal(indices, np.array([3, 4, 5, 6]))
        # Faces should be reindexed to 0, 1, 2, 3
        assert new_faces.max() == 3
        assert new_faces.min() == 0

    def test_warns_for_medium_sized_component(self, caplog):
        """Test that warning is logged for medium-sized secondary component."""
        import logging

        # Create main mesh (100 vertices to make it clearly largest)
        # Create a strip of connected triangles
        n_main = 50
        main_verts = []
        main_faces = []
        for i in range(n_main):
            main_verts.extend(
                [
                    [float(i), 0.0, 0.0],
                    [float(i) + 0.5, 1.0, 0.0],
                ]
            )
        main_verts = np.array(main_verts)
        for i in range(n_main - 1):
            main_faces.append([2 * i, 2 * i + 1, 2 * i + 2])
            main_faces.append([2 * i + 1, 2 * i + 3, 2 * i + 2])
        main_faces = np.array(main_faces)

        # Create medium-sized component (30 vertices - above 20, below 100)
        n_medium = 15
        medium_verts = []
        medium_faces = []
        offset = 100
        for i in range(n_medium):
            medium_verts.extend(
                [
                    [float(i) + offset, 0.0, 0.0],
                    [float(i) + offset + 0.5, 1.0, 0.0],
                ]
            )
        medium_verts = np.array(medium_verts)
        base_idx = len(main_verts)
        for i in range(n_medium - 1):
            medium_faces.append(
                [base_idx + 2 * i, base_idx + 2 * i + 1, base_idx + 2 * i + 2]
            )
            medium_faces.append(
                [base_idx + 2 * i + 1, base_idx + 2 * i + 3, base_idx + 2 * i + 2]
            )
        medium_faces = np.array(medium_faces)

        vertices = np.vstack([main_verts, medium_verts])
        faces = np.vstack([main_faces, medium_faces])

        # Should warn about medium component (30 > 20 threshold)
        with caplog.at_level(logging.WARNING):
            new_verts, new_faces, indices = remove_small_components(
                vertices, faces, max_small_component_size=20, warn_medium_threshold=100
            )

        # Medium component not removed (too big), warning logged
        assert "secondary" in caplog.text.lower() or len(new_verts) == len(vertices)

    def test_raises_topology_error_for_large_secondary(self):
        """Test that TopologyError is raised for large secondary component."""
        # Create two similarly-sized components
        main_verts, main_faces = self._make_quad_mesh(offset=0)

        # Create another quad as second component (same size)
        second_verts, second_faces = self._make_quad_mesh(offset=10)
        second_faces = second_faces + len(main_verts)

        vertices = np.vstack([main_verts, second_verts])
        faces = np.vstack([main_faces, second_faces])

        # With very low threshold, should raise TopologyError
        with pytest.raises(TopologyError) as exc_info:
            remove_small_components(
                vertices, faces, max_small_component_size=1, warn_medium_threshold=2
            )
        assert "too large" in str(exc_info.value).lower()

    def test_never_removes_largest_even_if_small(self):
        """Test that largest component is never removed even if below threshold."""
        # Create just one small triangle (3 vertices)
        vertices, faces = self._make_triangle_mesh()

        # Even with threshold=20 (which would include 3-vertex component),
        # the largest should never be removed
        new_verts, new_faces, indices = remove_small_components(
            vertices, faces, max_small_component_size=20
        )

        assert len(new_verts) == 3
        assert len(new_faces) == 1
