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
    FinalNegativeAreaRemovalConfig,
    get_kring_cache_filename,
)
from autoflatten.flatten import count_flipped_triangles
from autoflatten.flatten.algorithm import (
    remove_small_components,
    count_boundary_loops,
    TopologyError,
    _apply_area_preserving_scale,
)
from autoflatten.flatten.energy import (
    compute_3d_surface_area,
    compute_3d_surface_area_jax,
    compute_2d_areas,
)
import jax.numpy as jnp

import pytest


class TestKRingConfig:
    """Tests for KRingConfig dataclass."""

    def test_default_values(self):
        """Test default values for KRingConfig."""
        config = KRingConfig()
        assert config.k_ring == 7
        assert config.n_neighbors_per_ring == 12

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
        """Test default values for PhaseConfig (requires name)."""
        config = PhaseConfig(name="test")
        assert config.name == "test"
        assert config.l_nlarea == 1.0
        assert config.l_dist == 1.0
        assert config.enabled is True
        assert config.iters_per_level == 40  # FreeSurfer default
        assert config.base_tol is None
        assert len(config.smoothing_schedule) == 7

    def test_custom_phase(self):
        """Test custom phase configuration."""
        config = PhaseConfig(
            name="test_phase",
            l_nlarea=1.0,
            l_dist=0.1,
            enabled=True,
            iters_per_level=100,
            base_tol=0.5,
        )
        assert config.name == "test_phase"
        assert config.l_nlarea == 1.0
        assert config.l_dist == 0.1
        assert config.iters_per_level == 100
        assert config.base_tol == 0.5


class TestNegativeAreaRemovalConfig:
    """Tests for NegativeAreaRemovalConfig dataclass."""

    def test_default_values(self):
        """Test default values for NegativeAreaRemovalConfig."""
        config = NegativeAreaRemovalConfig()
        assert config.enabled is True
        assert config.base_averages == 1024  # FreeSurfer default
        assert config.min_area_pct == 0.5
        # Note: max_passes was removed - FreeSurfer always runs all ratios
        assert config.l_nlarea == 1.0  # Fixed area weight
        assert config.l_dist_ratios == [
            1e-6,
            1e-5,
            1e-3,
            1e-2,
            1e-1,
        ]  # FreeSurfer ratios (all 5 always run)
        assert config.iters_per_level == 30  # FreeSurfer default
        assert config.base_tol == 0.5
        # scale_area is disabled by default (FreeSurfer has this step commented out)
        assert config.scale_area is False


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
        assert len(config.phases) == 3  # 3 FreeSurfer-style epochs
        assert config.adaptive_recovery is False  # Disabled by default

    def test_default_phases(self):
        """Test that default phases are created correctly (FreeSurfer 3-epoch structure)."""
        config = FlattenConfig()
        phase_names = [p.name for p in config.phases]
        assert "epoch_1" in phase_names
        assert "epoch_2" in phase_names
        assert "epoch_3" in phase_names
        # Check FreeSurfer-style weights
        epoch_1 = config.phases[0]
        assert epoch_1.l_nlarea == 1.0
        assert epoch_1.l_dist == 0.1
        epoch_3 = config.phases[2]
        assert epoch_3.l_nlarea == 0.1
        assert epoch_3.l_dist == 1.0

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
                {"name": "test_phase", "l_nlarea": 1.0, "l_dist": 0.1},
            ],
        }
        config = FlattenConfig.from_dict(d)
        assert config.kring.k_ring == 15
        assert config.kring.n_neighbors_per_ring == 20
        assert config.verbose is False
        assert config.n_jobs == 4
        assert config.phases[0].l_nlarea == 1.0
        assert config.phases[0].l_dist == 0.1

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
        assert "secondary" in caplog.text.lower(), (
            "Expected warning about secondary component"
        )
        assert len(new_verts) == len(vertices), "Medium component should not be removed"

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


class TestCountBoundaryLoops:
    """Tests for count_boundary_loops function."""

    def test_single_triangle(self):
        """A single triangle has 1 boundary loop with 3 vertices."""
        faces = np.array([[0, 1, 2]])
        n_loops, loops = count_boundary_loops(faces)
        assert n_loops == 1
        assert len(loops[0]) == 3

    def test_two_triangles_sharing_edge(self):
        """Two triangles sharing an edge have 1 boundary loop with 4 vertices."""
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        n_loops, loops = count_boundary_loops(faces)
        assert n_loops == 1
        assert len(loops[0]) == 4

    def test_triangle_strip(self):
        """A strip of triangles has a single boundary loop."""
        # Create a strip: 3 triangles in a row
        # 0---1---3---5
        # |\ | \ | \ |
        # | \|  \|  \|
        # 2---4---6---7
        faces = np.array(
            [
                [0, 1, 2],
                [1, 4, 2],
                [1, 3, 4],
                [3, 6, 4],
                [3, 5, 6],
                [5, 7, 6],
            ]
        )
        n_loops, loops = count_boundary_loops(faces)
        assert n_loops == 1
        # Boundary should contain all outer vertices: 0, 2, 4, 6, 7, 5, 3, 1 (or some order)
        assert len(loops[0]) == 8

    def test_ring_with_hole(self):
        """A ring mesh (annulus) has 2 boundary loops."""
        # Create an annulus: outer ring and inner ring
        # Outer vertices: 0, 1, 2, 3 (square)
        # Inner vertices: 4, 5, 6, 7 (smaller square)
        # Triangulate the ring: connect outer[i] -> outer[i+1] -> inner[i+1]
        # and outer[i] -> inner[i+1] -> inner[i]
        faces = np.array(
            [
                # Top edge
                [0, 1, 5],
                [0, 5, 4],
                # Right edge
                [1, 2, 6],
                [1, 6, 5],
                # Bottom edge
                [2, 3, 7],
                [2, 7, 6],
                # Left edge
                [3, 0, 4],
                [3, 4, 7],
            ]
        )
        n_loops, loops = count_boundary_loops(faces)
        assert n_loops == 2
        # One loop should have 4 vertices (inner), one should have 4 (outer)
        loop_sizes = sorted([len(loop) for loop in loops])
        assert loop_sizes == [4, 4]

    def test_empty_faces(self):
        """Empty faces array returns 0 loops."""
        faces = np.array([]).reshape(0, 3).astype(int)
        n_loops, loops = count_boundary_loops(faces)
        assert n_loops == 0
        assert loops == []


class TestCompute3DSurfaceArea:
    """Tests for compute_3d_surface_area functions."""

    def test_single_triangle(self):
        """Test area of a single triangle."""
        # Right triangle with legs of length 1
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2]])
        area = compute_3d_surface_area(vertices, faces)
        assert np.isclose(area, 0.5)  # Area = 0.5 * base * height = 0.5

    def test_unit_square(self):
        """Test area of unit square (two triangles)."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )
        area = compute_3d_surface_area(vertices, faces)
        assert np.isclose(area, 1.0)  # Unit square

    def test_equilateral_triangle(self):
        """Test area of equilateral triangle with side length 2."""
        # Equilateral triangle with side length 2
        # Area = (sqrt(3)/4) * side^2 = sqrt(3)
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, np.sqrt(3), 0.0],
            ]
        )
        faces = np.array([[0, 1, 2]])
        area = compute_3d_surface_area(vertices, faces)
        expected = np.sqrt(3)  # ≈ 1.732
        assert np.isclose(area, expected, rtol=1e-5)

    def test_3d_surface(self):
        """Test area of a 3D surface (not flat)."""
        # Triangle tilted in 3D space
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )
        faces = np.array([[0, 1, 2]])
        area = compute_3d_surface_area(vertices, faces)
        # Cross product of edges: (1,0,0) x (0,1,1) = (0,-1,1)
        # |cross| = sqrt(0 + 1 + 1) = sqrt(2)
        # Area = 0.5 * sqrt(2)
        assert np.isclose(area, 0.5 * np.sqrt(2))

    def test_jax_version_matches(self):
        """Test that JIT version matches wrapper."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.5],
            ]
        )
        faces = np.array([[0, 1, 2]])
        area_wrapper = compute_3d_surface_area(vertices, faces)
        area_jax = float(
            compute_3d_surface_area_jax(jnp.asarray(vertices), jnp.asarray(faces))
        )
        assert np.isclose(area_wrapper, area_jax)

    def test_degenerate_triangle(self):
        """Test degenerate triangle (collinear points)."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],  # Collinear with first two
            ]
        )
        faces = np.array([[0, 1, 2]])
        area = compute_3d_surface_area(vertices, faces)
        assert np.isclose(area, 0.0)


class TestCompute2DAreas:
    """Tests for compute_2d_areas function."""

    def test_single_ccw_triangle(self):
        """Test single counter-clockwise (correct) triangle."""
        uv = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        faces = jnp.array([[0, 1, 2]])
        total_area, neg_area = compute_2d_areas(uv, faces)
        assert np.isclose(float(total_area), 0.5)
        assert np.isclose(float(neg_area), 0.0)

    def test_single_cw_triangle(self):
        """Test single clockwise (flipped) triangle."""
        uv = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        # Reversed winding: 0, 2, 1 instead of 0, 1, 2
        faces = jnp.array([[0, 2, 1]])
        total_area, neg_area = compute_2d_areas(uv, faces)
        assert np.isclose(float(total_area), -0.5)
        assert np.isclose(float(neg_area), 0.5)

    def test_mixed_orientation(self):
        """Test mesh with one CCW and one CW triangle."""
        uv = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [1.5, 1.0],
            ]
        )
        # First triangle CCW (positive), second triangle CW (negative)
        # Triangle [1, 2, 3]: v0=(1,0), v1=(0.5,1), v2=(1.5,1)
        # cross = (0.5-1)*(1-0) - (1.5-1)*(1-0) = -0.5 - 0.5 = -1.0 → negative
        faces = jnp.array(
            [
                [0, 1, 2],  # CCW - area = 0.5
                [1, 2, 3],  # CW - area = -0.5
            ]
        )
        total_area, neg_area = compute_2d_areas(uv, faces)
        # Total area: 0.5 - 0.5 = 0
        assert np.isclose(float(total_area), 0.0)
        # Negative area: |second triangle| = 0.5
        assert np.isclose(float(neg_area), 0.5)

    def test_total_plus_neg_equals_positive_area_sum(self):
        """Test that total_area + neg_area = sum of positive areas.

        The formula total_area + neg_area gives the sum of positive
        (non-flipped) triangle areas, which is used in FreeSurfer's
        area-preserving scaling to normalize by "useful" area.
        """
        uv = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ]
        )
        # Test with CCW (positive) triangle
        faces_ccw = jnp.array([[0, 1, 2]])
        total_ccw, neg_ccw = compute_2d_areas(uv, faces_ccw)

        # CCW: total=0.5, neg=0, total+neg=0.5 (positive area)
        assert np.isclose(float(total_ccw), 0.5)
        assert np.isclose(float(neg_ccw), 0.0)
        assert np.isclose(float(total_ccw + neg_ccw), 0.5)

        # Test with CW (negative/flipped) triangle
        faces_cw = jnp.array([[0, 2, 1]])
        total_cw, neg_cw = compute_2d_areas(uv, faces_cw)

        # CW: total=-0.5, neg=0.5, total+neg=0 (no positive area)
        assert np.isclose(float(total_cw), -0.5)
        assert np.isclose(float(neg_cw), 0.5)
        assert np.isclose(float(total_cw + neg_cw), 0.0)

    def test_unit_square(self):
        """Test unit square (two triangles, both positive)."""
        uv = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        faces = jnp.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )
        total_area, neg_area = compute_2d_areas(uv, faces)
        assert np.isclose(float(total_area), 1.0)
        assert np.isclose(float(neg_area), 0.0)

    def test_degenerate_triangle(self):
        """Test degenerate (zero area) triangle."""
        uv = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],  # Collinear
            ]
        )
        faces = jnp.array([[0, 1, 2]])
        total_area, neg_area = compute_2d_areas(uv, faces)
        assert np.isclose(float(total_area), 0.0)
        assert np.isclose(float(neg_area), 0.0)


class TestInitialScaleConfig:
    """Tests for initial_scale config parameter."""

    def test_default_value(self):
        """Test that default initial_scale is 3.0."""
        config = FlattenConfig()
        assert config.initial_scale == 3.0

    def test_custom_value(self):
        """Test setting custom initial_scale value."""
        config = FlattenConfig(initial_scale=5.0)
        assert config.initial_scale == 5.0

    def test_to_dict_includes_initial_scale(self):
        """Test that to_dict includes initial_scale."""
        config = FlattenConfig(initial_scale=4.0)
        d = config.to_dict()
        assert "initial_scale" in d
        assert d["initial_scale"] == 4.0

    def test_from_dict_loads_initial_scale(self):
        """Test that from_dict correctly loads initial_scale."""
        d = {
            "initial_scale": 2.5,
            "phases": [{"name": "test", "l_nlarea": 1.0, "l_dist": 1.0}],
        }
        config = FlattenConfig.from_dict(d)
        assert config.initial_scale == 2.5

    def test_json_roundtrip(self):
        """Test JSON roundtrip preserves initial_scale."""
        config = FlattenConfig(initial_scale=6.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            with open(temp_path, "w") as f:
                f.write(config.to_json())
            loaded = FlattenConfig.from_json_file(temp_path)
            assert loaded.initial_scale == 6.0
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestApplyAreaPreservingScale:
    """Tests for _apply_area_preserving_scale function."""

    def test_preserves_target_area(self):
        """Test that scaling achieves target area."""
        # Create a simple unit square
        uv = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        faces = jnp.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )
        orig_area = 4.0  # Target area = 4 (double the current)

        scaled_uv = _apply_area_preserving_scale(uv, faces, orig_area)

        # Compute new area
        total_area, _ = compute_2d_areas(scaled_uv, faces)
        assert np.isclose(float(total_area), orig_area, rtol=1e-5)

    def test_scaling_is_centered(self):
        """Test that scaling preserves centroid."""
        uv = jnp.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ]
        )
        faces = jnp.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )
        orig_area = 16.0  # Target: scale up

        original_centroid = jnp.mean(uv, axis=0)
        scaled_uv = _apply_area_preserving_scale(uv, faces, orig_area)
        scaled_centroid = jnp.mean(scaled_uv, axis=0)

        assert np.allclose(original_centroid, scaled_centroid, atol=1e-5)

    def test_scale_down(self):
        """Test scaling down (target area < current area)."""
        uv = jnp.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ]
        )
        faces = jnp.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )
        # Current area = 4, target = 1
        orig_area = 1.0

        scaled_uv = _apply_area_preserving_scale(uv, faces, orig_area)
        total_area, _ = compute_2d_areas(scaled_uv, faces)
        assert np.isclose(float(total_area), orig_area, rtol=1e-5)

    def test_handles_negative_area_triangles(self):
        """Test handling of mesh with flipped triangles."""
        # Create mesh with one flipped triangle
        uv = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [1.5, 0.5],
            ]
        )
        # First triangle CCW, second CW (flipped)
        faces = jnp.array(
            [
                [0, 1, 2],  # CCW
                [1, 2, 3],  # Could be either orientation
            ]
        )
        orig_area = 2.0

        # Should not raise, should handle gracefully
        scaled_uv = _apply_area_preserving_scale(uv, faces, orig_area)
        assert scaled_uv.shape == uv.shape

    def test_division_by_zero_protection(self):
        """Test that degenerate case doesn't cause division by zero."""
        # Degenerate mesh: all vertices at same point
        uv = jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        faces = jnp.array([[0, 1, 2]])
        orig_area = 1.0

        # Should not raise - epsilon protection should kick in
        scaled_uv = _apply_area_preserving_scale(uv, faces, orig_area)
        assert not jnp.any(jnp.isnan(scaled_uv))
        assert not jnp.any(jnp.isinf(scaled_uv))


class TestFinalNegativeAreaRemovalConfig:
    """Tests for FinalNegativeAreaRemovalConfig dataclass."""

    def test_default_values(self):
        """Test default values match FreeSurfer defaults."""
        config = FinalNegativeAreaRemovalConfig()
        assert config.enabled is True
        assert config.base_averages == 32  # Capped in FreeSurfer
        assert config.l_nlarea == 1.0
        # Uses full ratio schedule like initial NAR
        assert config.l_dist_ratios == [1e-6, 1e-5, 1e-3, 1e-2, 1e-1]
        assert config.base_tol == 0.01  # Tighter than initial
        assert config.iters_per_level == 30

    def test_custom_values(self):
        """Test custom values for FinalNegativeAreaRemovalConfig."""
        config = FinalNegativeAreaRemovalConfig(
            enabled=False,
            base_averages=16,
            l_nlarea=2.0,
            l_dist_ratios=[1e-4, 1e-3, 1e-2],
            base_tol=0.005,
            iters_per_level=50,
        )
        assert config.enabled is False
        assert config.base_averages == 16
        assert config.l_nlarea == 2.0
        assert config.l_dist_ratios == [1e-4, 1e-3, 1e-2]
        assert config.base_tol == 0.005
        assert config.iters_per_level == 50

    def test_disabled_by_default_is_false(self):
        """Verify final NAR is enabled by default (unlike scale_area)."""
        config = FinalNegativeAreaRemovalConfig()
        assert config.enabled is True


class TestFinalNegativeAreaRemovalSerialization:
    """Tests for FinalNegativeAreaRemovalConfig serialization in FlattenConfig."""

    def test_to_dict_includes_final_nar(self):
        """Test that to_dict includes final_negative_area_removal."""
        config = FlattenConfig()
        d = config.to_dict()
        assert "final_negative_area_removal" in d
        assert d["final_negative_area_removal"]["enabled"] is True
        assert d["final_negative_area_removal"]["base_averages"] == 32
        assert d["final_negative_area_removal"]["l_dist_ratios"] == [
            1e-6,
            1e-5,
            1e-3,
            1e-2,
            1e-1,
        ]

    def test_from_dict_loads_final_nar(self):
        """Test that from_dict correctly loads final_negative_area_removal."""
        d = {
            "final_negative_area_removal": {
                "enabled": False,
                "base_averages": 16,
                "l_nlarea": 0.5,
                "l_dist_ratios": [1e-4, 1e-3],
                "base_tol": 0.02,
                "iters_per_level": 20,
            }
        }
        config = FlattenConfig.from_dict(d)
        assert config.final_negative_area_removal.enabled is False
        assert config.final_negative_area_removal.base_averages == 16
        assert config.final_negative_area_removal.l_nlarea == 0.5
        assert config.final_negative_area_removal.l_dist_ratios == [1e-4, 1e-3]
        assert config.final_negative_area_removal.base_tol == 0.02
        assert config.final_negative_area_removal.iters_per_level == 20

    def test_json_roundtrip_preserves_final_nar(self):
        """Test JSON roundtrip preserves final_negative_area_removal settings."""
        config = FlattenConfig()
        config.final_negative_area_removal.enabled = False
        config.final_negative_area_removal.base_averages = 64
        config.final_negative_area_removal.l_dist_ratios = [1e-5, 1e-4, 1e-3]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            with open(temp_path, "w") as f:
                f.write(config.to_json())
            loaded = FlattenConfig.from_json_file(temp_path)
            assert loaded.final_negative_area_removal.enabled is False
            assert loaded.final_negative_area_removal.base_averages == 64
            assert loaded.final_negative_area_removal.l_dist_ratios == [
                1e-5,
                1e-4,
                1e-3,
            ]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestInitialScaleInProjection:
    """Tests for initial_scale configuration and its effect on projection."""

    def test_initial_scale_default_value(self):
        """Test that initial_scale defaults to 3.0 (FreeSurfer default)."""
        config = FlattenConfig()
        assert config.initial_scale == 3.0

    def test_initial_scale_in_to_dict(self):
        """Test that initial_scale is included in to_dict."""
        config = FlattenConfig()
        config.initial_scale = 2.5
        d = config.to_dict()
        assert "initial_scale" in d
        assert d["initial_scale"] == 2.5

    def test_initial_scale_from_dict(self):
        """Test that initial_scale is loaded from dict."""
        d = {"initial_scale": 4.0}
        config = FlattenConfig.from_dict(d)
        assert config.initial_scale == 4.0

    def test_initial_scale_json_roundtrip(self):
        """Test JSON roundtrip preserves initial_scale."""
        config = FlattenConfig()
        config.initial_scale = 5.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            with open(temp_path, "w") as f:
                f.write(config.to_json())
            loaded = FlattenConfig.from_json_file(temp_path)
            assert loaded.initial_scale == 5.0
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
