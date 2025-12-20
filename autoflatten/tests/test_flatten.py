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
        # FreeSurfer always runs all ratios in the l_dist_ratios list
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


class TestBoundaryVertexPreservation:
    """Tests for boundary vertex preservation through SurfaceFlattener."""

    def test_is_border_preserved_through_save_result(self):
        """
        Test that border vertex flags are preserved from input patch to output.

        This is a regression test for the bug where is_border was discarded
        during SurfaceFlattener.load_data() and not passed to write_patch().
        """
        import nibabel.freesurfer.io as fsio
        from autoflatten.freesurfer import write_patch, read_patch
        from autoflatten.flatten import SurfaceFlattener, FlattenConfig

        # Create a simple mesh: a strip of triangles
        # Vertices form a grid:
        #   0---2---4---6
        #   |\ | \ | \ |
        #   | \|  \|  \|
        #   1---3---5---7
        n_cols = 4
        vertices = []
        for i in range(n_cols):
            vertices.append([float(i), 0.0, 0.0])  # Top row
            vertices.append([float(i), 1.0, 0.0])  # Bottom row
        vertices = np.array(vertices, dtype=np.float32)

        # Create faces (counter-clockwise winding)
        faces = []
        for i in range(n_cols - 1):
            top_left = 2 * i
            top_right = 2 * i + 2
            bot_left = 2 * i + 1
            bot_right = 2 * i + 3
            faces.append([top_left, bot_left, top_right])
            faces.append([bot_left, bot_right, top_right])
        faces = np.array(faces, dtype=np.int32)

        n_vertices = len(vertices)
        original_indices = np.arange(n_vertices, dtype=np.int32)

        # Mark boundary vertices (outer edges of the strip)
        is_border = np.zeros(n_vertices, dtype=bool)
        is_border[0] = True  # First vertex
        is_border[1] = True  # Second vertex
        is_border[-2] = True  # Second-to-last vertex
        is_border[-1] = True  # Last vertex

        with tempfile.TemporaryDirectory() as temp_dir:
            # Write surface file using nibabel
            surface_path = os.path.join(temp_dir, "test.smoothwm")
            fsio.write_geometry(surface_path, vertices, faces)

            # Write patch file with border vertices
            patch_path = os.path.join(temp_dir, "test.patch.3d")
            write_patch(patch_path, vertices, original_indices, is_border)

            # Load through SurfaceFlattener
            config = FlattenConfig(verbose=False)
            flattener = SurfaceFlattener(config)
            flattener.load_data(patch_path, surface_path)

            # Verify is_border was stored
            assert hasattr(flattener, "is_border"), (
                "SurfaceFlattener should have is_border attribute"
            )
            assert flattener.is_border is not None, "is_border should not be None"

            # Create output UV (just use XY coordinates as simple "flat" result)
            uv = flattener.vertices[:, :2]

            # Save result
            output_path = os.path.join(temp_dir, "test.flat.patch.3d")
            flattener.save_result(uv, output_path)

            # Read back the output patch
            _, read_indices, read_is_border = read_patch(output_path)

            # Verify border vertices are preserved
            # Note: vertices may be reordered/filtered, so we need to check
            # that the border status matches for corresponding original indices
            for i, orig_idx in enumerate(flattener.orig_indices):
                expected_border = is_border[orig_idx]
                actual_border = read_is_border[i]
                assert actual_border == expected_border, (
                    f"Border status mismatch for original vertex {orig_idx}: "
                    f"expected {expected_border}, got {actual_border}"
                )

            # Also verify at least some border vertices exist in output
            n_border_out = np.sum(read_is_border)
            n_border_in = np.sum(flattener.is_border)
            assert n_border_out == n_border_in, (
                f"Number of border vertices mismatch: input had {n_border_in}, "
                f"output has {n_border_out}"
            )


# =============================================================================
# Tier 2: Energy Function Tests
# =============================================================================


class TestPrepareMetricData:
    """Tests for prepare_metric_data function."""

    def test_padding_with_variable_neighbors(self):
        """Test padding arrays with different neighbor counts."""
        from autoflatten.flatten.energy import prepare_metric_data

        # Vertex 0 has 2 neighbors, vertex 1 has 3 neighbors
        k_rings = [np.array([1, 2]), np.array([0, 2, 3])]
        target_distances = [np.array([1.0, 1.0]), np.array([1.0, 1.0, 1.414])]

        neighbors, targets, mask = prepare_metric_data(k_rings, target_distances)

        # Should be padded to max_neighbors=3
        assert neighbors.shape == (2, 3)
        assert targets.shape == (2, 3)
        assert mask.shape == (2, 3)

        # Check mask is correct
        assert mask[0, 0] and mask[0, 1] and not mask[0, 2]
        assert mask[1, 0] and mask[1, 1] and mask[1, 2]

    def test_single_vertex_single_neighbor(self):
        """Test with minimal case."""
        from autoflatten.flatten.energy import prepare_metric_data

        k_rings = [np.array([1])]
        target_distances = [np.array([1.0])]

        neighbors, targets, mask = prepare_metric_data(k_rings, target_distances)

        assert neighbors.shape == (1, 1)
        assert neighbors[0, 0] == 1
        assert targets[0, 0] == 1.0
        assert mask[0, 0]


class TestPrepareEdgeList:
    """Tests for prepare_edge_list function."""

    def test_edge_list_from_k_rings(self):
        """Test converting k-rings to edge list."""
        from autoflatten.flatten.energy import prepare_edge_list

        k_rings = [np.array([1, 2]), np.array([0, 2])]
        target_distances = [np.array([1.0, 1.5]), np.array([1.0, 1.2])]

        src, dst, targets, n_vertices = prepare_edge_list(k_rings, target_distances)

        assert n_vertices == 2
        assert len(src) == 4  # 2 + 2 edges
        assert len(dst) == 4
        assert len(targets) == 4

        # Source vertices should be in order
        assert list(src[:2]) == [0, 0]
        assert list(src[2:]) == [1, 1]

    def test_empty_k_ring(self):
        """Test with a vertex that has no neighbors."""
        from autoflatten.flatten.energy import prepare_edge_list

        k_rings = [np.array([1]), np.array([], dtype=np.int64)]
        target_distances = [np.array([1.0]), np.array([], dtype=np.float64)]

        src, dst, targets, n_vertices = prepare_edge_list(k_rings, target_distances)

        assert n_vertices == 2
        assert len(src) == 1


class TestMetricEnergyEdges:
    """Tests for compute_metric_energy_edges function."""

    def test_zero_distortion_on_isometric_embedding(self):
        """Test that isometric embedding has zero energy."""
        from autoflatten.flatten.energy import compute_metric_energy_edges

        # Square with unit edges
        uv = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        # Edge list: each vertex connected to its neighbors with distance 1.0
        src = jnp.array([0, 0, 1, 1, 2, 2, 3, 3])
        dst = jnp.array([1, 2, 0, 3, 0, 3, 1, 2])
        targets = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        energy = compute_metric_energy_edges(uv, src, dst, targets, 4)

        assert float(energy) < 1e-10

    def test_positive_energy_on_stretched_mesh(self):
        """Test that stretching produces positive energy."""
        from autoflatten.flatten.energy import compute_metric_energy_edges

        # Stretched square (2x1 instead of 1x1)
        uv = jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]])

        # Target distances are 1.0 but actual x-distances are 2.0
        src = jnp.array([0, 1])
        dst = jnp.array([1, 0])
        targets = jnp.array([1.0, 1.0])

        energy = compute_metric_energy_edges(uv, src, dst, targets, 4)

        # Actual distance is 2.0, target is 1.0, error = (2-1)^2 = 1 per edge
        assert float(energy) > 0


class TestSpringEnergy:
    """Tests for compute_spring_energy function."""

    def test_positive_when_vertices_not_at_centroids(self):
        """Test that vertices NOT at neighbor centroids have positive spring energy."""
        from autoflatten.flatten.energy import compute_spring_energy

        # Triangle mesh where vertices are displaced from neighbor centroids
        uv = jnp.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])

        # Vertex 2's neighbors are 0 and 1, centroid is (1, 0)
        # But vertex 2 is at (1, 1) - displaced from centroid by 1.0 in y
        neighbors = jnp.array([[1, 2], [0, 2], [0, 1]])
        mask = jnp.array([[True, True], [True, True], [True, True]])
        counts = jnp.array([3, 3, 3])  # degree + 1

        energy = compute_spring_energy(uv, neighbors, mask, counts)

        # Energy should be positive since vertices are not at centroids
        assert float(energy) > 0, f"Expected positive energy but got {float(energy)}"

    def test_spring_energy_increases_with_displacement(self):
        """Test that displacing a vertex increases spring energy."""
        from autoflatten.flatten.energy import compute_spring_energy

        neighbors = jnp.array([[1, 2], [0, 2], [0, 1]])
        mask = jnp.array([[True, True], [True, True], [True, True]])
        counts = jnp.array([3, 3, 3])

        # Baseline
        uv1 = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        energy1 = compute_spring_energy(uv1, neighbors, mask, counts)

        # Displaced vertex 2 further from centroid
        uv2 = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 2.0]])
        energy2 = compute_spring_energy(uv2, neighbors, mask, counts)

        assert float(energy2) > float(energy1)


class TestGetVerticesWithNegativeArea:
    """Tests for get_vertices_with_negative_area function."""

    def test_no_negative_area(self, simple_quad_uv):
        """Test mesh with no flipped triangles."""
        from autoflatten.flatten.energy import get_vertices_with_negative_area

        uv = jnp.array(simple_quad_uv)
        faces = jnp.array([[0, 1, 2], [1, 3, 2]])

        has_neg = get_vertices_with_negative_area(uv, faces)

        assert not any(has_neg)

    def test_detects_flipped_triangle(self):
        """Test detection of vertices in flipped triangles."""
        from autoflatten.flatten.energy import get_vertices_with_negative_area

        uv = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        # Second triangle is CW (flipped)
        faces = jnp.array([[0, 1, 2], [1, 2, 3]])

        has_neg = get_vertices_with_negative_area(uv, faces)

        # Vertices 1, 2, 3 are in the flipped triangle
        assert not has_neg[0]  # Only in CCW triangle
        assert has_neg[1]  # In both
        assert has_neg[2]  # In both
        assert has_neg[3]  # Only in CW triangle


class TestPrepareSmoothingData:
    """Tests for prepare_smoothing_data function."""

    def test_adjacency_from_quad_mesh(self, simple_quad_mesh):
        """Test 1-ring adjacency construction."""
        from autoflatten.flatten.energy import prepare_smoothing_data

        vertices, faces = simple_quad_mesh

        neighbors, mask, counts = prepare_smoothing_data(faces, len(vertices))

        assert neighbors.shape[0] == 4
        assert mask.shape[0] == 4
        assert len(counts) == 4

        # Each vertex should have 2-3 neighbors in a quad
        assert all(c >= 3 for c in counts)  # counts = degree + 1

    def test_triangle_adjacency(self):
        """Test adjacency for single triangle."""
        from autoflatten.flatten.energy import prepare_smoothing_data

        faces = np.array([[0, 1, 2]])

        neighbors, mask, counts = prepare_smoothing_data(faces, 3)

        # Each vertex has 2 neighbors
        assert all(counts == 3)  # 2 neighbors + 1
        assert np.sum(mask) == 6  # 3 vertices * 2 neighbors each


class TestSmoothGradientOnce:
    """Tests for smooth_gradient_once function."""

    def test_averaging_effect(self):
        """Test that smoothing averages gradients."""
        from autoflatten.flatten.energy import smooth_gradient_once

        # 3 vertices in a line: 0 -- 1 -- 2
        grad = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

        # Vertex 1 is connected to 0 and 2
        neighbors = jnp.array([[1, 0], [0, 2], [1, 0]])  # padding with 0
        mask = jnp.array([[True, False], [True, True], [True, False]])
        counts = jnp.array([2, 3, 2])  # degree + 1

        smoothed = smooth_gradient_once(grad, neighbors, mask, counts)

        # Vertex 1's gradient should decrease (averaged with zeros)
        assert float(smoothed[1, 0]) < 1.0
        # Vertex 0 should increase (averaged with vertex 1)
        assert float(smoothed[0, 0]) > 0.0


# =============================================================================
# Tier 2: Distance Function Tests
# =============================================================================


class TestBuildMeshGraph:
    """Tests for build_mesh_graph function."""

    def test_csr_matrix_construction(self, simple_quad_mesh):
        """Test sparse matrix construction."""
        from autoflatten.flatten.distance import build_mesh_graph

        vertices, faces = simple_quad_mesh
        graph = build_mesh_graph(vertices, faces)

        # Should be 4x4 sparse matrix
        assert graph.shape == (4, 4)

        # Should be symmetric
        assert (graph != graph.T).nnz == 0

    def test_edge_weights_from_geometry(self):
        """Test that edge weights match geometric distances."""
        from autoflatten.flatten.distance import build_mesh_graph

        # Right triangle with known edge lengths
        vertices = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
        faces = np.array([[0, 1, 2]])

        graph = build_mesh_graph(vertices, faces)

        # Edge 0-1 should be length 3
        assert np.isclose(graph[0, 1], 3.0)
        # Edge 0-2 should be length 4
        assert np.isclose(graph[0, 2], 4.0)
        # Edge 1-2 should be length 5 (hypotenuse)
        assert np.isclose(graph[1, 2], 5.0)


class TestGetKRing:
    """Tests for get_k_ring function."""

    def test_1_ring_on_quad(self, simple_quad_mesh):
        """Test 1-ring on simple quad with specific neighbor verification.

        Quad mesh layout:
            2---3
            |\\  |
            | \\ |
            |  \\|
            0---1

        Faces: [0,1,2] and [1,3,2]
        """
        from autoflatten.flatten.distance import get_k_ring

        vertices, faces = simple_quad_mesh
        k_rings = get_k_ring(faces, len(vertices), k=1)

        assert len(k_rings) == 4, f"Expected 4 vertices, got {len(k_rings)}"

        # Verify specific adjacencies based on mesh topology
        # Vertex 0: adjacent to 1 (shared edge) and 2 (shared edge)
        assert set(k_rings[0]) == {1, 2}, (
            f"Vertex 0 should be adjacent to {{1, 2}}, got {set(k_rings[0])}"
        )
        # Vertex 1: adjacent to 0, 2 (from face [0,1,2]) and 3 (from face [1,3,2])
        assert set(k_rings[1]) == {0, 2, 3}, (
            f"Vertex 1 should be adjacent to {{0, 2, 3}}, got {set(k_rings[1])}"
        )
        # Vertex 2: adjacent to 0, 1 (from face [0,1,2]) and 3 (from face [1,3,2])
        assert set(k_rings[2]) == {0, 1, 3}, (
            f"Vertex 2 should be adjacent to {{0, 1, 3}}, got {set(k_rings[2])}"
        )
        # Vertex 3: adjacent to 1 and 2 (from face [1,3,2])
        assert set(k_rings[3]) == {1, 2}, (
            f"Vertex 3 should be adjacent to {{1, 2}}, got {set(k_rings[3])}"
        )

    def test_2_ring_includes_more_vertices(self, triangle_strip_mesh):
        """Test that 2-ring includes more vertices than 1-ring."""
        from autoflatten.flatten.distance import get_k_ring

        vertices, faces = triangle_strip_mesh

        k_rings_1 = get_k_ring(faces, len(vertices), k=1)
        k_rings_2 = get_k_ring(faces, len(vertices), k=2)

        # 2-ring should have at least as many vertices as 1-ring
        for kr1, kr2 in zip(k_rings_1, k_rings_2):
            assert len(kr2) >= len(kr1)


class TestGetSingleKRing:
    """Tests for get_single_k_ring function."""

    def test_single_vertex_k_ring(self, simple_quad_mesh):
        """Test k-ring for a single vertex."""
        from autoflatten.flatten.distance import get_single_k_ring
        import igl

        vertices, faces = simple_quad_mesh
        adj = igl.adjacency_list(faces.astype(np.int64))

        # Get 1-ring of vertex 0
        kr = get_single_k_ring(adj, center_vertex=0, k=1)

        # Vertex 0 is connected to 1 and 2 in the quad
        assert len(kr) == 2
        assert 0 not in kr  # Center vertex excluded


class TestSelectAngularSamples:
    """Tests for select_angular_samples function."""

    def test_returns_all_if_fewer_than_n_samples(self):
        """Test that all points are returned if fewer than n_samples."""
        from autoflatten.flatten.distance import select_angular_samples

        angles = np.array([0.0, np.pi / 2, np.pi])
        selected = select_angular_samples(angles, n_samples=8)

        assert len(selected) == 3
        assert set(selected) == {0, 1, 2}

    def test_uniform_sampling(self):
        """Test angular sampling with uniform distribution."""
        from autoflatten.flatten.distance import select_angular_samples

        # 16 uniformly distributed angles
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        selected = select_angular_samples(angles, n_samples=8)

        # With uniform distribution, should get exactly 8 samples
        assert len(selected) == 8, (
            f"Expected 8 samples from 16 uniform angles, got {len(selected)}"
        )
        # Selected indices should be valid
        assert all(0 <= idx < 16 for idx in selected), (
            f"Selected indices out of range: {selected}"
        )
        # No duplicates
        assert len(set(selected)) == len(selected), (
            f"Selected indices contain duplicates: {selected}"
        )

    def test_empty_input(self):
        """Test with empty input returns empty array of correct type."""
        from autoflatten.flatten.distance import select_angular_samples

        angles = np.array([])
        selected = select_angular_samples(angles, n_samples=8)

        assert isinstance(selected, np.ndarray), (
            f"Expected ndarray, got {type(selected)}"
        )
        assert len(selected) == 0, f"Expected empty array, got {len(selected)} elements"


class TestThreadConfig:
    """Tests for set_num_threads and get_num_threads."""

    def test_set_get_num_threads(self):
        """Test setting and getting thread count."""
        from autoflatten.flatten.distance import set_num_threads, get_num_threads

        original = get_num_threads()

        try:
            set_num_threads(2)
            assert get_num_threads() == 2
        finally:
            # Restore original
            set_num_threads(original)


# =============================================================================
# Tier 2: Algorithm Utility Tests
# =============================================================================


class TestValidateTopology:
    """Tests for validate_topology function."""

    def test_valid_disk_topology(self, simple_quad_mesh):
        """Test that a quad mesh has valid disk topology (Euler characteristic 1)."""
        from autoflatten.flatten.algorithm import validate_topology

        vertices, faces = simple_quad_mesh

        # Should return 1 (Euler characteristic for disk topology)
        result = validate_topology(vertices, faces)
        assert result == 1

    def test_single_triangle_is_valid(self):
        """Test that a single triangle is valid (Euler characteristic 1)."""
        from autoflatten.flatten.algorithm import validate_topology

        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])

        # Should return 1 (Euler characteristic for disk topology)
        result = validate_topology(vertices, faces)
        assert result == 1


# =============================================================================
# Tier 3: Area Energy Function Tests
# =============================================================================


class TestComputeAreaEnergy:
    """Tests for compute_area_energy (sigmoid-weighted area energy)."""

    def test_zero_energy_on_matching_areas(self):
        """Test that matching 2D and 3D areas give low energy."""
        from autoflatten.flatten.energy import compute_area_energy

        # Unit square with matching areas
        uv = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        faces = jnp.array([[0, 1, 2], [0, 2, 3]])

        # Original areas match 2D areas (each triangle is 0.5)
        original_areas = jnp.array([0.5, 0.5])

        energy = compute_area_energy(uv, faces, original_areas)

        # Energy should be low (sigmoid weights are low for positive ratios)
        assert float(energy) < 0.1

    def test_high_energy_on_flipped_triangles(self):
        """Test that flipped triangles produce high energy."""
        from autoflatten.flatten.energy import compute_area_energy

        # Square with one triangle flipped (CW winding)
        uv = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        faces = jnp.array([[0, 1, 2], [0, 3, 2]])  # Second triangle is CW

        original_areas = jnp.array([0.5, 0.5])

        energy = compute_area_energy(uv, faces, original_areas)

        # Energy should be higher than for correctly oriented mesh
        assert float(energy) > 0.1

    def test_energy_increases_with_more_flips(self):
        """Test that more flipped triangles increase energy."""
        from autoflatten.flatten.energy import compute_area_energy

        uv = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        original_areas = jnp.array([0.5, 0.5])

        # Zero flipped triangles
        faces_good = jnp.array([[0, 1, 2], [0, 2, 3]])
        energy_good = compute_area_energy(uv, faces_good, original_areas)

        # One flipped triangle
        faces_one_flip = jnp.array([[0, 1, 2], [0, 3, 2]])
        energy_one = compute_area_energy(uv, faces_one_flip, original_areas)

        # Two flipped triangles
        faces_two_flips = jnp.array([[0, 2, 1], [0, 3, 2]])
        energy_two = compute_area_energy(uv, faces_two_flips, original_areas)

        assert float(energy_one) > float(energy_good)
        assert float(energy_two) > float(energy_one)

    def test_neg_area_k_parameter(self):
        """Test that neg_area_k affects sigmoid steepness."""
        from autoflatten.flatten.energy import compute_area_energy

        uv = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        faces = jnp.array([[0, 2, 1]])  # Flipped
        original_areas = jnp.array([0.25])

        # Lower k = gentler sigmoid
        energy_low_k = compute_area_energy(uv, faces, original_areas, neg_area_k=1.0)
        # Higher k = steeper sigmoid
        energy_high_k = compute_area_energy(uv, faces, original_areas, neg_area_k=20.0)

        # Both should be positive (flipped triangle)
        assert float(energy_low_k) > 0
        assert float(energy_high_k) > 0


class TestComputeAreaEnergyFsV6:
    """Tests for compute_area_energy_fs_v6 (log-softplus area energy)."""

    def test_low_energy_on_positive_areas(self):
        """Test that positive areas give low energy."""
        from autoflatten.flatten.energy import compute_area_energy_fs_v6

        # CCW triangles (positive areas)
        uv = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        faces = jnp.array([[0, 1, 2], [0, 2, 3]])

        energy = compute_area_energy_fs_v6(uv, faces)

        # Energy should be low for positive areas
        assert float(energy) < 1.0

    def test_high_energy_on_negative_areas(self):
        """Test that negative (flipped) areas give high energy."""
        from autoflatten.flatten.energy import compute_area_energy_fs_v6

        # CW triangle (negative area)
        uv = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        faces_good = jnp.array([[0, 1, 2]])  # CCW
        faces_bad = jnp.array([[0, 2, 1]])  # CW

        energy_good = compute_area_energy_fs_v6(uv, faces_good)
        energy_bad = compute_area_energy_fs_v6(uv, faces_bad)

        assert float(energy_bad) > float(energy_good)

    def test_linear_penalty_for_very_negative_areas(self):
        """Test that log-softplus gives linear penalty for large negative areas."""
        from autoflatten.flatten.energy import compute_area_energy_fs_v6

        # Large negative area (scaled up triangle, flipped)
        uv_small = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        uv_large = jnp.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
        faces = jnp.array([[0, 2, 1]])  # CW (flipped)

        energy_small = compute_area_energy_fs_v6(uv_small, faces)
        energy_large = compute_area_energy_fs_v6(uv_large, faces)

        # Larger negative area should have higher energy
        assert float(energy_large) > float(energy_small)


class TestComputeLogBarrierAreaEnergy:
    """Tests for compute_log_barrier_area_energy (hybrid barrier energy)."""

    def test_low_energy_at_original_size(self):
        """Test that triangles at original proportions have low energy."""
        from autoflatten.flatten.energy import compute_log_barrier_area_energy

        uv = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        faces = jnp.array([[0, 1, 2], [0, 2, 3]])

        # Equal fractions (each triangle is 50% of total)
        original_fracs = jnp.array([0.5, 0.5])

        energy = compute_log_barrier_area_energy(uv, faces, original_fracs)

        # Energy should be low at original proportions
        assert float(energy) < 1.0

    def test_high_energy_when_shrunk(self):
        """Test that shrunk triangles produce higher energy."""
        from autoflatten.flatten.energy import compute_log_barrier_area_energy

        # One triangle is much smaller relative to original
        uv = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.01]])
        faces = jnp.array([[0, 1, 2], [0, 2, 3]])

        # Original had equal fractions
        original_fracs = jnp.array([0.5, 0.5])

        energy = compute_log_barrier_area_energy(uv, faces, original_fracs)

        # Energy should be higher due to shrinkage
        assert float(energy) > 0.5

    def test_barrier_weight_affects_energy(self):
        """Test that barrier_weight parameter affects energy."""
        from autoflatten.flatten.energy import compute_log_barrier_area_energy

        uv = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.1]])
        faces = jnp.array([[0, 1, 2], [0, 2, 3]])
        original_fracs = jnp.array([0.5, 0.5])

        energy_low = compute_log_barrier_area_energy(
            uv, faces, original_fracs, barrier_weight=0.01
        )
        energy_high = compute_log_barrier_area_energy(
            uv, faces, original_fracs, barrier_weight=1.0
        )

        # Higher barrier weight should increase energy for shrunk triangles
        assert float(energy_high) > float(energy_low)


class TestComputeBothEnergies:
    """Tests for compute_both_energies (combined metric + area energy)."""

    def test_returns_two_values(self, simple_quad_mesh, simple_quad_uv):
        """Test that both energy values are returned."""
        from autoflatten.flatten.energy import (
            compute_both_energies,
            prepare_metric_data,
        )

        vertices, faces = simple_quad_mesh
        uv = jnp.array(simple_quad_uv)
        faces_jax = jnp.array(faces)

        # Create k-ring data (simple 1-ring)
        k_rings = [
            np.array([1, 2]),
            np.array([0, 3]),
            np.array([0, 3]),
            np.array([1, 2]),
        ]
        target_distances = [
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        ]

        neighbors, targets, mask = prepare_metric_data(k_rings, target_distances)
        neighbors_jax = jnp.array(neighbors)
        targets_jax = jnp.array(targets)
        mask_jax = jnp.array(mask)

        original_areas = jnp.array([0.5, 0.5])

        J_d, J_a = compute_both_energies(
            uv, neighbors_jax, targets_jax, mask_jax, faces_jax, original_areas
        )

        assert np.isfinite(float(J_d))
        assert np.isfinite(float(J_a))

    def test_matches_individual_calls(self, simple_quad_mesh, simple_quad_uv):
        """Test that combined call matches individual energy functions."""
        from autoflatten.flatten.energy import (
            compute_both_energies,
            compute_metric_energy,
            compute_area_energy,
            prepare_metric_data,
        )

        vertices, faces = simple_quad_mesh
        uv = jnp.array(simple_quad_uv)
        faces_jax = jnp.array(faces)

        k_rings = [
            np.array([1, 2]),
            np.array([0, 3]),
            np.array([0, 3]),
            np.array([1, 2]),
        ]
        target_distances = [
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        ]

        neighbors, targets, mask = prepare_metric_data(k_rings, target_distances)
        neighbors_jax = jnp.array(neighbors)
        targets_jax = jnp.array(targets)
        mask_jax = jnp.array(mask)

        original_areas = jnp.array([0.5, 0.5])

        # Combined call
        J_d_combined, J_a_combined = compute_both_energies(
            uv, neighbors_jax, targets_jax, mask_jax, faces_jax, original_areas
        )

        # Individual calls
        J_d_single = compute_metric_energy(uv, neighbors_jax, targets_jax, mask_jax)
        J_a_single = compute_area_energy(uv, faces_jax, original_areas)

        assert np.isclose(float(J_d_combined), float(J_d_single))
        assert np.isclose(float(J_a_combined), float(J_a_single))


# =============================================================================
# Tier 3: Distance Function Tests
# =============================================================================


class TestGetRingsByLevel:
    """Tests for get_rings_by_level and get_rings_by_level_fast functions."""

    def test_rings_separated_by_level(self, simple_quad_mesh):
        """Test that rings are correctly separated by level."""
        from autoflatten.flatten.distance import get_rings_by_level

        vertices, faces = simple_quad_mesh
        n_vertices = len(vertices)

        rings = get_rings_by_level(faces, n_vertices, k=2)

        # Each vertex should have k=2 levels
        assert len(rings) == n_vertices
        for v_rings in rings:
            assert len(v_rings) == 2

    def test_1_ring_contains_direct_neighbors(self, simple_quad_mesh):
        """Test that level 0 (1-ring) contains only direct neighbors."""
        from autoflatten.flatten.distance import get_rings_by_level
        import igl

        vertices, faces = simple_quad_mesh
        n_vertices = len(vertices)

        rings = get_rings_by_level(faces, n_vertices, k=1)

        # Compare with igl adjacency list
        adj = igl.adjacency_list(faces.astype(np.int64))

        for v in range(n_vertices):
            ring_0 = set(rings[v][0])
            adj_set = set(adj[v])
            assert ring_0 == adj_set

    def test_2_ring_excludes_1_ring_vertices(self, triangle_strip_mesh):
        """Test that level 1 (2-ring) doesn't include 1-ring vertices."""
        from autoflatten.flatten.distance import get_rings_by_level

        vertices, faces = triangle_strip_mesh
        n_vertices = len(vertices)

        rings = get_rings_by_level(faces, n_vertices, k=2)

        for v in range(n_vertices):
            ring_0 = set(rings[v][0])
            ring_1 = set(rings[v][1])
            # No overlap between levels
            assert ring_0.isdisjoint(ring_1)

    def test_fast_version_matches_slow(self, simple_quad_mesh):
        """Test that Numba-accelerated version matches pure Python."""
        from autoflatten.flatten.distance import (
            get_rings_by_level,
            get_rings_by_level_fast,
        )

        vertices, faces = simple_quad_mesh
        n_vertices = len(vertices)

        rings_slow = get_rings_by_level(faces, n_vertices, k=2)
        rings_fast = get_rings_by_level_fast(faces, n_vertices, k=2)

        for v in range(n_vertices):
            for level in range(2):
                slow_set = set(rings_slow[v][level])
                fast_set = set(rings_fast[v][level])
                assert slow_set == fast_set


class TestProjectToTangentPlane:
    """Tests for project_to_tangent_plane function."""

    def test_projects_to_2d(self):
        """Test that projection produces correct 2D coordinates."""
        from autoflatten.flatten.distance import project_to_tangent_plane

        center = np.array([0.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])  # z-axis normal
        neighbors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

        xy = project_to_tangent_plane(center, normal, neighbors)

        assert xy.shape == (3, 2), f"Expected shape (3, 2), got {xy.shape}"

        # With z-axis normal, points in xy-plane project to themselves
        # Check that distances from origin are preserved
        expected_distances = np.array([1.0, 1.0, 1.0])
        actual_distances = np.linalg.norm(xy, axis=1)
        np.testing.assert_allclose(
            actual_distances,
            expected_distances,
            rtol=1e-5,
            err_msg="Projection should preserve distances for coplanar points",
        )

    def test_preserves_distances_on_plane(self):
        """Test that distances are approximately preserved for coplanar points."""
        from autoflatten.flatten.distance import project_to_tangent_plane

        center = np.array([0.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])

        # Points on the xy-plane
        neighbors = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]])

        xy = project_to_tangent_plane(center, normal, neighbors)

        # Check distances from origin
        dist_3d = np.linalg.norm(neighbors, axis=1)
        dist_2d = np.linalg.norm(xy, axis=1)

        np.testing.assert_allclose(dist_2d, dist_3d, rtol=1e-5)

    def test_handles_different_normal_orientations(self):
        """Test projection with normals in different directions."""
        from autoflatten.flatten.distance import project_to_tangent_plane

        center = np.array([0.0, 0.0, 0.0])
        neighbors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        # X-axis normal
        normal_x = np.array([1.0, 0.0, 0.0])
        xy_x = project_to_tangent_plane(center, normal_x, neighbors)
        assert xy_x.shape == (2, 2)

        # Y-axis normal
        normal_y = np.array([0.0, 1.0, 0.0])
        xy_y = project_to_tangent_plane(center, normal_y, neighbors)
        assert xy_y.shape == (2, 2)

    def test_single_neighbor(self):
        """Test with a single neighbor."""
        from autoflatten.flatten.distance import project_to_tangent_plane

        center = np.array([0.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        neighbors = np.array([[1.0, 0.0, 0.0]])

        xy = project_to_tangent_plane(center, normal, neighbors)

        assert xy.shape == (1, 2)
        assert np.isclose(np.linalg.norm(xy[0]), 1.0)


class TestComputeVertexNormals:
    """Tests for compute_vertex_normals function."""

    def test_unit_normals(self, simple_quad_mesh):
        """Test that vertex normals are unit vectors."""
        from autoflatten.flatten.distance import compute_vertex_normals

        vertices, faces = simple_quad_mesh

        normals = compute_vertex_normals(vertices.astype(np.float64), faces)

        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_flat_surface_has_uniform_normals(self, simple_quad_mesh):
        """Test that a flat surface has uniform normals."""
        from autoflatten.flatten.distance import compute_vertex_normals

        vertices, faces = simple_quad_mesh

        normals = compute_vertex_normals(vertices.astype(np.float64), faces)

        # All normals should be parallel (same direction)
        # Check that all normals are close to the first one
        for i in range(1, len(normals)):
            dot = np.dot(normals[0], normals[i])
            assert np.isclose(abs(dot), 1.0, rtol=1e-5)


# =============================================================================
# Additional Tests: compute_total_energy
# =============================================================================


class TestComputeTotalEnergy:
    """Tests for compute_total_energy function (weighted combination)."""

    def test_returns_three_values(self, simple_quad_mesh, simple_quad_uv):
        """Test that compute_total_energy returns total, J_d, and J_a."""
        from autoflatten.flatten.energy import compute_total_energy, prepare_metric_data

        vertices, faces = simple_quad_mesh
        uv = jnp.array(simple_quad_uv)
        faces_jax = jnp.array(faces)

        # Simple k-ring data
        k_rings = [
            np.array([1, 2]),
            np.array([0, 3]),
            np.array([0, 3]),
            np.array([1, 2]),
        ]
        target_distances = [
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        ]

        neighbors, targets, mask = prepare_metric_data(k_rings, target_distances)
        neighbors_jax = jnp.array(neighbors)
        targets_jax = jnp.array(targets)
        mask_jax = jnp.array(mask)

        original_areas = jnp.array([0.5, 0.5])

        total, J_d, J_a = compute_total_energy(
            uv, neighbors_jax, targets_jax, mask_jax, faces_jax, original_areas
        )

        assert np.isfinite(float(total)), f"Total energy is not finite: {total}"
        assert np.isfinite(float(J_d)), f"J_d is not finite: {J_d}"
        assert np.isfinite(float(J_a)), f"J_a is not finite: {J_a}"

    def test_weighted_sum_formula(self, simple_quad_mesh, simple_quad_uv):
        """Test that total = lambda_d * J_d + lambda_a * J_a."""
        from autoflatten.flatten.energy import compute_total_energy, prepare_metric_data

        vertices, faces = simple_quad_mesh
        uv = jnp.array(simple_quad_uv)
        faces_jax = jnp.array(faces)

        k_rings = [
            np.array([1, 2]),
            np.array([0, 3]),
            np.array([0, 3]),
            np.array([1, 2]),
        ]
        target_distances = [
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        ]

        neighbors, targets, mask = prepare_metric_data(k_rings, target_distances)
        neighbors_jax = jnp.array(neighbors)
        targets_jax = jnp.array(targets)
        mask_jax = jnp.array(mask)

        original_areas = jnp.array([0.5, 0.5])

        lambda_d, lambda_a = 2.0, 3.0
        total, J_d, J_a = compute_total_energy(
            uv,
            neighbors_jax,
            targets_jax,
            mask_jax,
            faces_jax,
            original_areas,
            lambda_d=lambda_d,
            lambda_a=lambda_a,
        )

        expected_total = lambda_d * float(J_d) + lambda_a * float(J_a)
        assert np.isclose(float(total), expected_total, rtol=1e-5), (
            f"Expected total={expected_total}, got {float(total)}"
        )

    def test_default_weights_are_one(self, simple_quad_mesh, simple_quad_uv):
        """Test that default lambda_d=1 and lambda_a=1."""
        from autoflatten.flatten.energy import compute_total_energy, prepare_metric_data

        vertices, faces = simple_quad_mesh
        uv = jnp.array(simple_quad_uv)
        faces_jax = jnp.array(faces)

        k_rings = [
            np.array([1, 2]),
            np.array([0, 3]),
            np.array([0, 3]),
            np.array([1, 2]),
        ]
        target_distances = [
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        ]

        neighbors, targets, mask = prepare_metric_data(k_rings, target_distances)
        neighbors_jax = jnp.array(neighbors)
        targets_jax = jnp.array(targets)
        mask_jax = jnp.array(mask)

        original_areas = jnp.array([0.5, 0.5])

        # Default weights
        total, J_d, J_a = compute_total_energy(
            uv, neighbors_jax, targets_jax, mask_jax, faces_jax, original_areas
        )

        # With lambda_d=1, lambda_a=1, total should equal J_d + J_a
        expected = float(J_d) + float(J_a)
        assert np.isclose(float(total), expected, rtol=1e-5), (
            f"With default weights, total should be J_d + J_a = {expected}, got {float(total)}"
        )


# =============================================================================
# Additional Tests: _limited_dijkstra (Python fallback)
# =============================================================================


class TestLimitedDijkstra:
    """Tests for _limited_dijkstra Python fallback function."""

    def test_computes_correct_distances(self, simple_quad_mesh):
        """Test that Dijkstra computes correct shortest path distances."""
        from autoflatten.flatten.distance import _limited_dijkstra, build_mesh_graph

        vertices, faces = simple_quad_mesh
        graph = build_mesh_graph(vertices, faces)

        # From vertex 0, find distances to vertices 1, 2, 3
        k_ring = np.array([1, 2, 3])
        distances = _limited_dijkstra(0, k_ring, graph, correction=1.0)

        assert len(distances) == 3, f"Expected 3 distances, got {len(distances)}"

        # Vertex 0 is at (0,0,0), vertex 1 at (1,0,0), vertex 2 at (0,1,0)
        # Direct distance 0->1 = 1.0, 0->2 = 1.0
        assert np.isclose(distances[0], 1.0, rtol=1e-5), (
            f"Distance to vertex 1 should be 1.0, got {distances[0]}"
        )
        assert np.isclose(distances[1], 1.0, rtol=1e-5), (
            f"Distance to vertex 2 should be 1.0, got {distances[1]}"
        )

    def test_empty_k_ring_returns_empty(self, simple_quad_mesh):
        """Test that empty k_ring returns empty array."""
        from autoflatten.flatten.distance import _limited_dijkstra, build_mesh_graph

        vertices, faces = simple_quad_mesh
        graph = build_mesh_graph(vertices, faces)

        k_ring = np.array([], dtype=np.int64)
        distances = _limited_dijkstra(0, k_ring, graph, correction=1.0)

        assert len(distances) == 0, (
            f"Expected empty array, got {len(distances)} elements"
        )

    def test_correction_factor_applied(self, simple_quad_mesh):
        """Test that correction factor is applied as a divisor.

        The correction factor divides the raw graph distance: result = dist / correction.
        This is used to convert graph distances to geodesic estimates.
        """
        from autoflatten.flatten.distance import _limited_dijkstra, build_mesh_graph

        vertices, faces = simple_quad_mesh
        graph = build_mesh_graph(vertices, faces)

        k_ring = np.array([1])

        # Without correction (correction=1.0)
        dist_no_corr = _limited_dijkstra(0, k_ring, graph, correction=1.0)
        # With correction factor of 2.0 (divides distance by 2)
        dist_with_corr = _limited_dijkstra(0, k_ring, graph, correction=2.0)

        # Correction is a divisor: result = dist / correction
        expected = dist_no_corr[0] / 2.0
        assert np.isclose(dist_with_corr[0], expected, rtol=1e-5), (
            f"Expected corrected distance {expected}, got {dist_with_corr[0]}"
        )


# =============================================================================
# Additional Tests: Invalid Input Handling
# =============================================================================


class TestInvalidInputHandling:
    """Tests for proper handling of invalid or edge-case inputs."""

    def test_prepare_metric_data_empty_input_raises(self):
        """Test prepare_metric_data raises ValueError on empty k_rings list.

        Empty input is invalid because max() fails on empty sequence.
        This is expected behavior - the algorithm needs at least one vertex.
        """
        from autoflatten.flatten.energy import prepare_metric_data

        k_rings = []
        target_distances = []

        # Error message varies by Python version:
        # - Python 3.10: "max() arg is an empty sequence"
        # - Python 3.12: "max() iterable argument is empty"
        with pytest.raises(ValueError, match="empty"):
            prepare_metric_data(k_rings, target_distances)

    def test_build_mesh_graph_single_triangle(self):
        """Test building graph from single triangle mesh."""
        from autoflatten.flatten.distance import build_mesh_graph

        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])

        graph = build_mesh_graph(vertices, faces)

        assert graph.shape == (3, 3), f"Expected (3,3) graph, got {graph.shape}"
        # Each vertex should be connected to 2 others
        assert graph.nnz == 6, (  # 3 edges * 2 (symmetric)
            f"Expected 6 non-zero entries (symmetric), got {graph.nnz}"
        )

    def test_count_boundary_loops_no_shared_edges(self):
        """Test boundary counting with two separate triangles (no shared edges)."""
        # Two separate triangles (not connected)
        faces = np.array([[0, 1, 2], [3, 4, 5]])

        n_loops, loops = count_boundary_loops(faces)

        # Each triangle has its own boundary loop
        assert n_loops == 2, f"Expected 2 boundary loops, got {n_loops}"

    def test_get_vertices_with_negative_area_empty_faces(self):
        """Test negative area detection with empty faces array."""
        from autoflatten.flatten.energy import get_vertices_with_negative_area

        uv = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        faces = jnp.array([]).reshape(0, 3).astype(jnp.int32)

        has_neg = get_vertices_with_negative_area(uv, faces)

        # No faces means no negative areas
        assert not any(has_neg), "Expected no negative area flags with empty faces"

    def test_compute_2d_areas_single_face(self):
        """Test 2D area computation with single face."""
        uv = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        faces = jnp.array([[0, 1, 2]])

        total_area, neg_area = compute_2d_areas(uv, faces)

        assert np.isclose(float(total_area), 0.5), (
            f"Expected area 0.5 for unit right triangle, got {float(total_area)}"
        )
        assert np.isclose(float(neg_area), 0.0), (
            f"Expected no negative area for CCW triangle, got {float(neg_area)}"
        )


# =============================================================================
# Tests Using mesh_with_flipped_triangle Fixture
# =============================================================================


class TestFlippedTriangleMesh:
    """Tests using the mesh_with_flipped_triangle fixture.

    These tests verify behavior when a mesh contains triangles with
    incorrect winding order (CW instead of CCW), which results in
    negative signed area in 2D projections.
    """

    def test_negative_area_detection(self, mesh_with_flipped_triangle):
        """Test that get_vertices_with_negative_area detects flipped triangle."""
        from autoflatten.flatten.energy import get_vertices_with_negative_area

        vertices, faces = mesh_with_flipped_triangle
        # Use XY coordinates as UV
        uv = jnp.array(vertices[:, :2])
        faces_jax = jnp.array(faces)

        has_neg = get_vertices_with_negative_area(uv, faces_jax)

        # Vertices 1, 2, 3 are in the flipped triangle [1, 2, 3]
        # Vertex 0 is only in the CCW triangle [0, 1, 2]
        assert not has_neg[0], "Vertex 0 should not be flagged (only in CCW triangle)"
        assert has_neg[1], "Vertex 1 should be flagged (in flipped triangle)"
        assert has_neg[2], "Vertex 2 should be flagged (in flipped triangle)"
        assert has_neg[3], "Vertex 3 should be flagged (in flipped triangle)"

    def test_compute_2d_areas_with_flip(self, mesh_with_flipped_triangle):
        """Test that compute_2d_areas reports negative area for flipped triangle."""
        vertices, faces = mesh_with_flipped_triangle
        uv = jnp.array(vertices[:, :2])
        faces_jax = jnp.array(faces)

        total_area, neg_area = compute_2d_areas(uv, faces_jax)

        # One triangle has positive area (~0.5), one has negative (~0.5)
        # Total absolute area should be ~1.0, negative area should be ~0.5
        assert float(neg_area) > 0.4, (
            f"Expected significant negative area from flipped triangle, got {float(neg_area)}"
        )

    def test_area_energy_higher_with_flip(
        self, mesh_with_flipped_triangle, simple_quad_mesh
    ):
        """Test that area energy is higher for mesh with flipped triangle."""
        from autoflatten.flatten.energy import compute_area_energy

        # Good mesh (no flips)
        vertices_good, faces_good = simple_quad_mesh
        uv_good = jnp.array(vertices_good[:, :2])
        faces_good_jax = jnp.array(faces_good)
        # Compute original 3D areas (both triangles have area 0.5)
        original_areas_good = jnp.array([0.5, 0.5])

        # Bad mesh (one flip)
        vertices_bad, faces_bad = mesh_with_flipped_triangle
        uv_bad = jnp.array(vertices_bad[:, :2])
        faces_bad_jax = jnp.array(faces_bad)
        original_areas_bad = jnp.array([0.5, 0.5])

        energy_good = compute_area_energy(uv_good, faces_good_jax, original_areas_good)
        energy_bad = compute_area_energy(uv_bad, faces_bad_jax, original_areas_bad)

        assert float(energy_bad) > float(energy_good), (
            f"Flipped mesh should have higher area energy: "
            f"good={float(energy_good):.4f}, bad={float(energy_bad):.4f}"
        )

    def test_build_mesh_graph_with_flip(self, mesh_with_flipped_triangle):
        """Test that build_mesh_graph works correctly regardless of winding."""
        from autoflatten.flatten.distance import build_mesh_graph

        vertices, faces = mesh_with_flipped_triangle
        graph = build_mesh_graph(vertices, faces)

        # Graph should be 4x4 symmetric matrix
        assert graph.shape == (4, 4)
        assert (graph != graph.T).nnz == 0, "Graph should be symmetric"

        # All edges should have positive weights (distances)
        assert (graph.data > 0).all(), "All edge weights should be positive"


# =============================================================================
# Tests Using mesh_with_isolated_vertex Fixture
# =============================================================================


class TestIsolatedVertexMesh:
    """Tests using the mesh_with_isolated_vertex fixture.

    These tests verify behavior when a mesh contains vertices that are
    not connected to any faces, which can occur in real data after
    certain mesh operations.
    """

    def test_k_ring_connected_vertices(self, mesh_with_isolated_vertex):
        """Test k-ring for connected vertices in mesh with isolated vertex.

        Note: get_k_ring uses igl.adjacency_list which only includes vertices
        that appear in faces. For meshes with isolated vertices, only query
        k-rings for connected vertices (indices 0 to n_faces_vertices-1).
        """
        from autoflatten.flatten.distance import get_k_ring

        vertices, faces = mesh_with_isolated_vertex
        # Only get k-rings for vertices that appear in faces (0, 1, 2)
        n_connected = 3  # Vertices 0, 1, 2 are in the single triangle
        k_rings = get_k_ring(faces, n_connected, k=1)

        # Each vertex in the triangle has 2 neighbors
        assert len(k_rings[0]) == 2, "Vertex 0 should have 2 neighbors"
        assert len(k_rings[1]) == 2, "Vertex 1 should have 2 neighbors"
        assert len(k_rings[2]) == 2, "Vertex 2 should have 2 neighbors"

    def test_build_mesh_graph_isolated_vertex(self, mesh_with_isolated_vertex):
        """Test that build_mesh_graph handles isolated vertices correctly."""
        from autoflatten.flatten.distance import build_mesh_graph

        vertices, faces = mesh_with_isolated_vertex
        graph = build_mesh_graph(vertices, faces)

        # Graph should be 4x4 (includes isolated vertex)
        assert graph.shape == (4, 4)

        # Isolated vertex (index 3) should have no edges
        assert graph[3, :].nnz == 0, "Isolated vertex should have no outgoing edges"
        assert graph[:, 3].nnz == 0, "Isolated vertex should have no incoming edges"

        # Connected vertices should have edges
        assert graph[0, 1] > 0, "Edge 0-1 should exist"
        assert graph[0, 2] > 0, "Edge 0-2 should exist"
        assert graph[1, 2] > 0, "Edge 1-2 should exist"

    def test_prepare_smoothing_data_isolated_vertex(self, mesh_with_isolated_vertex):
        """Test that prepare_smoothing_data handles isolated vertices."""
        from autoflatten.flatten.energy import prepare_smoothing_data

        vertices, faces = mesh_with_isolated_vertex

        neighbors, mask, counts = prepare_smoothing_data(faces, len(vertices))

        # Isolated vertex should have count=1 (degree 0 + 1)
        assert counts[3] == 1, (
            f"Isolated vertex should have count=1 (degree+1), got {counts[3]}"
        )

        # Isolated vertex should have all-False mask (no valid neighbors)
        assert not mask[3].any(), (
            "Isolated vertex should have no valid neighbors in mask"
        )

    def test_get_vertices_with_negative_area_ignores_isolated(
        self, mesh_with_isolated_vertex
    ):
        """Test that negative area detection ignores isolated vertices."""
        from autoflatten.flatten.energy import get_vertices_with_negative_area

        vertices, faces = mesh_with_isolated_vertex
        uv = jnp.array(vertices[:, :2])
        faces_jax = jnp.array(faces)

        has_neg = get_vertices_with_negative_area(uv, faces_jax)

        # No triangles are flipped, so no vertices should be flagged
        assert not any(has_neg[:3]), "Connected vertices should not be flagged"
        # Isolated vertex is not in any face, so should not be flagged
        assert not has_neg[3], "Isolated vertex should not be flagged"

    def test_compute_2d_areas_ignores_isolated(self, mesh_with_isolated_vertex):
        """Test that 2D area computation ignores isolated vertices."""
        vertices, faces = mesh_with_isolated_vertex
        uv = jnp.array(vertices[:, :2])
        faces_jax = jnp.array(faces)

        total_area, neg_area = compute_2d_areas(uv, faces_jax)

        # Single CCW triangle with vertices at (0,0), (1,0), (0.5, 1)
        # Area = 0.5 * |1 * 1 - 0 * 0.5| = 0.5
        assert np.isclose(float(total_area), 0.5, atol=0.01), (
            f"Expected area ~0.5 for single triangle, got {float(total_area)}"
        )
        assert np.isclose(float(neg_area), 0.0), (
            f"Expected no negative area, got {float(neg_area)}"
        )
