#!/usr/bin/env python
"""
Example: Using geodesic refinement for projected cuts

This example demonstrates how to use the geodesic refinement feature to optimize
projected cuts by replacing them with shortest geodesic paths between endpoints.

The geodesic refinement step can help reduce distortion during flattening by:
1. Removing "wiggling" artifacts from surface registration
2. Creating more direct paths on the target surface geometry
3. Reducing the total length of cuts while preserving connectivity
"""

import numpy as np
from autoflatten.core import (
    map_cuts_to_subject,
    ensure_continuous_cuts,
    refine_cuts_with_geodesic,
)
from autoflatten.utils import load_json
from autoflatten.config import fsaverage_cut_template


# Example 1: Basic usage with default fsaverage template
def example_basic_refinement():
    """Basic example of geodesic refinement."""
    subject = "subject_id"  # Replace with your subject ID
    hemi = "rh"

    # Load template
    print("Loading template...")
    template_data = load_json(fsaverage_cut_template)

    # Extract hemisphere-specific data
    vertex_dict = {}
    prefix = f"{hemi}_"
    for key, value in template_data.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            vertex_dict[new_key] = np.array(value)

    # Map cuts to target subject
    print(f"Mapping cuts to {subject}...")
    vertex_dict_mapped = map_cuts_to_subject(vertex_dict, subject, hemi)

    # Ensure continuity
    print("Ensuring continuous cuts...")
    vertex_dict_fixed = ensure_continuous_cuts(vertex_dict_mapped.copy(), subject, hemi)

    # Apply geodesic refinement
    print("Refining cuts with geodesic shortest paths...")
    vertex_dict_refined = refine_cuts_with_geodesic(
        vertex_dict_fixed,
        subject,
        hemi,
        medial_wall_vertices=vertex_dict_fixed.get("mwall"),
    )

    # Compare before and after
    print("\n=== Comparison ===")
    for cut_name in ["cut1", "cut2", "cut3", "cut4", "cut5"]:
        if cut_name in vertex_dict_fixed and cut_name in vertex_dict_refined:
            before = len(vertex_dict_fixed[cut_name])
            after = len(vertex_dict_refined[cut_name])
            reduction = 100 * (1 - after / before)
            print(
                f"{cut_name}: {before} → {after} vertices ({reduction:.1f}% reduction)"
            )

    return vertex_dict_refined


# Example 2: Using with CLI
def example_cli_usage():
    """Example CLI command with geodesic refinement."""
    command = """
    # Without geodesic refinement (default)
    autoflatten run subject_id --hemispheres rh

    # With geodesic refinement
    autoflatten run subject_id --hemispheres rh --refine-geodesic

    # Full example with multiple options
    autoflatten run subject_id \\
        --hemispheres both \\
        --refine-geodesic \\
        --nthreads 4 \\
        --passes 2 \\
        --seed 42
    """
    print(command)


# Example 3: Comparing refinement impact
def example_compare_refinement(subject, hemi):
    """
    Compare cuts before and after geodesic refinement.

    Parameters
    ----------
    subject : str
        Subject ID
    hemi : str
        Hemisphere ('lh' or 'rh')
    """
    from autoflatten.freesurfer import load_surface
    import networkx as nx

    # Load template and map to subject
    template_data = load_json(fsaverage_cut_template)
    vertex_dict = {}
    prefix = f"{hemi}_"
    for key, value in template_data.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            vertex_dict[new_key] = np.array(value)

    vertex_dict_mapped = map_cuts_to_subject(vertex_dict, subject, hemi)
    vertex_dict_fixed = ensure_continuous_cuts(vertex_dict_mapped.copy(), subject, hemi)

    # Apply refinement
    vertex_dict_refined = refine_cuts_with_geodesic(
        vertex_dict_fixed.copy(),
        subject,
        hemi,
        medial_wall_vertices=vertex_dict_fixed.get("mwall"),
    )

    # Load surface for path length calculation
    pts, polys = load_surface(subject, "fiducial", hemi)

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(len(pts)))
    for triangle in polys:
        for i in range(3):
            v1 = triangle[i]
            for j in range(i + 1, 3):
                v2 = triangle[j]
                weight = np.linalg.norm(pts[v1] - pts[v2])
                G.add_edge(v1, v2, weight=weight)

    # Calculate total path lengths
    print("\n=== Path Length Comparison ===")
    for cut_name in ["cut1", "cut2", "cut3", "cut4", "cut5"]:
        if cut_name not in vertex_dict_fixed:
            continue

        # Before refinement
        vertices_before = vertex_dict_fixed[cut_name]
        length_before = calculate_path_length(G, vertices_before)

        # After refinement
        vertices_after = vertex_dict_refined[cut_name]
        length_after = calculate_path_length(G, vertices_after)

        improvement = 100 * (1 - length_after / length_before)
        print(f"{cut_name}:")
        print(f"  Before: {len(vertices_before)} vertices, length {length_before:.2f}")
        print(f"  After:  {len(vertices_after)} vertices, length {length_after:.2f}")
        print(f"  Length reduction: {improvement:.1f}%")


def calculate_path_length(G, vertices):
    """Calculate total length of a path through vertices."""
    total_length = 0
    for i in range(len(vertices) - 1):
        v1, v2 = vertices[i], vertices[i + 1]
        if G.has_edge(v1, v2):
            total_length += G[v1][v2]["weight"]
    return total_length


if __name__ == "__main__":
    print("=" * 60)
    print("Geodesic Refinement Examples")
    print("=" * 60)

    print("\n1. Basic Usage:")
    print("-" * 60)
    # Uncomment to run:
    # vertex_dict = example_basic_refinement()

    print("\n2. CLI Usage Examples:")
    print("-" * 60)
    example_cli_usage()

    print("\n3. Detailed Comparison:")
    print("-" * 60)
    print("To run comparison:")
    print("  python example_geodesic_refinement.py <subject_id> <hemi>")

    import sys

    if len(sys.argv) == 3:
        subject = sys.argv[1]
        hemi = sys.argv[2]
        example_compare_refinement(subject, hemi)
