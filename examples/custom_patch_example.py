#!/usr/bin/env python
"""
Example: Creating a custom patch with arbitrary cuts

This example demonstrates how to use autoflatten with flexible patch
configurations that are not limited to the standard 5-cut template.

You can create patches with:
- Arbitrary number of cuts (not limited to 5)
- Custom cut names (e.g., 'occipital', 'temporal', 'custom1')
- Regions without a medial wall
- Single isolated patches

The key is that the vertex_dict can have any keys except 'mwall' is
reserved for medial wall vertices.
"""

import numpy as np


def example_1_custom_cuts():
    """
    Example 1: Create a patch with custom cut names
    
    Instead of the standard calcarine, medial1-3, temporal cuts,
    you can define your own arbitrary cuts.
    """
    print("\n" + "=" * 60)
    print("Example 1: Custom cut names")
    print("=" * 60)
    
    # Define a patch with custom cut names
    # (These would normally come from your own template or manual selection)
    vertex_dict = {
        "mwall": np.array([100, 101, 102]),  # Medial wall vertices
        "occipital_cut": np.array([1, 2, 3, 4]),  # Custom cut 1
        "ventral_cut": np.array([10, 11, 12]),  # Custom cut 2
        "anterior_cut": np.array([20, 21, 22]),  # Custom cut 3
    }
    
    print("Vertex dict keys:", list(vertex_dict.keys()))
    print("This will work with ensure_continuous_cuts and refine_cuts_with_geodesic")
    print("because they now dynamically handle all non-'mwall' keys")


def example_2_single_patch():
    """
    Example 2: Create a single isolated patch (no medial wall)
    
    This could represent just the occipital pole, temporal lobe, etc.
    """
    print("\n" + "=" * 60)
    print("Example 2: Single isolated patch")
    print("=" * 60)
    
    # Define a single patch region with boundary cuts
    vertex_dict = {
        "mwall": np.array([]),  # No medial wall
        "boundary": np.array([1, 2, 3, 4, 5]),  # Single boundary cut
    }
    
    print("Vertex dict keys:", list(vertex_dict.keys()))
    print("Empty medial wall is allowed - creates an isolated patch")


def example_3_arbitrary_number_of_cuts():
    """
    Example 3: Arbitrary number of cuts (not limited to 5)
    
    The system now supports any number of cuts.
    """
    print("\n" + "=" * 60)
    print("Example 3: Arbitrary number of cuts")
    print("=" * 60)
    
    # Create a patch with 7 cuts
    vertex_dict = {
        "mwall": np.array([100, 101]),
        "cut1": np.array([1, 2]),
        "cut2": np.array([3, 4]),
        "cut3": np.array([5, 6]),
        "cut4": np.array([7, 8]),
        "cut5": np.array([9, 10]),
        "cut6": np.array([11, 12]),
        "cut7": np.array([13, 14]),
    }
    
    print("Vertex dict keys:", list(vertex_dict.keys()))
    print(f"Number of cuts: {len(vertex_dict) - 1}")  # -1 for mwall
    print("No hardcoded limit on number of cuts")


def example_4_create_custom_template():
    """
    Example 4: Creating a custom template JSON file
    
    You can create a template file for any patch configuration that
    can then be applied to multiple subjects.
    """
    print("\n" + "=" * 60)
    print("Example 4: Creating a custom template")
    print("=" * 60)
    
    # Create template for both hemispheres
    template_dict = {
        # Left hemisphere with custom patch
        "lh_mwall": [100, 101, 102],  # Example vertices
        "lh_occipital_boundary": [1, 2, 3, 4, 5],
        "lh_temporal_boundary": [10, 11, 12, 13],
        
        # Right hemisphere with custom patch
        "rh_mwall": [100, 101, 102],
        "rh_occipital_boundary": [1, 2, 3, 4, 5],
        "rh_temporal_boundary": [10, 11, 12, 13],
    }
    
    print("Template structure:")
    for key in template_dict.keys():
        print(f"  {key}: {len(template_dict[key])} vertices")
    
    print("\nTo save: save_json('custom_patch_template.json', template_dict)")
    print("(requires: from autoflatten.utils import save_json)")


def example_5_validate_topology():
    """
    Example 5: Validate patch topology before flattening
    
    Use the new validate_patch_topology function to check if your
    custom patch has valid disk topology.
    """
    print("\n" + "=" * 60)
    print("Example 5: Validate patch topology")
    print("=" * 60)
    
    print("To validate topology before flattening:")
    print("```python")
    print("from autoflatten import validate_patch_topology")
    print("")
    print("is_valid, issues, info = validate_patch_topology(")
    print("    vertex_dict, subject='your_subject', hemi='lh'")
    print(")")
    print("if is_valid:")
    print("    print('Patch has valid disk topology!')")
    print("else:")
    print("    print('Issues:', issues)")
    print("```")


def main():
    """Run all examples"""
    print("\n")
    print("=" * 60)
    print("AUTOFLATTEN: Flexible Patch Configuration Examples")
    print("=" * 60)
    print("\nThis demonstrates how autoflatten now supports:")
    print("  - Arbitrary number of cuts (not limited to 5)")
    print("  - Custom cut names (not hardcoded to calcarine, medial1-3, temporal)")
    print("  - Patches without medial wall")
    print("  - Topology validation for custom patches")
    
    example_1_custom_cuts()
    example_2_single_patch()
    example_3_arbitrary_number_of_cuts()
    example_4_create_custom_template()
    example_5_validate_topology()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("The key changes that enable flexibility:")
    print("  1. Core functions now extract cut names dynamically from vertex_dict")
    print("  2. Template functions support arbitrary patch configurations")
    print("  3. New validate_patch_topology() function checks topology")
    print("  4. No more hardcoded assumptions about cut names or counts")
    print("\nYou can now create custom patches for specific brain regions!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
