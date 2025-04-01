#!/usr/bin/env python
"""
Automatic Surface Flattening Pipeline

This script implements a pipeline for automatic flattening of cortical surfaces
using medial wall and cut vertices from fsaverage mapped to a target subject.
"""

import argparse
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import cortex

# Import necessary functions from your helpers module
from autoflatten.core import (
    ensure_continuous_cuts,
    map_cuts_to_subject,
)
from autoflatten.freesurfer import create_patch_file, run_mris_flatten
from autoflatten.template import identify_surface_components


def check_freesurfer_environment():
    """
    Check if FreeSurfer environment is properly set up.

    Returns
    -------
    bool
        True if FreeSurfer environment is properly set up, False otherwise
    dict
        Environment variables including FREESURFER_HOME and SUBJECTS_DIR
    """
    # Check if FREESURFER_HOME and SUBJECTS_DIR are set
    freesurfer_home = os.environ.get("FREESURFER_HOME")
    subjects_dir = os.environ.get("SUBJECTS_DIR")

    env_vars = {"FREESURFER_HOME": freesurfer_home, "SUBJECTS_DIR": subjects_dir}

    if not freesurfer_home:
        print("Error: FREESURFER_HOME environment variable is not set.")
        return False, env_vars

    if not subjects_dir:
        print("Error: SUBJECTS_DIR environment variable is not set.")
        return False, env_vars

    # Check if mris_flatten is available in PATH
    if shutil.which("mris_flatten") is None:
        print("Error: mris_flatten not found in PATH.")
        return False, env_vars

    # Try to get FreeSurfer version to verify installation
    try:
        result = subprocess.run(
            ["mri_info", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            print(f"FreeSurfer version: {result.stdout.strip()}")
        else:
            print(
                "Warning: Could not determine FreeSurfer version, but commands are available."
            )
    except FileNotFoundError:
        print(
            "Error: mri_info not found in PATH. FreeSurfer may not be properly installed."
        )
        return False, env_vars

    return True, env_vars


def process_hemisphere(
    subject, hemi, output_dir, run_flatten=False, iterations=None, overwrite=False
):
    """
    Process a single hemisphere through the flattening pipeline.

    Parameters
    ----------
    subject : str
        FreeSurfer subject identifier
    hemi : str
        Hemisphere ('lh' or 'rh')
    output_dir : str
        Directory to save output files
    run_flatten : bool, optional
        Whether to run mris_flatten
    iterations : int or None, optional
        Number of iterations for flattening. If None, use mris_flatten default.
    overwrite : bool, optional
        Whether to overwrite existing files

    Returns
    -------
    dict
        Information about the processed hemisphere
    """
    print(f"\nProcessing {hemi} hemisphere for subject {subject}")
    start_time = time.time()

    # Create patch file name
    patch_file = os.path.join(output_dir, f"{hemi}.autoflatten.patch.3d")

    # Check if patch file exists and whether to overwrite
    if os.path.exists(patch_file) and not overwrite:
        print(
            f"Patch file {patch_file} already exists, skipping. Use --overwrite to force regeneration."
        )

        result = {
            "subject": subject,
            "hemi": hemi,
            "patch_file": patch_file,
            "patch_vertices": None,
            "vertex_dict": None,
        }

        # If flattening is requested and patch file exists, still run flattening
        if run_flatten:
            print(f"Running mris_flatten for {subject} {hemi}")
            flat_file = run_mris_flatten(
                subject, hemi, patch_file, output_dir, iterations, overwrite
            )
            result["flat_file"] = flat_file

        return result

    # Step 1: Get medial wall and cuts from fsaverage
    print(f"Getting medial wall and cuts from fsaverage for {hemi}")
    vertex_dict = identify_surface_components("fsaverage", hemi)

    # Step 2: Map cuts to target subject
    print(f"Mapping cuts from fsaverage to {subject} for {hemi}")
    vertex_dict_mapped = map_cuts_to_subject(vertex_dict, subject, hemi)

    # Step 3: Ensure cuts are continuous in target subject
    print(f"Ensuring continuous cuts for {subject} {hemi}")
    vertex_dict_fixed = ensure_continuous_cuts(vertex_dict_mapped.copy(), subject, hemi)

    # Get subject surface data
    pts, polys = cortex.db.get_surf(subject, "inflated", hemisphere=hemi)

    # Create patch file
    print(f"Creating patch file for {subject} {hemi}")
    patch_file, patch_vertices = create_patch_file(
        patch_file, pts, polys, vertex_dict_fixed
    )

    result = {
        "subject": subject,
        "hemi": hemi,
        "patch_file": patch_file,
        "patch_vertices": patch_vertices,
        "vertex_dict": vertex_dict_fixed,
    }

    # Run flattening if requested
    if run_flatten:
        print(f"Running mris_flatten for {subject} {hemi}")
        flat_file = run_mris_flatten(
            subject, hemi, patch_file, output_dir, iterations, overwrite
        )
        result["flat_file"] = flat_file

    elapsed_time = time.time() - start_time
    print(f"Completed {hemi} hemisphere in {elapsed_time:.2f} seconds")

    return result


def main():
    """Main function to run the automatic flattening pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Automatic Surface Flattening Pipeline"
    )
    parser.add_argument("subject", help="FreeSurfer subject identifier")
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory to save output files (default: subject's FreeSurfer surf directory)"
        ),
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Run mris_flatten after creating patch files",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Process hemispheres in parallel"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of iterations for mris_flatten (if not specified, uses mris_flatten default)",
    )
    parser.add_argument(
        "--hemispheres",
        type=str,
        choices=["lh", "rh", "both"],
        default="both",
        help="Hemispheres to process: left (lh), right (rh), or both (default)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )

    args = parser.parse_args()

    # Check FreeSurfer environment
    fs_check, env_vars = check_freesurfer_environment()
    if not fs_check:
        print("FreeSurfer environment is not properly set up. Exiting.")
        return 1

    # Set default output directory to the subject's surf directory if not specified
    if args.output_dir:
        output_dir = args.output_dir
    else:
        subjects_dir = env_vars["SUBJECTS_DIR"]
        output_dir = os.path.join(subjects_dir, args.subject, "surf")

    # Verify that the output directory exists
    if not os.path.isdir(output_dir):
        print(f"Output directory {output_dir} does not exist. Creating it...")
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Failed to create output directory: {e}")
            return 1

    print(f"Using output directory: {output_dir}")

    # Determine which hemispheres to process
    if args.hemispheres == "both":
        hemispheres = ["lh", "rh"]
    else:
        hemispheres = [args.hemispheres]

    print(f"Processing hemispheres: {', '.join(hemispheres)}")

    results = {}

    # Process hemispheres (parallel or sequential)
    if args.parallel and len(hemispheres) > 1:
        print("Processing hemispheres in parallel")
        with ProcessPoolExecutor(max_workers=len(hemispheres)) as executor:
            future_to_hemi = {
                executor.submit(
                    process_hemisphere,
                    args.subject,
                    hemi,
                    output_dir,
                    args.flatten,
                    args.iterations,
                    args.overwrite,
                ): hemi
                for hemi in hemispheres
            }

            for future in future_to_hemi:
                hemi = future_to_hemi[future]
                try:
                    results[hemi] = future.result()
                except Exception as e:
                    print(f"Error processing {hemi} hemisphere: {e}")
    else:
        if args.parallel and len(hemispheres) == 1:
            print(
                "Parallel processing requested but only one hemisphere selected, using sequential processing"
            )
        else:
            print("Processing hemispheres sequentially")

        for hemi in hemispheres:
            try:
                results[hemi] = process_hemisphere(
                    args.subject,
                    hemi,
                    output_dir,
                    args.flatten,
                    args.iterations,
                    args.overwrite,
                )
            except Exception as e:
                print(f"Error processing {hemi} hemisphere: {e}")
                print(f"Traceback: {e.__traceback__}")

    # Print summary
    print("\nSummary:")
    for hemi in hemispheres:
        if hemi in results:
            patch_file = results[hemi].get("patch_file", "Not created")
            flat_file = results[hemi].get("flat_file", "Not created")

            print(f"{hemi.upper()} Hemisphere:")
            print(f"  Patch file: {patch_file}")
            if args.flatten:
                print(f"  Flat file: {flat_file}")

    return 0


if __name__ == "__main__":
    exit(main())
