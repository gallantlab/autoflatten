#!/usr/bin/env python
"""
Automatic Surface Flattening Pipeline

This script implements a pipeline for automatic flattening of cortical surfaces
using medial wall and cut vertices from fsaverage mapped to a target subject.

CLI Structure:
    autoflatten /path/to/subject     - Full pipeline: project + flatten (default)
    autoflatten project /path/to/subject  - Projection only: creates patch file
    autoflatten flatten PATCH_FILE   - Flattening only: flattens existing patch
    autoflatten plot-projection PATCH     - Visualize projection (3D surface with cuts)
    autoflatten plot-flatmap FLAT_PATCH   - Visualize flattened surface
"""

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from autoflatten.config import fsaverage_cut_template
from autoflatten.core import (
    ensure_continuous_cuts,
    fill_holes_in_patch,
    map_cuts_to_subject,
    refine_cuts_with_geodesic,
)
from autoflatten.freesurfer import create_patch_file, load_surface
from autoflatten.logging import restore_logging, setup_logging
from autoflatten.template import identify_surface_components
from autoflatten.utils import load_json
from autoflatten.viz import plot_patch, plot_projection


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

    # Check if mri_label2label is available (needed for projection)
    if shutil.which("mri_label2label") is None:
        print("Error: mri_label2label not found in PATH.")
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
            fs_version = result.stdout.strip()
            print(f"FreeSurfer version: {fs_version}")

            # Extract version number using regex
            version_match = re.search(r"(\d+\.\d+(?:\.\d+)?)", fs_version)
            if version_match:
                from packaging.version import Version

                version_number = version_match.group(1)
                try:
                    if Version(version_number) < Version("7.0"):
                        raise ValueError(
                            f"FreeSurfer version {version_number} is below 7.0. "
                            "This tool requires FreeSurfer 7.0 or higher."
                        )
                except ValueError as e:
                    print(f"Error: {str(e)}")
                    return False, env_vars
        else:
            print(
                "Warning: Could not determine FreeSurfer version, "
                "but commands are available."
            )
    except FileNotFoundError:
        print(
            "Error: mri_info not found in PATH. "
            "FreeSurfer may not be properly installed."
        )
        return False, env_vars

    return True, env_vars


def run_projection(
    subject_dir,
    hemi,
    output_dir,
    template_file=None,
    overwrite=False,
    refine_geodesic=True,
    verbose=True,
):
    """
    Run the projection phase to create a patch file.

    Parameters
    ----------
    subject_dir : str
        Path to the FreeSurfer subject directory
    hemi : str
        Hemisphere ('lh' or 'rh')
    output_dir : str
        Directory to save output files
    template_file : str, optional
        Path to the template file containing cut definitions
    overwrite : bool
        Whether to overwrite existing files
    refine_geodesic : bool
        Whether to refine cuts using geodesic shortest paths
    verbose : bool
        Whether to print progress messages

    Returns
    -------
    str
        Path to the created patch file
    """
    subject = Path(subject_dir).name
    patch_file = os.path.join(output_dir, f"{hemi}.autoflatten.patch.3d")
    log_base = os.path.join(output_dir, f"{hemi}.autoflatten.projection")

    if os.path.exists(patch_file) and not overwrite:
        if verbose:
            print(
                f"Patch file {patch_file} already exists, skipping (use --overwrite to force)"
            )
        return patch_file

    # Setup logging - all print() output goes to both console and log file
    # Log file will be created at log_base + ".log"
    original_stdout, log_file = setup_logging(log_base, verbose=verbose)
    start_time = time.time()

    try:
        # Header
        print("Autoflatten Projection Log")
        print("=" * 60)
        print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print(f"Subject: {subject}")
        print(f"Hemisphere: {hemi}")
        print(f"Subject directory: {subject_dir}")
        print(f"Output directory: {output_dir}")

        # Get cuts template
        print("\nTemplate Loading")
        print("-" * 16)
        if template_file is None:
            template_file = fsaverage_cut_template
        print(f"Template file: {template_file}")
        template_data = load_json(template_file)

        vertex_dict = {}
        prefix = f"{hemi}_"
        for key, value in template_data.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                vertex_dict[new_key] = np.array(value)
                print(f"  {new_key}: {len(value)} vertices")

        # Map cuts to target subject
        print("\nCut Mapping (mri_label2label)")
        print("-" * 29)
        step_start = time.time()
        vertex_dict_mapped = map_cuts_to_subject(vertex_dict, subject, hemi)
        step_elapsed = time.time() - step_start
        print(f"Mapping completed in {step_elapsed:.2f}s")
        for key, vertices in vertex_dict_mapped.items():
            orig_count = len(vertex_dict.get(key, []))
            mapped_count = len(vertices)
            print(f"  {key}: {orig_count} -> {mapped_count} vertices")

        # Ensure cuts are continuous
        print("\nContinuity Fixing")
        print("-" * 17)
        step_start = time.time()
        vertex_dict_fixed = ensure_continuous_cuts(
            vertex_dict_mapped.copy(), subject, hemi
        )
        step_elapsed = time.time() - step_start
        print(f"Continuity fixing completed in {step_elapsed:.2f}s")
        for key, vertices in vertex_dict_fixed.items():
            pre_count = len(vertex_dict_mapped.get(key, []))
            post_count = len(vertices)
            if post_count != pre_count:
                print(
                    f"  {key}: {pre_count} -> {post_count} vertices "
                    f"(added {post_count - pre_count})"
                )
            else:
                print(f"  {key}: {post_count} vertices (no changes)")

        # Optionally refine with geodesic paths
        if refine_geodesic:
            print("\nGeodesic Refinement")
            print("-" * 19)
            step_start = time.time()
            vertex_dict_refined = refine_cuts_with_geodesic(
                vertex_dict_fixed,
                subject,
                hemi,
                medial_wall_vertices=vertex_dict_fixed.get("mwall"),
            )
            step_elapsed = time.time() - step_start
            print(f"Geodesic refinement completed in {step_elapsed:.2f}s")
            for key, vertices in vertex_dict_refined.items():
                pre_count = len(vertex_dict_fixed.get(key, []))
                post_count = len(vertices)
                if post_count != pre_count:
                    print(f"  {key}: {pre_count} -> {post_count} vertices")
                else:
                    print(f"  {key}: {post_count} vertices (unchanged)")
            vertex_dict_fixed = vertex_dict_refined
        else:
            print("\nGeodesic Refinement")
            print("-" * 19)
            print("Skipped (--no-refine-geodesic)")

        # Get subject surface data
        print("\nPatch Creation")
        print("-" * 14)
        pts, polys = load_surface(subject, "inflated", hemi)
        print(f"Surface loaded: {len(pts)} vertices, {len(polys)} faces")

        # Fill holes in patch (detect and exclude hole boundary vertices)
        print("\nHole Filling")
        print("-" * 12)
        step_start = time.time()
        excluded_vertices = set()
        for vertices in vertex_dict_fixed.values():
            excluded_vertices.update(int(v) for v in vertices)

        hole_vertices = fill_holes_in_patch(polys, excluded_vertices)
        step_elapsed = time.time() - step_start
        if hole_vertices:
            print(f"Filled {len(hole_vertices)} hole boundary vertices")
            vertex_dict_fixed["_hole_fill"] = np.array(list(hole_vertices))
        else:
            print("No holes detected")
        print(f"Hole filling completed in {step_elapsed:.2f}s")

        # Create patch file
        print("\nPatch File Creation")
        print("-" * 19)
        step_start = time.time()
        patch_file, patch_vertices = create_patch_file(
            patch_file, pts, polys, vertex_dict_fixed
        )
        step_elapsed = time.time() - step_start

        # Log patch statistics
        total_excluded = sum(len(v) for v in vertex_dict_fixed.values())
        n_patch_vertices = len(patch_vertices)
        print(f"Patch creation completed in {step_elapsed:.2f}s")
        print(f"  Total surface vertices: {len(pts)}")
        print(f"  Excluded vertices (cuts + medial wall): {total_excluded}")
        print(f"  Patch vertices: {n_patch_vertices}")
        print(f"  Output file: {patch_file}")

        # Generate projection plot
        print("\nGenerating Projection Plot")
        print("-" * 26)
        plot_output = os.path.join(output_dir, f"{hemi}.autoflatten.patch.png")
        try:
            plot_projection(
                patch_path=patch_file,
                subject_dir=subject_dir,
                output_path=plot_output,
                overwrite=overwrite,
            )
        except (ValueError, FileNotFoundError, OSError) as e:
            print(f"Warning: Failed to generate projection plot: {e}")
            traceback.print_exc()
            print("Continuing without projection plot...")
        except Exception:
            print("Unexpected error while generating projection plot:")
            traceback.print_exc()
            raise

        print("\nRESULT")
        print("-" * 6)
        print(f"Patch file created: {patch_file}")
        if os.path.exists(plot_output):
            print(f"Projection plot: {plot_output}")
        print(f"Log file: {log_base}.log")

        # Footer
        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        restore_logging(original_stdout, log_file)

    return patch_file


def run_flatten_backend(
    patch_path,
    surface_path,
    output_path,
    backend_name=None,
    verbose=True,
    **backend_kwargs,
):
    """
    Run flattening using the specified backend.

    Parameters
    ----------
    patch_path : str
        Path to the input patch file
    surface_path : str
        Path to the base surface file
    output_path : str
        Path for the output flat patch file
    backend_name : str, optional
        Backend name ('pyflatten' or 'freesurfer'). If None, uses default.
    verbose : bool
        Whether to print progress messages
    **backend_kwargs
        Additional arguments passed to the backend

    Returns
    -------
    str
        Path to the output flat patch file
    """
    from autoflatten.backends import get_backend, get_default_backend

    if backend_name:
        backend = get_backend(backend_name)
    else:
        backend = get_default_backend()

    if verbose:
        print(f"Using {backend.name} backend for flattening")

    return backend.flatten(
        patch_path=patch_path,
        surface_path=surface_path,
        output_path=output_path,
        verbose=verbose,
        **backend_kwargs,
    )


def process_hemisphere(
    subject_dir,
    hemi,
    output_dir,
    template_file=None,
    run_flatten=True,
    overwrite=False,
    refine_geodesic=True,
    backend=None,
    verbose=True,
    run_plot=True,
    base_surface=None,
    **backend_kwargs,
):
    """
    Process a single hemisphere through the full pipeline.

    Parameters
    ----------
    subject_dir : str
        Path to the FreeSurfer subject directory
    hemi : str
        Hemisphere ('lh' or 'rh')
    output_dir : str
        Directory to save output files
    template_file : str, optional
        Path to template file
    run_flatten : bool
        Whether to run flattening after projection
    overwrite : bool
        Whether to overwrite existing files
    refine_geodesic : bool
        Whether to refine cuts with geodesic paths
    backend : str, optional
        Backend name for flattening
    verbose : bool
        Print progress messages
    run_plot : bool
        Whether to generate PNG plot after flattening
    base_surface : str, optional
        Path to base surface file for flattening. If None, auto-detects
        {hemi}.fiducial or {hemi}.smoothwm in the subject's surf/ directory.
    **backend_kwargs
        Additional arguments for the backend

    Returns
    -------
    dict
        Results including patch_file, flat_file, and plot_file paths
    """
    subject = Path(subject_dir).name
    if verbose:
        print(f"\nProcessing {hemi} hemisphere for subject {subject}")
    start_time = time.time()

    result = {
        "subject": subject,
        "hemi": hemi,
    }

    # Run projection
    patch_file = run_projection(
        subject_dir=subject_dir,
        hemi=hemi,
        output_dir=output_dir,
        template_file=template_file,
        overwrite=overwrite,
        refine_geodesic=refine_geodesic,
        verbose=verbose,
    )
    result["patch_file"] = patch_file

    # Run flattening if requested
    if run_flatten:
        # Determine surface path
        if base_surface:
            surface_path = base_surface
            if not os.path.exists(surface_path):
                raise FileNotFoundError(
                    f"Base surface not found: {surface_path}\n"
                    "Please provide a valid path with --base-surface."
                )
        else:
            surf_dir = os.path.join(subject_dir, "surf")
            surface_path = os.path.join(surf_dir, f"{hemi}.fiducial")
            if not os.path.exists(surface_path):
                surface_path = os.path.join(surf_dir, f"{hemi}.smoothwm")
            if not os.path.exists(surface_path):
                raise FileNotFoundError(
                    f"Base surface not found. Looked for:\n"
                    f"  - {os.path.join(surf_dir, f'{hemi}.fiducial')}\n"
                    f"  - {os.path.join(surf_dir, f'{hemi}.smoothwm')}\n"
                    "Please provide a valid path with --base-surface."
                )

        # Determine output path
        flat_file = os.path.join(output_dir, f"{hemi}.autoflatten.flat.patch.3d")

        if os.path.exists(flat_file) and not overwrite:
            if verbose:
                print(
                    f"Flat file {flat_file} exists, skipping (use --overwrite to force)"
                )
        else:
            flat_file = run_flatten_backend(
                patch_path=patch_file,
                surface_path=surface_path,
                output_path=flat_file,
                backend_name=backend,
                verbose=verbose,
                **backend_kwargs,
            )

        result["flat_file"] = flat_file

        # Generate PNG plot if requested
        if run_plot:
            plot_file = os.path.join(output_dir, f"{hemi}.autoflatten.flat.patch.png")
            if os.path.exists(plot_file) and not overwrite:
                if verbose:
                    print(
                        f"Plot file {plot_file} exists, skipping "
                        "(use --overwrite to force)"
                    )
            else:
                surf_dir = os.path.join(subject_dir, "surf")
                try:
                    plot_file = plot_patch(
                        flat_file,
                        subject,
                        surf_dir,
                        output_dir=output_dir,
                        surface=f"{hemi}.inflated",
                        overwrite=overwrite,
                    )
                    if verbose:
                        print(f"Generated plot: {plot_file}")
                except Exception as e:
                    print(f"Warning: Failed to generate plot: {e}")
                    plot_file = None
            result["plot_file"] = plot_file

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Completed {hemi} hemisphere in {elapsed_time:.2f} seconds")

    return result


# =============================================================================
# CLI Commands
# =============================================================================


def cmd_default(args):
    """Default command: full pipeline (project + flatten)."""
    return cmd_run_full_pipeline(args)


def cmd_run_full_pipeline(args):
    """Run the full pipeline: projection + flattening."""
    print("Starting Autoflatten Pipeline...")

    # Check FreeSurfer environment (needed for projection)
    fs_check, env_vars = check_freesurfer_environment()
    if not fs_check:
        print("FreeSurfer environment is not properly set up. Exiting.")
        return 1

    # Resolve subject directory
    subject_dir = os.path.abspath(args.subject_dir)
    if not os.path.isdir(subject_dir):
        print(f"Error: Subject directory not found: {subject_dir}")
        return 1

    subject = Path(subject_dir).name
    print(f"Subject: {subject}")
    print(f"Subject directory: {subject_dir}")

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(subject_dir, "surf")
        print(f"Warning: No --output-dir specified. Using: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine hemispheres
    if args.hemispheres == "both":
        hemispheres = ["lh", "rh"]
    else:
        hemispheres = [args.hemispheres]

    print(f"Processing hemispheres: {', '.join(hemispheres)}")

    # Determine cores per hemisphere
    n_cores = args.n_cores
    if args.parallel and len(hemispheres) > 1 and n_cores > 0:
        # Split cores between hemispheres when running in parallel
        n_cores_per_hemi = max(1, n_cores // 2)
        print(
            f"Parallel mode: {n_cores} cores split to {n_cores_per_hemi} per hemisphere"
        )
    else:
        n_cores_per_hemi = n_cores

    # Collect backend kwargs
    backend_kwargs = {}
    if args.backend == "pyflatten":
        backend_kwargs.update(
            {
                "k_ring": args.k_ring,
                "n_neighbors_per_ring": args.n_neighbors,
                "skip_phases": args.skip_phase,
                "skip_spring_smoothing": args.skip_spring_smoothing,
                "skip_neg_area": args.skip_neg_area,
                "config_path": args.pyflatten_config,
                "n_jobs": n_cores_per_hemi,
                "cache_distances": args.debug_save_distances,
                "print_every": args.print_every,
            }
        )
    elif args.backend == "freesurfer":
        backend_kwargs.update(
            {
                "seed": args.seed
                if args.seed is not None
                else random.randint(0, 99999),
                "threads": args.nthreads,
                "distances": tuple(args.distances) if args.distances else (15, 80),
                "n": args.n_iterations,
                "dilate": args.dilate,
                "passes": args.passes,
                "tol": args.tol,
                "debug": args.debug,
            }
        )

    results = {}

    # Process hemispheres
    if args.parallel and len(hemispheres) > 1:
        print("Processing hemispheres in parallel")
        with ProcessPoolExecutor(max_workers=len(hemispheres)) as executor:
            future_to_hemi = {
                executor.submit(
                    process_hemisphere,
                    subject_dir,
                    hemi,
                    output_dir,
                    args.template_file,
                    True,  # run_flatten
                    args.overwrite,
                    not args.no_refine_geodesic,
                    args.backend,
                    True,  # verbose
                    True,  # run_plot
                    None,  # base_surface (auto-detect)
                    **{**backend_kwargs, "tqdm_position": idx},
                ): hemi
                for idx, hemi in enumerate(hemispheres)
            }
            for future in future_to_hemi:
                hemi = future_to_hemi[future]
                try:
                    results[hemi] = future.result()
                except Exception as e:
                    print(f"Error processing {hemi} hemisphere: {e}")
                    traceback.print_exc()
    else:
        for hemi in hemispheres:
            try:
                results[hemi] = process_hemisphere(
                    subject_dir,
                    hemi,
                    output_dir,
                    args.template_file,
                    True,  # run_flatten
                    args.overwrite,
                    not args.no_refine_geodesic,
                    args.backend,
                    True,  # verbose
                    True,  # run_plot
                    None,  # base_surface (auto-detect)
                    **backend_kwargs,
                )
            except Exception:
                print(f"Error processing {hemi} hemisphere:")
                traceback.print_exc()
                return 1

    # Print summary
    print("\nSummary:")
    for hemi in hemispheres:
        if hemi in results:
            print(f"{hemi.upper()} Hemisphere:")
            print(f"  Patch file: {results[hemi].get('patch_file', 'Not created')}")
            print(f"  Flat file: {results[hemi].get('flat_file', 'Not created')}")
            print(f"  Plot file: {results[hemi].get('plot_file', 'Not created')}")

    return 0


def cmd_project(args):
    """Run projection only (create patch file)."""
    print("Starting Autoflatten Projection...")

    # Check FreeSurfer environment
    fs_check, env_vars = check_freesurfer_environment()
    if not fs_check:
        print("FreeSurfer environment is not properly set up. Exiting.")
        return 1

    subject_dir = os.path.abspath(args.subject_dir)
    if not os.path.isdir(subject_dir):
        print(f"Error: Subject directory not found: {subject_dir}")
        return 1

    subject = Path(subject_dir).name
    print(f"Subject: {subject}")

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(subject_dir, "surf")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine hemispheres
    if args.hemispheres == "both":
        hemispheres = ["lh", "rh"]
    else:
        hemispheres = [args.hemispheres]

    results = {}
    for hemi in hemispheres:
        try:
            patch_file = run_projection(
                subject_dir=subject_dir,
                hemi=hemi,
                output_dir=output_dir,
                template_file=args.template_file,
                overwrite=args.overwrite,
                refine_geodesic=not args.no_refine_geodesic,
                verbose=True,
            )
            results[hemi] = patch_file
            print(f"{hemi.upper()}: Created {patch_file}")
        except Exception:
            print(f"Error processing {hemi} hemisphere:")
            traceback.print_exc()
            return 1

    return 0


def cmd_flatten(args):
    """Run flattening only on an existing patch file."""
    print("Starting Autoflatten Flattening...")

    patch_path = os.path.abspath(args.patch_file)
    if not os.path.exists(patch_path):
        print(f"Error: Patch file not found: {patch_path}")
        return 1

    # Auto-detect base surface if not specified
    if args.base_surface:
        surface_path = args.base_surface
        if not os.path.exists(surface_path):
            print(f"Error: Base surface not found: {surface_path}")
            return 1
    else:
        from autoflatten.backends import find_base_surface

        surface_path = find_base_surface(patch_path)
        if surface_path is None:
            # Determine hemisphere for informative error message
            patch_name = os.path.basename(patch_path)
            hemi = "lh" if patch_name.startswith("lh.") else "rh"
            patch_dir = os.path.dirname(patch_path)
            print(
                f"Error: Could not auto-detect base surface. Looked for:\n"
                f"  - {os.path.join(patch_dir, f'{hemi}.fiducial')}\n"
                f"  - {os.path.join(patch_dir, f'{hemi}.smoothwm')}\n"
                "Please specify --base-surface."
            )
            return 1
        print(f"Auto-detected base surface: {surface_path}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default: replace .patch.3d with .flat.patch.3d
        base = patch_path.replace(".patch.3d", "")
        output_path = f"{base}.flat.patch.3d"

    print(f"Input patch: {patch_path}")
    print(f"Base surface: {surface_path}")
    print(f"Output: {output_path}")

    # Collect backend kwargs
    backend_kwargs = {}
    if args.backend == "pyflatten":
        backend_kwargs.update(
            {
                "k_ring": args.k_ring,
                "n_neighbors_per_ring": args.n_neighbors,
                "skip_phases": args.skip_phase,
                "skip_spring_smoothing": args.skip_spring_smoothing,
                "skip_neg_area": args.skip_neg_area,
                "config_path": args.pyflatten_config,
                "n_jobs": args.n_cores,
                "cache_distances": args.debug_save_distances,
                "print_every": args.print_every,
            }
        )
    elif args.backend == "freesurfer":
        # For FreeSurfer backend, we need subject info
        backend_kwargs.update(
            {
                "subject": args.subject,
                "seed": args.seed
                if args.seed is not None
                else random.randint(0, 99999),
                "threads": args.nthreads,
                "distances": tuple(args.distances) if args.distances else (15, 80),
                "n": args.n_iterations,
                "dilate": args.dilate,
                "passes": args.passes,
                "tol": args.tol,
                "debug": args.debug,
            }
        )

    try:
        result = run_flatten_backend(
            patch_path=patch_path,
            surface_path=surface_path,
            output_path=output_path,
            backend_name=args.backend,
            verbose=True,
            **backend_kwargs,
        )
        print(f"Successfully created: {result}")
        return 0
    except Exception as e:
        print(f"Error during flattening: {e}")
        traceback.print_exc()
        return 1


def cmd_plot_flatmap(args):
    """Plot a flat patch file."""
    print("Starting Autoflatten Flatmap Plotting...")

    flat_patch_file = args.flat_patch
    if not os.path.exists(flat_patch_file):
        print(f"Error: Flat patch file not found: {flat_patch_file}")
        return 1

    # Determine hemisphere from filename
    basename = os.path.basename(flat_patch_file)
    if basename.startswith("lh."):
        hemi = "lh"
    elif basename.startswith("rh."):
        hemi = "rh"
    else:
        print(f"Error: Could not determine hemisphere from filename: {basename}")
        return 1

    # Determine subject directory with auto-detection
    if args.subject_dir:
        subject_dir = args.subject_dir
    else:
        # Auto-detect: if patch is in surf/ directory, use parent as subject_dir
        patch_dir = Path(flat_patch_file).resolve().parent
        if patch_dir.name == "surf":
            subject_dir = str(patch_dir)
        elif os.path.isfile(os.path.join(patch_dir, f"{hemi}.inflated")):
            # patch_dir itself contains the surface files
            subject_dir = str(patch_dir)
        else:
            print(
                f"Error: Cannot auto-detect subject directory from {flat_patch_file}. "
                "Please provide --subject-dir argument."
            )
            return 1

    # Derive subject name from directory path
    subject_dir_path = Path(subject_dir).resolve()
    if subject_dir_path.name == "surf":
        subject = subject_dir_path.parent.name
    else:
        subject = subject_dir_path.name

    # Verify surface exists
    surface_file = os.path.join(subject_dir, f"{hemi}.inflated")
    if not os.path.exists(surface_file):
        print(f"Error: Inflated surface not found: {surface_file}")
        return 1

    # Determine output directory
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if not output_dir:
            output_dir = os.path.dirname(os.path.abspath(flat_patch_file))
    else:
        output_dir = os.path.dirname(os.path.abspath(flat_patch_file))

    print(f"Flat patch file: {flat_patch_file}")
    print(f"Subject: {subject}")

    overwrite = getattr(args, "overwrite", False)

    try:
        result = plot_patch(
            flat_patch_file,
            subject,
            subject_dir,
            output_dir=output_dir,
            surface=f"{hemi}.inflated",
            overwrite=overwrite,
        )
        if args.output:
            final_output = os.path.abspath(args.output)
            if result != final_output:
                os.rename(result, final_output)
                print(f"Successfully saved plot: {final_output}")
                return 0
        print(f"Successfully saved plot: {result}")
        return 0
    except Exception as e:
        print(f"Failed to generate plot: {e}")
        traceback.print_exc()
        return 1


def cmd_plot_projection(args):
    """Plot a projection patch file showing the surface with cuts highlighted."""
    print("Starting Autoflatten Projection Plotting...")

    patch_file = args.patch
    if not os.path.exists(patch_file):
        print(f"Error: Patch file not found: {patch_file}")
        return 1

    # Determine subject directory
    subject_dir = args.subject_dir if args.subject_dir else None

    # Determine output path
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        # Default: same directory as patch, with .png extension
        output_path = patch_file.replace(".3d", ".png")

    print(f"Patch file: {patch_file}")
    if subject_dir:
        print(f"Subject directory: {subject_dir}")
    print(f"Output: {output_path}")

    overwrite = getattr(args, "overwrite", False)

    try:
        result = plot_projection(
            patch_path=patch_file,
            subject_dir=subject_dir,
            output_path=output_path,
            overwrite=overwrite,
        )
        print(f"Successfully saved projection plot: {result}")
        return 0
    except FileNotFoundError as e:
        print(f"Failed to generate projection plot (file not found): {e}")
        traceback.print_exc()
        return 1
    except (ValueError, OSError) as e:
        print(f"Failed to generate projection plot: {e}")
        traceback.print_exc()
        return 1


# =============================================================================
# Argument Parsers
# =============================================================================


def add_projection_args(parser):
    """Add projection-related arguments to a parser."""
    parser.add_argument(
        "--template-file",
        help="Path to custom JSON template file defining cuts",
    )
    parser.add_argument(
        "--no-refine-geodesic",
        action="store_true",
        help="Disable geodesic refinement of projected cuts",
    )


def add_backend_args(parser):
    """Add backend selection arguments to a parser."""
    parser.add_argument(
        "--backend",
        choices=["pyflatten", "freesurfer"],
        default="pyflatten",
        help="Flattening backend (default: pyflatten)",
    )


def add_pyflatten_args(parser):
    """Add pyflatten-specific arguments to a parser."""
    group = parser.add_argument_group("pyflatten options")
    group.add_argument(
        "--k-ring",
        type=int,
        default=7,
        help="K-ring neighborhood size (default: 7)",
    )
    group.add_argument(
        "--n-neighbors",
        type=int,
        default=12,
        help="Neighbors per ring for angular sampling (default: 12)",
    )
    group.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print progress every N iterations (default: 1 = every iteration)",
    )
    group.add_argument(
        "--skip-phase",
        type=str,
        nargs="*",
        choices=[
            "epoch_1",
            "epoch_2",
            "epoch_3",
        ],
        help="Phases to skip (epoch_1, epoch_2, epoch_3)",
    )
    group.add_argument(
        "--skip-spring-smoothing",
        action="store_true",
        help="Skip final spring smoothing",
    )
    group.add_argument(
        "--skip-neg-area",
        action="store_true",
        help="Skip negative area removal phase",
    )
    group.add_argument(
        "--pyflatten-config",
        help="Path to JSON configuration file for pyflatten",
    )
    group.add_argument(
        "--n-cores",
        type=int,
        default=-1,
        help="Number of CPU cores to use (-1 = all cores). "
        "When combined with --parallel, cores are split between hemispheres.",
    )
    group.add_argument(
        "--debug-save-distances",
        action="store_true",
        help="Save k-ring distances to cache file for debugging",
    )


def add_freesurfer_args(parser):
    """Add FreeSurfer mris_flatten arguments to a parser."""
    group = parser.add_argument_group("freesurfer options")
    group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for mris_flatten",
    )
    group.add_argument(
        "--nthreads",
        type=int,
        default=1,
        help="Number of threads for mris_flatten",
    )
    group.add_argument(
        "--distances",
        type=int,
        nargs=2,
        default=[15, 80],
        metavar=("DIST1", "DIST2"),
        help="Distance parameters for mris_flatten",
    )
    group.add_argument(
        "--n-iterations",
        type=int,
        default=200,
        help="Maximum iterations for mris_flatten",
    )
    group.add_argument(
        "--dilate",
        type=int,
        default=1,
        help="Number of dilations for mris_flatten",
    )
    group.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Number of passes for mris_flatten",
    )
    group.add_argument(
        "--tol",
        type=float,
        default=0.005,
        help="Tolerance for mris_flatten",
    )
    group.add_argument(
        "--debug",
        action="store_true",
        help="Keep temporary files for debugging",
    )


def add_common_args(parser):
    """Add common arguments to a parser."""
    parser.add_argument(
        "--output-dir",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--hemispheres",
        choices=["lh", "rh", "both"],
        default="both",
        help="Hemispheres to process (default: both)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process hemispheres in parallel",
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="autoflatten: Automatic Cortical Surface Flattening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (projection + flattening) using pyflatten backend:
  autoflatten /path/to/subjects/sub-01

  # Projection only (create patch file):
  autoflatten project /path/to/subjects/sub-01

  # Flatten an existing patch file:
  autoflatten flatten lh.autoflatten.patch.3d

  # Use FreeSurfer backend instead of pyflatten:
  autoflatten /path/to/subjects/sub-01 --backend freesurfer

  # Plot a flattened surface:
  autoflatten plot-flatmap lh.autoflatten.flat.patch.3d --subject-dir /path/to/subject/surf
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Hidden 'run' subcommand for default full pipeline behavior
    # This is inserted automatically when user runs: autoflatten /path/to/subject
    # Users can also explicitly call: autoflatten run /path/to/subject
    parser_run = subparsers.add_parser(
        "run", help="Run full pipeline (projection + flattening)"
    )
    parser_run.add_argument(
        "subject_dir",
        help="Path to FreeSurfer subject directory",
    )
    add_common_args(parser_run)
    add_projection_args(parser_run)
    add_backend_args(parser_run)
    add_pyflatten_args(parser_run)
    add_freesurfer_args(parser_run)

    # 'project' subcommand
    parser_project = subparsers.add_parser(
        "project",
        help="Run projection only (create patch file)",
    )
    parser_project.add_argument(
        "subject_dir",
        help="Path to FreeSurfer subject directory",
    )
    add_common_args(parser_project)
    add_projection_args(parser_project)
    parser_project.set_defaults(func=cmd_project)

    # 'flatten' subcommand
    parser_flatten = subparsers.add_parser(
        "flatten",
        help="Flatten an existing patch file",
    )
    parser_flatten.add_argument(
        "patch_file",
        help="Path to the input patch file",
    )
    parser_flatten.add_argument(
        "--base-surface",
        help=(
            "Path to base surface file. "
            "By default, auto-detects {hemi}.fiducial or {hemi}.smoothwm "
            "in the same directory as the patch file."
        ),
    )
    parser_flatten.add_argument(
        "-o",
        "--output",
        help="Output path for flat patch file",
    )
    parser_flatten.add_argument(
        "--subject",
        help="FreeSurfer subject ID (needed for freesurfer backend)",
    )
    add_backend_args(parser_flatten)
    add_pyflatten_args(parser_flatten)
    add_freesurfer_args(parser_flatten)
    parser_flatten.set_defaults(func=cmd_flatten)

    # 'plot-flatmap' subcommand
    parser_plot_flatmap = subparsers.add_parser(
        "plot-flatmap",
        help="Plot a flat patch file",
    )
    parser_plot_flatmap.add_argument(
        "flat_patch",
        help="Path to the flat patch file",
    )
    parser_plot_flatmap.add_argument(
        "--subject-dir",
        help="Path to subject's surf directory (auto-detected if patch is in surf/)",
    )
    parser_plot_flatmap.add_argument(
        "-o",
        "--output",
        help="Output path for the PNG image",
    )
    parser_plot_flatmap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )
    parser_plot_flatmap.set_defaults(func=cmd_plot_flatmap)

    # 'plot-projection' subcommand
    parser_plot_projection = subparsers.add_parser(
        "plot-projection",
        help="Plot a projection patch file (3D surface with cuts)",
    )
    parser_plot_projection.add_argument(
        "patch",
        help="Path to the projection patch file (e.g., lh.autoflatten.patch.3d)",
    )
    parser_plot_projection.add_argument(
        "--subject-dir",
        help="Path to FreeSurfer subject directory (auto-detected if patch is in surf/)",
    )
    parser_plot_projection.add_argument(
        "-o",
        "--output",
        help="Output path for the PNG image",
    )
    parser_plot_projection.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )
    parser_plot_projection.set_defaults(func=cmd_plot_projection)

    # Handle default case: autoflatten /path/to/subject [options]
    # Insert 'run' subcommand when first arg looks like a path
    known_commands = {"project", "flatten", "plot-flatmap", "plot-projection", "run"}
    if (
        len(sys.argv) > 1
        and sys.argv[1] not in known_commands
        and not sys.argv[1].startswith("-")
    ):
        sys.argv.insert(1, "run")

    args = parser.parse_args()

    # Handle command dispatch
    if args.command == "project":
        return cmd_project(args)
    elif args.command == "flatten":
        return cmd_flatten(args)
    elif args.command == "plot-flatmap":
        return cmd_plot_flatmap(args)
    elif args.command == "plot-projection":
        return cmd_plot_projection(args)
    elif args.command == "run":
        return cmd_run_full_pipeline(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
