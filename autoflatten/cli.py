#!/usr/bin/env python
"""
Automatic Surface Flattening Pipeline

This script implements a pipeline for automatic flattening of cortical surfaces
using medial wall and cut vertices from fsaverage mapped to a target subject.
It also provides functionality to plot the results.
"""

import argparse
import glob
import os
import random
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import cortex
import numpy as np

from autoflatten.config import fsaverage_cut_template
from autoflatten.core import ensure_continuous_cuts, map_cuts_to_subject
from autoflatten.freesurfer import create_patch_file, load_surface, run_mris_flatten
from autoflatten.template import identify_surface_components
from autoflatten.utils import load_json
from autoflatten.viz import plot_patch


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
            fs_version = result.stdout.strip()
            print(f"FreeSurfer version: {fs_version}")

            # Check if it's stable6 version
            if "stable6" not in fs_version:
                print(
                    "Warning: FreeSurfer version does not appear to be 'stable6'. "
                    "mris_flatten may not work properly with this version."
                )
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
    subject,
    hemi,
    output_dir,
    template_file=None,
    run_flatten=True,
    overwrite=False,
    seed=0,
    threads=32,
    distances=(15, 80),
    n=80,
    dilate=1,
    passes=1,
    extra_params=None,
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
    template_file : str, optional
        Path to the template file containing cut definitions.
        If None, uses the default template.
    run_flatten : bool, optional
        Whether to run mris_flatten (default: True)
    overwrite : bool, optional
        Whether to overwrite existing files
    seed : int
        Random seed value to use with -seed flag for mris_flatten.
    threads : int, optional
        Number of threads to use (default: 32)
    distances : tuple of int, optional
        Distance parameters as a tuple (distance1, distance2) (default: (15, 80))
    n : int, optional
        Maximum number of iterations to run, used with -n flag (default: 80)
    dilate : int, optional
        Number of dilations to perform, used with -dilate flag (default: 1)
    passes : int, optional
        Number of passes for mris_flatten, used with -p flag (default: 1)
    extra_params : dict, optional
        Dictionary of additional parameters to pass to mris_flatten as -key value pairs

    Returns
    -------
    dict
        Information about the processed hemisphere, including the seed used for flattening.
    """
    print(
        f"\nProcessing {hemi} hemisphere for subject {subject} (flattening seed: {seed})"
    )
    start_time = time.time()

    # Create patch file name (deterministic, no seed)
    patch_file = os.path.join(output_dir, f"{hemi}.autoflatten.patch.3d")

    # Check if patch file exists and whether to overwrite
    if os.path.exists(patch_file) and not overwrite:
        print(
            f"Patch file {patch_file} already exists, skipping patch generation. "
            "Use --overwrite to force regeneration."
        )
        result = {
            "subject": subject,
            "hemi": hemi,
            "patch_file": patch_file,
            "patch_vertices": None,
            "vertex_dict": None,
            "seed": seed,
        }

        # If flattening is requested and patch file exists, still run flattening
        if run_flatten:
            print(
                f"Running mris_flatten for {subject} {hemi} with {passes} passes, seed {seed}"
            )
            current_extra_params = extra_params.copy() if extra_params else {}
            current_extra_params["p"] = passes

            flat_file = run_mris_flatten(
                subject,
                hemi,
                patch_file,
                output_dir,
                output_name=None,
                seed=seed,
                threads=threads,
                distances=distances,
                n=n,
                dilate=dilate,
                extra_params=current_extra_params,
                overwrite=overwrite,
            )
            result["flat_file"] = flat_file
            result["passes"] = passes

        return result

    # Step 1: Get cuts template
    if template_file:
        print(f"Loading cuts template from {template_file}")
        template_data = load_json(template_file)
        vertex_dict = {}
        # Extract hemisphere-specific data from the template
        prefix = f"{hemi}_"
        for key, value in template_data.items():
            if key.startswith(prefix):
                # Remove the hemisphere prefix
                new_key = key[len(prefix) :]
                vertex_dict[new_key] = np.array(value)
    else:
        print(f"Using default fsaverage cuts template for {hemi}")
        # Load predefined template
        if os.path.exists(fsaverage_cut_template):
            template_data = load_json(fsaverage_cut_template)
            vertex_dict = {}
            # Extract hemisphere-specific data from the template
            prefix = f"{hemi}_"
            for key, value in template_data.items():
                if key.startswith(prefix):
                    # Remove the hemisphere prefix
                    new_key = key[len(prefix) :]
                    vertex_dict[new_key] = np.array(value)
        else:
            print(f"Default template not found, generating from fsaverage for {hemi}")
            vertex_dict = identify_surface_components("fsaverage", hemi)

    # Step 2: Map cuts to target subject
    print(f"Mapping cuts to {subject} for {hemi}")
    vertex_dict_mapped = map_cuts_to_subject(vertex_dict, subject, hemi)

    # Step 3: Ensure cuts are continuous in target subject
    print(f"Ensuring continuous cuts for {subject} {hemi}")
    vertex_dict_fixed = ensure_continuous_cuts(vertex_dict_mapped.copy(), subject, hemi)

    # Get subject surface data
    pts, polys = load_surface(subject, "inflated", hemi)

    # Create patch file
    print(f"Creating patch file: {patch_file}")
    patch_file, patch_vertices = create_patch_file(
        patch_file, pts, polys, vertex_dict_fixed
    )

    result = {
        "subject": subject,
        "hemi": hemi,
        "patch_file": patch_file,
        "patch_vertices": patch_vertices,
        "vertex_dict": vertex_dict_fixed,
        "seed": seed,
    }

    # Run flattening if requested
    if run_flatten:
        print(
            f"Running mris_flatten for {subject} {hemi} with {passes} passes, seed {seed}"
        )
        current_extra_params = extra_params.copy() if extra_params else {}
        current_extra_params["p"] = passes

        flat_file = run_mris_flatten(
            subject,
            hemi,
            patch_file,
            output_dir,
            output_name=None,
            seed=seed,
            threads=threads,
            distances=distances,
            n=n,
            dilate=dilate,
            extra_params=current_extra_params,
            overwrite=overwrite,
        )
        result["flat_file"] = flat_file
        result["passes"] = passes

    elapsed_time = time.time() - start_time
    print(f"Completed {hemi} hemisphere in {elapsed_time:.2f} seconds")

    return result


def run_flattening(args):
    """Handles the 'run' subcommand to perform flattening."""
    print("Starting Autoflatten Run Pipeline...")

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

    # Parse the flatten-extra parameter if provided
    extra_params = {}
    if args.flatten_extra:
        try:
            for param in args.flatten_extra.split(","):
                if "=" in param:
                    key, value = param.split("=", 1)
                    # Try to convert value to int or float if possible
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            # Keep as string if not a number
                            pass
                    extra_params[key] = value
                else:
                    # If no value provided, set to True (flag parameter)
                    extra_params[param] = True
        except Exception as e:
            print(f"Error parsing flatten-extra parameter: {e}")
            print("Format should be: key1=value1,key2=value2")
            return 1

    # Determine seed to use
    if args.seed is None:
        selected_seed = random.randint(0, 99999)
        print(
            f"No seed provided, using randomly generated seed for flattening: {selected_seed}"
        )
    else:
        selected_seed = args.seed
        print(f"Using provided seed for flattening: {selected_seed}")

    # Determine which hemispheres to process
    if args.hemispheres == "both":
        hemispheres = ["lh", "rh"]
    else:
        hemispheres = [args.hemispheres]

    print(f"Processing hemispheres: {', '.join(hemispheres)}")

    results = {}

    # Set the flatten parameter (default is True, unless --no-flatten is specified)
    run_flatten = not args.no_flatten

    # Calculate threads per hemisphere when running in parallel
    threads_per_hemisphere = args.nthreads
    if args.parallel and len(hemispheres) > 1:
        # Distribute threads evenly, but at least 1 per hemisphere
        threads_per_hemisphere = max(1, args.nthreads // len(hemispheres))
        print(f"Using {threads_per_hemisphere} threads per hemisphere")

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
                    args.template_file,
                    run_flatten,
                    args.overwrite,
                    selected_seed,
                    threads_per_hemisphere,
                    tuple(args.distances),
                    args.n,
                    args.dilate,
                    args.passes,
                    extra_params,
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
                    args.template_file,
                    run_flatten,
                    args.overwrite,
                    selected_seed,
                    args.nthreads,
                    tuple(args.distances),
                    args.n,
                    args.dilate,
                    args.passes,
                    extra_params,
                )
            except Exception:
                print(f"Error processing {hemi} hemisphere:")
                traceback.print_exc()  # This prints the full exception traceback
                return 1

    # Print summary
    print("\nSummary:")
    for hemi in hemispheres:
        if hemi in results:
            patch_file = results[hemi].get("patch_file", "Not created")
            flat_file = results[hemi].get("flat_file", "Not created")
            passes_used = results[hemi].get("passes", args.passes)
            seed_used = results[hemi].get("seed", "Unknown")

            print(f"{hemi.upper()} Hemisphere:")
            print(f"  Patch file: {patch_file}")
            if run_flatten:
                print(
                    f"  Flat file (seed={seed_used}, passes={passes_used}): {flat_file}"
                )
            elif not run_flatten:
                print("  Flattening skipped.")

    return 0


def run_plotting(args):
    """Handles the 'plot' subcommand to generate visualizations."""
    print("Starting Autoflatten Plotting...")

    # Check FreeSurfer environment needed for pycortex potentially
    fs_check, env_vars = check_freesurfer_environment()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        subjects_dir = env_vars.get("SUBJECTS_DIR")
        if not subjects_dir:
            print("Error: SUBJECTS_DIR not found and --output-dir not specified.")
            return 1
        output_dir = os.path.join(subjects_dir, args.subject, "surf")

    if not os.path.isdir(output_dir):
        print(f"Error: Output directory {output_dir} does not exist.")
        return 1

    print(f"Looking for flat patches in: {output_dir} for subject {args.subject}")

    plot_count = 0
    error_count = 0

    for hemi in ["lh", "rh"]:
        print(f"\nProcessing {hemi.upper()} Hemisphere:")
        # This is the patch definition file used to create the flat map
        base_patch_file = os.path.join(output_dir, f"{hemi}.autoflatten.patch.3d")

        if not os.path.exists(base_patch_file):
            print(
                f"  Base patch definition file not found: {base_patch_file}. Skipping {hemi}."
            )
            continue

        # Find all *flat surface files* generated by autoflatten for this hemisphere
        flat_surface_pattern = os.path.join(
            output_dir, f"{hemi}.autoflatten*.flat.patch.3d"
        )
        flat_surface_files = glob.glob(flat_surface_pattern)

        if not flat_surface_files:
            print(
                f"  No autoflatten flat surface files found matching pattern: {flat_surface_pattern}"
            )
            continue

        print(f"  Found {len(flat_surface_files)} flat surface file(s).")
        print(f"  Using base patch definition file: {base_patch_file}")

        for flat_surf_file in flat_surface_files:
            flat_filename_full = os.path.basename(flat_surf_file)
            viz_output_filename = flat_filename_full.replace(
                ".flat.patch.3d", ".flat.png"
            )
            viz_output_file = os.path.join(output_dir, viz_output_filename)

            print(f"  Generating plot for: {flat_filename_full}")
            print(f"    Using patch definition: {os.path.basename(base_patch_file)}")
            print(f"    Output image: {viz_output_file}")

            try:
                # Pass the flat surface file and the original patch definition file
                plot_patch(
                    flat_surf_file,  # The actual flat surface file (*.flat.patch.3d)
                    os.path.dirname(
                        flat_surf_file
                    ),  # Directory of the flat surface file
                    surface=f"{hemi}.inflated",  # The surface to plot on
                )
                print(f"    Successfully saved plot: {viz_output_file}")
                plot_count += 1
            except ImportError as e:
                print(f"    Failed to generate plot for {flat_filename_full}: {e}")
                print("    Make sure pycortex is installed and configured correctly.")
                error_count += 1
                # Stop trying further plots if pycortex is missing
                print("    Aborting further plotting attempts due to import error.")
                return 1  # Indicate failure
            except Exception as e:
                print(f"    Failed to generate plot for {flat_filename_full}: {e}")
                # import traceback
                # traceback.print_exc() # Uncomment for detailed debugging
                error_count += 1

    print("\n--- Plotting Summary ---")
    print(f"Generated {plot_count} plot(s).")
    if error_count > 0:
        print(f"Encountered {error_count} error(s) during plotting.")
        return 1
    else:
        print("Autoflatten Plotting Finished Successfully.")
        return 0


def main():
    """Main function to parse arguments and dispatch subcommands."""
    parser = argparse.ArgumentParser(
        description="Autoflatten: Automatic Surface Flattening and Plotting Pipeline"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Subcommand to run"
    )

    # 'run' subcommand
    parser_run = subparsers.add_parser("run", help="Run the flattening pipeline")
    parser_run.add_argument("subject", help="FreeSurfer subject identifier")
    parser_run.add_argument(
        "--output-dir",
        help=(
            "Directory to save output files (default: subject's FreeSurfer surf directory)"
        ),
    )
    parser_run.add_argument(
        "--no-flatten",
        action="store_true",
        help="Do not run mris_flatten after creating patch files (flattening is done by default)",
    )
    parser_run.add_argument(
        "--template-file",
        help="Path to a custom JSON template file defining cuts (default: uses built-in fsaverage template)",
    )
    parser_run.add_argument(
        "--parallel", action="store_true", help="Process hemispheres in parallel"
    )
    parser_run.add_argument(
        "--hemispheres",
        type=str,
        choices=["lh", "rh", "both"],
        default="both",
        help="Hemispheres to process: left (lh), right (rh), or both (default)",
    )
    parser_run.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser_run.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for mris_flatten. If not provided, a random seed will be generated.",
    )
    parser_run.add_argument(
        "--nthreads",
        type=int,
        default=32,
        help="Number of threads to use for mris_flatten (default: 32)",
    )
    parser_run.add_argument(
        "--distances",
        type=int,
        nargs=2,
        default=[15, 80],
        help="Distance parameters for mris_flatten as two integers (default: 15 80)",
        metavar=("DIST1", "DIST2"),
    )
    parser_run.add_argument(
        "--n-iterations",
        type=int,
        default=80,
        help="Maximum number of iterations for mris_flatten (default: 80)",
        dest="n",
    )
    parser_run.add_argument(
        "--dilate",
        type=int,
        default=1,
        help="Number of dilations for mris_flatten (default: 1)",
    )
    parser_run.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Number of passes for mris_flatten (-p flag) (default: 1)",
    )
    parser_run.add_argument(
        "--flatten-extra",
        type=str,
        help="Additional parameters for mris_flatten in format 'key1=value1,key2=value2'",
    )
    parser_run.set_defaults(func=run_flattening)

    # 'plot' subcommand
    parser_plot = subparsers.add_parser(
        "plot", help="Plot existing autoflatten results"
    )
    parser_plot.add_argument("subject", help="FreeSurfer subject identifier")
    parser_plot.add_argument(
        "--output-dir",
        help=(
            "Directory containing the autoflatten output files (default: $SUBJECTS_DIR/subject/surf)"
        ),
    )
    parser_plot.set_defaults(func=run_plotting)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    exit(main())
