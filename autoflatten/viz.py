import os
import shutil
import subprocess as sp
import tempfile
import time

from autoflatten.freesurfer import _create_temp_surf_directory


def plot_patch(
    patch_file, subject, subject_dir, output_dir=None, surface="lh.inflated", trim=True
):
    """
    Generate a PNG image of a FreeSurfer patch file using FreeView.

    Requires FreeView to be installed and accessible in the environment,
    potentially via xvfb-run for headless operation.

    Parameters
    ----------
    patch_file : str
        Path to the input patch file (e.g., *.3d).
    subject : str
        FreeSurfer subject identifier.
    subject_dir : str
        Path to the specific subject's surf directory within SUBJECTS_DIR.
    output_dir : str or None, optional
        Directory where the output PNG image will be saved.
        If None, the image is saved in the same directory as `patch_file`.
        Default is None.
    surface : str, optional
        The surface file to display the patch on (default is 'lh.inflated').
        This should be relative to `subject_dir`.
    trim : bool, optional
        Whether to trim the image after saving. Default is True.

    Returns
    -------
    str
        Path to the generated PNG image.

    Raises
    ------
    FileNotFoundError
        If the input patch file does not exist.
    subprocess.CalledProcessError
        If the FreeView command fails.
    """
    if not os.path.exists(patch_file):
        raise FileNotFoundError(f"Patch file not found: {patch_file}")

    # Default output directory to patch file's directory if None
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(patch_file))

    os.makedirs(output_dir, exist_ok=True)
    final_img_name = os.path.join(
        output_dir, os.path.basename(patch_file).replace(".3d", ".png")
    )

    if os.path.exists(final_img_name):
        print(
            f"Image already exists: {final_img_name}. Deleting it if you want to re-run."
        )
        return final_img_name

    # Create temporary directory for isolated execution
    temp_root = tempfile.mkdtemp(prefix="autoflatten_plot_")
    try:
        # Create temporary surf directory with symlinks to original files
        temp_surf_dir = _create_temp_surf_directory(subject, subject_dir, temp_root)

        # Copy patch file to temp surf directory
        patch_basename = os.path.basename(patch_file)
        temp_patch = os.path.join(temp_surf_dir, patch_basename)
        shutil.copy2(patch_file, temp_patch)
        print(f"Copied patch file to temporary location: {temp_patch}")

        # Construct paths for FreeView command (both files now in same directory)
        temp_surface = os.path.join(temp_surf_dir, surface)
        temp_img = os.path.join(temp_surf_dir, patch_basename.replace(".3d", ".png"))

        # Construct the command
        # Uses xvfb-run for headless environments
        # Sets camera elevation and roll for a consistent view
        # Dolly out (zoom < 1.0) to see the entire flatmap
        # Saves screenshot (-ss) with a factor of 8 for higher resolution
        cmd = (
            f"xvfb-run -a freeview -f {temp_surface}:patch={temp_patch} "
            "-viewport 3d -cam elevation 90 roll 180 dolly 0.5 "
            f"-ss {temp_img} 8"
        )
        print(f"Running command: {cmd}")
        try:
            sp.check_call(cmd.split())
            # Brief pause to ensure the file system catches up
            time.sleep(1)
            print(f"Image saved to temporary location: {temp_img}")
        except sp.CalledProcessError as e:
            print(f"Error running FreeView command: {e}")
            raise
        except FileNotFoundError:
            print(
                "Error: 'xvfb-run' or 'freeview' command not found. "
                "Ensure FreeSurfer is sourced and xvfb is installed."
            )
            raise

        # Trim if requested
        if trim:
            cmd_trim = (
                f"convert {temp_img} -trim -bordercolor black -border 20 {temp_img}"
            )
            try:
                sp.check_call(cmd_trim.split())
                print(f"Trimmed image with black border: {temp_img}")
            except (sp.CalledProcessError, FileNotFoundError) as e:
                print(f"Warning: Could not trim image {temp_img}. Error: {e}")

        # Copy final image to desired output location
        if not os.path.exists(temp_img):
            raise RuntimeError(
                f"FreeView succeeded but output image not found: {temp_img}"
            )
        shutil.copy2(temp_img, final_img_name)
        print(f"Image saved to final location: {final_img_name}")

        return final_img_name

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_root):
            try:
                shutil.rmtree(temp_root)
                print(f"Cleaned up temporary directory: {temp_root}")
            except Exception as e:
                print(f"Warning: Failed to clean up {temp_root}: {e}")
                print("You may need to manually remove it.")
