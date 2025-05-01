import os
import subprocess as sp
import time


def plot_patch(
    patch_file, subject_dir, output_dir=None, surface="lh.inflated", trim=True
):
    """
    Generate a PNG image of a FreeSurfer patch file using FreeView.

    Requires FreeView to be installed and accessible in the environment,
    potentially via xvfb-run for headless operation.

    Parameters
    ----------
    patch_file : str
        Path to the input patch file (e.g., *.3d).
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
    img_name = os.path.join(
        output_dir, os.path.basename(patch_file).replace(".3d", ".png")
    )
    surface_path = os.path.join(subject_dir, surface)

    if not os.path.exists(img_name):
        # Construct the command
        # Uses xvfb-run for headless environments
        # Sets camera elevation and roll for a consistent view
        # Saves screenshot (-ss) with a factor of 4 for higher resolution
        cmd = (
            f"xvfb-run -a freeview -f {surface_path}:patch={patch_file} "
            "-viewport 3d -cam elevation 90 roll 180 "
            f"-ss {img_name} 4"
        )
        print(f"Running command: {cmd}")
        try:
            sp.check_call(cmd.split())
            # Brief pause to ensure the file system catches up
            time.sleep(1)
            print(f"Image saved to: {img_name}")
        except sp.CalledProcessError as e:
            print(f"Error running FreeView command: {e}")
            raise
        except FileNotFoundError:
            print(
                "Error: 'xvfb-run' or 'freeview' command not found. "
                "Ensure FreeSurfer is sourced and xvfb is installed."
            )
            raise
        if trim:
            cmd_trim = f"convert {img_name} -trim {img_name}"
            try:
                sp.check_call(cmd_trim.split())
            except (sp.CalledProcessError, FileNotFoundError) as e:
                print(f"Warning: Could not trim image {img_name}. Error: {e}")
    else:
        print(f"Image already exists: {img_name}. Deleting it if you want to re-run.")

    return img_name
