"""FreeSurfer mris_flatten backend.

This module wraps FreeSurfer's mris_flatten command as a flattening backend.
"""

from pathlib import Path
from typing import Optional

from .base import FlattenBackend
from ..freesurfer import run_mris_flatten, is_freesurfer_available


class FreeSurferBackend(FlattenBackend):
    """FreeSurfer mris_flatten backend.

    This backend wraps FreeSurfer's mris_flatten command for cortical
    surface flattening. Requires FreeSurfer to be installed and configured.
    """

    @property
    def name(self) -> str:
        return "freesurfer"

    def is_available(self) -> bool:
        """Check if FreeSurfer is available."""
        return is_freesurfer_available()

    def get_install_instructions(self) -> str:
        return (
            "FreeSurfer backend requires FreeSurfer to be installed.\n"
            "Visit https://surfer.nmr.mgh.harvard.edu/ for installation instructions.\n"
            "Ensure FREESURFER_HOME and SUBJECTS_DIR environment variables are set."
        )

    def flatten(
        self,
        patch_path: str,
        surface_path: str,
        output_path: str,
        verbose: bool = True,
        subject: Optional[str] = None,
        seed: int = 0,
        threads: int = 16,
        distances: tuple = (15, 80),
        n: int = 200,
        dilate: int = 1,
        passes: int = 1,
        tol: float = 0.005,
        overwrite: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> str:
        """Flatten a cortical surface patch using mris_flatten.

        Parameters
        ----------
        patch_path : str
            Path to the input patch file
        surface_path : str
            Path to the base surface file (used to determine subject/hemi)
        output_path : str
            Path for the output flat patch file
        verbose : bool
            Whether to print progress messages
        subject : str, optional
            FreeSurfer subject ID. If None, will be inferred from surface_path.
        seed : int
            Random seed for mris_flatten
        threads : int
            Number of threads to use
        distances : tuple
            Distance parameters (distance1, distance2)
        n : int
            Maximum number of iterations
        dilate : int
            Number of dilations
        passes : int
            Number of passes
        tol : float
            Convergence tolerance
        overwrite : bool
            Whether to overwrite existing output
        debug : bool
            If True, preserve temporary directory for debugging
        **kwargs
            Additional arguments (ignored)

        Returns
        -------
        str
            Path to the output flat patch file
        """
        # Determine hemisphere from patch filename
        patch_name = Path(patch_path).name
        if patch_name.startswith("lh."):
            hemi = "lh"
        elif patch_name.startswith("rh."):
            hemi = "rh"
        else:
            raise ValueError(
                f"Cannot determine hemisphere from patch filename: {patch_name}"
            )

        # Determine subject from surface path or explicit argument
        if subject is None:
            # Try to infer subject from path structure:
            # .../SUBJECTS_DIR/subject_name/surf/lh.white
            surface_path_obj = Path(surface_path).resolve()
            if surface_path_obj.parent.name == "surf":
                subject = surface_path_obj.parent.parent.name
            else:
                raise ValueError(
                    f"Cannot infer subject from surface path: {surface_path}. "
                    "Please provide --subject argument."
                )

        # Determine output directory and name
        output_path_obj = Path(output_path)
        output_dir = str(output_path_obj.parent)
        output_name = output_path_obj.name

        if verbose:
            print(f"Running FreeSurfer mris_flatten backend")
            print(f"  Subject: {subject}")
            print(f"  Hemisphere: {hemi}")
            print(f"  Input: {patch_path}")
            print(f"  Output: {output_path}")

        # Call the existing run_mris_flatten function
        result_path = run_mris_flatten(
            subject=subject,
            hemi=hemi,
            patch_file=patch_path,
            output_dir=output_dir,
            output_name=output_name,
            seed=seed,
            threads=threads,
            distances=distances,
            n=n,
            dilate=dilate,
            passes=passes,
            tol=tol,
            overwrite=overwrite,
            debug=debug,
        )

        return result_path
