"""
This scripts stores the cut template for a subject in the default template directory.
In this way we don't have to recompute it at every call of autoflatten.

Parameters
----------
subject : str
    The subject name for which to generate the cut template.
"""

import argparse
import os

from autoflatten.config import fsaverage_cut_template
from autoflatten.core import ensure_continuous_cuts, map_cuts_to_subject
from autoflatten.freesurfer import setup_freesurfer
from autoflatten.template import identify_surface_components
from autoflatten.utils import save_json

if not setup_freesurfer():
    raise RuntimeError(
        "FreeSurfer environment setup failed. Make sure to source FreeSurfer."
    )

parser = argparse.ArgumentParser(
    description="Store the cut template for a subject in the default template directory."
)
parser.add_argument(
    "subject", type=str, help="The subject name for which to generate the cut template."
)
args = parser.parse_args()
subject = args.subject

all_vertex_dict = {}
for hemi in ["lh", "rh"]:
    # Step 1: Identify cuts in the subject
    vertex_dict = identify_surface_components(subject, hemi)
    # Step 2: Map cuts to fsaverage
    print(f"Mapping cuts from {subject} to fsaverage for {hemi}")
    vertex_dict_mapped = map_cuts_to_subject(
        vertex_dict, "fsaverage", hemi, source_subject=subject
    )
    # Step 3: Ensure cuts are continuous in target subject
    print(f"Ensuring continuous cuts on fsaverage for {subject} {hemi}")
    vertex_dict_fixed = ensure_continuous_cuts(
        vertex_dict_mapped.copy(), "fsaverage", hemi
    )
    # Store in a single dictionary
    all_vertex_dict.update(
        {f"{hemi}_{key}": value for key, value in vertex_dict_fixed.items()}
    )

# Save in the same directory as fsaverage_cut_template, but with subject in filename
base_dir = os.path.dirname(fsaverage_cut_template)
outfile = os.path.join(base_dir, f"{subject}_cut_template.json")

print(f"Saving {subject} cut template to {outfile}")
save_json(outfile, all_vertex_dict)
