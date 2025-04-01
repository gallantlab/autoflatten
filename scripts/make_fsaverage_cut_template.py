"""This scripts stores the cut template for fsaverage in the default template directory.
In this way we don't have to recompute it at every call of autoflatten."""

from autoflatten.config import fsaverage_cut_template
from autoflatten.freesurfer import setup_freesurfer
from autoflatten.template import identify_surface_components
from autoflatten.utils import save_json

if not setup_freesurfer():
    raise RuntimeError(
        "FreeSurfer environment setup failed. Make sure to source FreeSurfer."
    )

subject = "fsaverage"
vertex_dict_lh = identify_surface_components(subject, "lh")
vertex_dict_rh = identify_surface_components(subject, "rh")
# Store in a single dictionary
vertex_dict = {f"lh_{key}": value for key, value in vertex_dict_lh.items()}
vertex_dict.update({f"rh_{key}": value for key, value in vertex_dict_rh.items()})

print(f"Saving fsaverage cut template to {fsaverage_cut_template}")
save_json(fsaverage_cut_template, vertex_dict)
