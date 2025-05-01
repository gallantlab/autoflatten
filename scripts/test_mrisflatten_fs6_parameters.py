import os

from autoflatten.freesurfer import run_mris_flatten, setup_freesurfer

# Setup freesurfer 6
setup_freesurfer("/data2/freesurfer-6.0", "/data2/freesurfer_subjects/all-subjects")

# -distances N N parameter
# used to build the distance matrix for the flattening
# the first number specifies the neighborhood size
# the second number specifies the maximum number of neighbors
distances_param = [
    # (7, 12),   # default
    # (20, 20),  # recommended by Bruce Fischl
    # (30, 30),
    # (40, 20),
    # (50, 20),
    # (60, 20),  # too many, gives MRISsampleDistances: too many neighbors
]

# I know that distances = (30, 30) and dilate = 1 works well.
# But I want to figure out if we can avoid dilating (and thus removing vertices.)
dilate_params = [0, 1]
# Use default values for the rest
# n = 80  # number of max iterations
# passes = 5  # number of passes

# Patch file
HERE = os.path.dirname(os.path.abspath(__file__))
test_patch = os.path.join(HERE, "lh.REDACTED_SUBJ_autoflatten.patch.3d")
output_dir = os.path.join(HERE, "test_params")
hemi = "lh"

for dilate in dilate_params:
    print(f"Running fs6 mris_flatten with dilate = {dilate}")
    output_name = f"{hemi}.REDACTED_SUBJ_autoflatten_fs6_default_dilate{dilate}"
    output_name += ".flat.patch.3d"
    run_mris_flatten(
        "REDACTED_SUBJ",
        "lh",
        test_patch,
        output_dir,
        output_name=output_name,
        n=None,
        threads=None,
        distances=None,
        dilate=dilate,
        extra_params=None,
    )
