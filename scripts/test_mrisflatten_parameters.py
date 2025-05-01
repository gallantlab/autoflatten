import os

from autoflatten.freesurfer import run_mris_flatten, setup_freesurfer

# Setup freesurfer 8
setup_freesurfer("/data2/freesurfer-8.0.0", "/data2/freesurfer_subjects/all-subjects")

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
    (20, 40),
    (30, 40),
]

# I know that distances = (30, 30) and dilate = 1 works well.
# But I want to figure out if we can avoid dilating (and thus removing vertices.)
dilate = 1
n_threads = 16  # make it fast
n = 80  # number of max iterations
passes = 3  # number of passes

# Patch file
HERE = os.path.dirname(os.path.abspath(__file__))
test_patch = os.path.join(HERE, "lh.MVauto_autoflatten.patch.3d")
output_dir = os.path.join(HERE, "test_params")
hemi = "lh"

for distances in distances_param:
    print(f"Running mris_flatten with distances = {distances} and dilate = {dilate}")
    distances_str = f"distances{distances[0]:02d}{distances[1]:02d}"
    output_name = f"{hemi}.MVauto_autoflatten_{distances_str}_n{n}_dilate{dilate}"
    extra_params = None
    if passes > 1:
        output_name += f"_passes{passes}"
        extra_params = {"p": passes}
    output_name += ".flat.patch.3d"
    run_mris_flatten(
        "MVauto",
        "lh",
        test_patch,
        output_dir,
        output_name=output_name,
        n=n,
        threads=n_threads,
        distances=distances,
        dilate=dilate,
        extra_params=extra_params,
    )
