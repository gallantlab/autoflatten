import os
from glob import glob

from autoflatten.freesurfer import setup_freesurfer
from autoflatten.viz import plot_patch

setup_freesurfer("/data2/freesurfer-6.0", "/data2/freesurfer_subjects/all-subjects")
HERE = os.path.dirname(os.path.abspath(__file__))

subject_dir = "/data2/freesurfer_subjects/all-subjects/REDACTED_SUBJ/surf"
patch_dir = os.path.join(HERE, "test_params")
img_dir = os.path.join(HERE, "test_params", "images")
os.makedirs(img_dir, exist_ok=True)

patch_files = sorted(glob(os.path.join(patch_dir, "*.3d")))

for patch_name in patch_files:
    try:
        # Call the plot_patch function, explicitly setting output_dir
        img_name = plot_patch(
            patch_file=patch_name,
            subject_dir=subject_dir,
            output_dir=img_dir,  # Explicitly pass the desired output directory
            surface="lh.inflated",
        )
        print(f"Processed {os.path.basename(patch_name)}, output: {img_name}")
    except Exception as e:
        print(f"Failed to process {patch_name}: {e}")

print("Finished processing all patch files.")
