import os
import subprocess as sp
import time
from glob import glob

from autoflatten.freesurfer import setup_freesurfer

setup_freesurfer("/data2/freesurfer-6.0", "/data2/freesurfer_subjects/all-subjects")
HERE = os.path.dirname(os.path.abspath(__file__))

subject_dir = "/data2/freesurfer_subjects/all-subjects/MVauto/surf"
patch_dir = os.path.join(HERE, "test_params")
img_dir = os.path.join(HERE, "test_params", "images")
os.makedirs(img_dir, exist_ok=True)

patch_files = sorted(glob(os.path.join(patch_dir, "*.3d")))

for patch_name in patch_files:
    img_name = os.path.join(
        img_dir, os.path.basename(patch_name).replace(".3d", ".png")
    )
    if not os.path.exists(img_name):
        cmd = (
            f"xvfb-run -a freeview -f {subject_dir}/lh.inflated:patch={patch_name} "
            "-viewport 3d -cam elevation 90 roll 180 "
            f"-ss {img_name} 4"
        )
        print(cmd)
        sp.check_call(cmd.split())
        time.sleep(1)  # give it a second to save the image
    # # trim black border using imagemagick
    # cmd = f"convert {img_name} -trim {img_name}"
    # print(cmd)
    # sp.check_call(cmd.split())
