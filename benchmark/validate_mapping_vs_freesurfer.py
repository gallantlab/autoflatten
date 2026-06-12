"""Gold-standard check: Python cut mapper vs real ``mri_label2label`` (FreeSurfer).

:mod:`benchmark.validate_projection` proves the *end-to-end* patch is bit-identical to the
cached FreeSurfer patches, but the cut vertices it compares have already been replaced by
geodesic refinement, so that test cannot see the raw mapping. This script closes the gap:
it runs the **actual** ``autoflatten.core.map_cuts_to_subject`` (which shells out to
``mri_label2label --regmethod surface``) and compares its mapped vertex IDs, cut by cut,
against :func:`benchmark.projection.map_cuts_to_subject_python`.

Requires FreeSurfer on PATH. On this machine::

    source ~/bin/source_freesurfer.sh   # FreeSurfer 6.0
    export SUBJECTS_DIR=/data2/projects/idem/exps/narratives/datalad-narratives/derivatives/freesurfer
    export PYTHONPATH=/home/jlg/mvdoc/repos/autoflatten
    python -m benchmark.validate_mapping_vs_freesurfer

Result (3 dev hemispheres): total vertex-ID symmetric difference = 0 -- the Python
``union(push, pull)`` KDTree mapper reproduces ``mri_label2label`` exactly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

import numpy as np

from . import paths
from .ledger import Ledger, new_record
from .projection import map_cuts_to_subject_python

from autoflatten.config import fsaverage_cut_template
from autoflatten.core import map_cuts_to_subject

CUTS = ["mwall", "calcarine", "medial1", "medial2", "medial3", "temporal"]


def compare_hemi(subject, hemi):
    template = json.load(open(fsaverage_cut_template))
    vd = {k[3:]: np.array(v) for k, v in template.items() if k.startswith(hemi + "_")}

    t0 = time.time()
    fs = map_cuts_to_subject(vd, subject, hemi)
    fs_s = time.time() - t0
    t0 = time.time()
    py = map_cuts_to_subject_python(vd, subject, hemi)
    py_s = time.time() - t0

    per_cut = {}
    total_diff = 0
    for cut in CUTS:
        a = set(int(x) for x in fs.get(cut, []))
        b = set(int(x) for x in py.get(cut, []))
        diff = len(a ^ b)
        total_diff += diff
        per_cut[cut] = {
            "n_fs": len(a),
            "n_py": len(b),
            "jaccard": (len(a & b) / len(a | b)) if (a | b) else 1.0,
            "n_diff": diff,
        }
    return {
        "subject": subject,
        "hemi": hemi,
        "fs_s": fs_s,
        "py_s": py_s,
        "total_diff": total_diff,
        "per_cut": per_cut,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--hemis",
        nargs="*",
        default=["sub-022:lh", "sub-022:rh", "sub-026:lh"],
        help="subject:hemi pairs (default: 3 dev hemispheres)",
    )
    args = ap.parse_args()

    if shutil.which("mri_label2label") is None:
        print(
            "mri_label2label not on PATH -- source FreeSurfer first "
            "(e.g. `source ~/bin/source_freesurfer.sh`).",
            file=sys.stderr,
        )
        return 1

    paths.ensure_output_dirs()
    ledger = Ledger()
    results = []
    grand_total = 0
    for spec in args.hemis:
        subject, hemi = spec.split(":")
        r = compare_hemi(subject, hemi)
        results.append(r)
        grand_total += r["total_diff"]
        print(f"=== {subject} {hemi}  FS:{r['fs_s']:.1f}s  PY:{r['py_s']:.2f}s ===")
        for cut in CUTS:
            c = r["per_cut"][cut]
            tag = "EXACT" if c["n_diff"] == 0 else f"diff={c['n_diff']}"
            print(
                f"  {cut:11} FS={c['n_fs']:6d} PY={c['n_py']:6d} "
                f"jaccard={c['jaccard'] * 100:6.2f}%  {tag}"
            )

        rec = new_record(
            kind="experiment",
            label=f"exp:mapping_vs_freesurfer:{subject}.{hemi}",
            subjects=[{"subject": subject, "hemi": hemi}],
            method={
                "name": "python_mapper_vs_mri_label2label",
                "freesurfer": "6.0",
                "mapper": "python_kdtree_union_push_pull",
            },
            repro_command=f"python -m benchmark.validate_mapping_vs_freesurfer --hemis {spec}",
        )
        rec.metrics = {
            "total_vertex_id_diff": r["total_diff"],
            "exact": r["total_diff"] == 0,
            "freesurfer_s": r["fs_s"],
            "python_s": r["py_s"],
            "per_cut": r["per_cut"],
        }
        rec.status = "ok"
        rec.decision["hypothesis"] = (
            "The Python union(push, pull) KDTree mapper reproduces mri_label2label "
            "--regmethod surface vertex-for-vertex (not just in count)."
        )
        rec.decision["conclusion"] = (
            f"{subject} {hemi}: {r['total_diff']} vertex-ID differences across all cuts "
            f"(FreeSurfer {r['fs_s']:.1f}s vs Python {r['py_s']:.2f}s)."
        )
        ledger.append(rec)

    print(
        f"\nTOTAL vertex-ID symmetric difference across "
        f"{len(results)} hemispheres: {grand_total}"
    )
    return 0 if grand_total == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
