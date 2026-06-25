"""Build the benchmark manifest from the public Narratives FreeSurfer derivatives.

Selects subjects that already have a projection patch (``{hemi}.autoflatten.patch.3d``)
and a materialized base surface (``{hemi}.fiducial``, fallback ``smoothwm``), then writes
a documented ``manifest.json`` with a deterministic train/holdout split.

No FreeSurfer, no fetching: everything is already on disk and git-annex tracked.

Usage
-----
    python -m benchmark.build_dataset                 # default ~18-subject benchmark
    python -m benchmark.build_dataset --dev            # tiny: 2 subjects (4 hemis)
    python -m benchmark.build_dataset --n-subjects 30  # custom size
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import paths

HEMIS = ("lh", "rh")
DATASET_DOI = "10.18112/openneuro.ds002345"  # OpenNeuro Narratives (Nastase et al.)


def _datalad_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(paths.NARRATIVES_FS), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _base_surface(surf: Path, hemi: str) -> Path | None:
    for name in (f"{hemi}.fiducial", f"{hemi}.smoothwm"):
        p = surf / name
        if p.exists():  # follows symlink -> True only if annex content is present
            return p
    return None


def discover_subjects() -> list[dict]:
    """Return manifest entries for every (subject, hemi) with patch + base surface present."""
    entries = []
    for subj_dir in sorted(paths.NARRATIVES_FS.glob("sub-*")):
        surf = subj_dir / "surf"
        if not surf.is_dir():
            continue
        for hemi in HEMIS:
            patch = surf / f"{hemi}.autoflatten.patch.3d"
            base = _base_surface(surf, hemi)
            if patch.exists() and base is not None:
                entries.append(
                    {
                        "subject": subj_dir.name,
                        "hemi": hemi,
                        "patch_path": str(patch),
                        "surface_path": str(base),
                        "surface_kind": base.name.split(".", 1)[1],
                    }
                )
    return entries


def assign_splits(entries: list[dict], holdout_every: int = 3) -> list[dict]:
    """Deterministic split: every ``holdout_every``-th *subject* (sorted) is holdout.

    Splitting by subject (not hemisphere) keeps both hemispheres of a subject in the same
    split, avoiding leakage. No RNG, so the split is fully reproducible.
    """
    subjects = sorted({e["subject"] for e in entries})
    holdout = {s for i, s in enumerate(subjects) if i % holdout_every == 0}
    for e in entries:
        e["split"] = "holdout" if e["subject"] in holdout else "train"
    return entries


def build(n_subjects: int | None, dev: bool, holdout_every: int) -> dict:
    all_entries = discover_subjects()
    all_subjects = sorted({e["subject"] for e in all_entries})

    if dev:
        n_subjects = 2
    if n_subjects is not None:
        keep = set(all_subjects[:n_subjects])
        entries = [e for e in all_entries if e["subject"] in keep]
    else:
        entries = all_entries

    entries = assign_splits(entries, holdout_every=holdout_every)
    subjects = sorted({e["subject"] for e in entries})

    return {
        "dataset": "narratives",
        "dataset_doi": DATASET_DOI,
        "dataset_source": str(paths.NARRATIVES_FS),
        "datalad_commit": _datalad_commit(),
        "created": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "rule": "first N subjects (sorted) with patch+base surface present; "
            f"every {holdout_every}rd subject -> holdout",
            "n_subjects_requested": n_subjects,
            "holdout_every": holdout_every,
            "dev": dev,
        },
        "n_subjects": len(subjects),
        "n_subjects_available": len(all_subjects),
        "n_entries": len(entries),
        "n_train": sum(e["split"] == "train" for e in entries),
        "n_holdout": sum(e["split"] == "holdout" for e in entries),
        "entries": entries,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-subjects", type=int, default=18, help="benchmark size (subjects)"
    )
    ap.add_argument("--dev", action="store_true", help="tiny 2-subject dev manifest")
    ap.add_argument("--holdout-every", type=int, default=3)
    ap.add_argument("--out", type=Path, default=paths.MANIFEST_PATH)
    args = ap.parse_args()

    paths.ensure_output_dirs()
    manifest = build(args.n_subjects, args.dev, args.holdout_every)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest -> {args.out}")
    print(
        f"  {manifest['n_subjects']} subjects "
        f"({manifest['n_subjects_available']} available), "
        f"{manifest['n_entries']} hemispheres "
        f"[train={manifest['n_train']}, holdout={manifest['n_holdout']}]"
    )


if __name__ == "__main__":
    main()
