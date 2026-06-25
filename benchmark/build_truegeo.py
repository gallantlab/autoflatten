"""Compute and cache the true-geodesic reference for a (subject, hemi).

Heat-method geodesic fields from a fixed set of sampled sources, used by
:mod:`benchmark.truedist` to score maps independently of the k-ring energy. Run once per
hemisphere you want to evaluate on; the result is cached under the k-ring cache dir as
``{subject}_{hemi}.truegeo.npz``.

Usage
-----
    python -m benchmark.build_truegeo --subject sub-026 --hemi lh
    python -m benchmark.build_truegeo --subject sub-022 --hemi rh --n-sources 200 --radius 30
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import paths
from .harness import build_flattener, load_manifest
from .truedist import compute_truegeo, truegeo_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--hemi", required=True, choices=["lh", "rh"])
    ap.add_argument("--n-sources", type=int, default=200)
    ap.add_argument("--radius", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = truegeo_path(args.subject, args.hemi)
    if out.exists() and not args.force:
        print(f"Already cached: {out}")
        return 0

    from autoflatten.flatten import FlattenConfig

    cfg = FlattenConfig()
    cfg.verbose = False
    manifest = load_manifest()
    entry = [
        e
        for e in manifest["entries"]
        if e["subject"] == args.subject and e["hemi"] == args.hemi
    ]
    if not entry:
        print(f"No manifest entry for {args.subject} {args.hemi}", file=sys.stderr)
        return 1

    paths.ensure_output_dirs()
    t0 = time.time()
    flattener = build_flattener(entry[0], cfg, use_cache=True)
    ref = compute_truegeo(
        flattener, n_sources=args.n_sources, radius=args.radius, seed=args.seed
    )
    paths.KRING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out, **ref)  # includes surface/sanitized provenance
    print(
        f"Wrote {out} ({ref['srcs'].size} sources, R={ref['R']}mm) in {time.time() - t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
