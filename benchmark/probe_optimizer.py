"""Probe: swap the optimization *algorithm* on the same energy.

The current refinement minimizes the metric+area energy (J_d + J_a) with a hand-rolled
vectorized quadratic line-search gradient descent wrapped in a multi-level gradient-
smoothing *continuation* (coarse-to-fine). This probe keeps the energy but replaces the
optimizer with an off-the-shelf one (scipy L-BFGS-B / CG, or Adam), starting from the
flip-free Tutte init, and scores the result with the same harness.

Caveat: the multi-level smoothing in the FreeSurfer-style loop is a continuation that helps
escape poor local minima. A "stronger" optimizer on the raw energy may actually do worse
(more flips / higher distortion). That comparison is the point.

The energy minimized is ``(l_dist/avg_nbrs)*J_d + l_nlarea*J_a`` — the ``1/avg_nbrs`` folds in
the FreeSurfer gradient normalization so we minimize the same effective objective.

Usage
-----
    python -m benchmark.probe_optimizer --optimizer lbfgs --subset 1 --save
    python -m benchmark.probe_optimizer --optimizer cg --max-iter 400 --subset 1
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import paths
from .harness import evaluate, load_manifest, register_flatten_fn, select_entries
from .ledger import Ledger, file_hash, new_record
from .probe_tutte_init import flipfree_init, scale_to_area


def make_optimizer_flatten_fn(
    optimizer: str = "lbfgs",
    max_iter: int = 300,
    l_dist: float = 1.0,
    l_nlarea: float = 1.0,
    init: str = "tutte",
):
    """Build a ``flatten_fn`` that minimizes J_d + J_a with the chosen optimizer."""

    def _fn(flattener):
        import jax
        import jax.numpy as jnp

        from autoflatten.flatten.algorithm import make_energy_fn

        faces_np = np.asarray(flattener.faces)
        x0 = flipfree_init(flattener.vertices, flattener.faces, method=init)
        x0 = scale_to_area(x0, faces_np, flattener.orig_area)
        n_v = x0.shape[0]

        avg = flattener.avg_neighbors or 1.0
        energy_fn = make_energy_fn(
            l_dist / avg,
            l_nlarea,
            flattener.neighbors_jax,
            flattener.targets_jax,
            flattener.mask_jax,
            flattener.faces_jax,
        )
        vg = jax.jit(jax.value_and_grad(lambda u: energy_fn(u)[0]))

        def val_grad_flat(x):
            uv = jnp.asarray(x.reshape(n_v, 2))
            e, g = vg(uv)
            return float(e), np.asarray(g, dtype=np.float64).reshape(-1)

        x0f = x0.reshape(-1).astype(np.float64)

        if optimizer in ("lbfgs", "cg"):
            from scipy.optimize import minimize

            method = "L-BFGS-B" if optimizer == "lbfgs" else "CG"
            res = minimize(
                val_grad_flat,
                x0f,
                jac=True,
                method=method,
                options={"maxiter": max_iter},
            )
            uv = res.x.reshape(n_v, 2)
        elif optimizer == "adam":
            # Minimal Adam (optax is not installed).
            lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
            x = x0f.copy()
            m = np.zeros_like(x)
            v = np.zeros_like(x)
            for t in range(1, max_iter + 1):
                _, g = val_grad_flat(x)
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                mh = m / (1 - b1**t)
                vh = v / (1 - b2**t)
                x = x - lr * mh / (np.sqrt(vh) + eps)
            uv = x.reshape(n_v, 2)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer!r}")
        return uv

    return _fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--optimizer", default="lbfgs", choices=["lbfgs", "cg", "adam"])
    ap.add_argument("--init", default="tutte", choices=["tutte", "lscm"])
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--l-dist", type=float, default=1.0)
    ap.add_argument("--l-nlarea", type=float, default=1.0)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--split", default=None, choices=["train", "holdout"])
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    label = f"optim_{args.optimizer}"
    register_flatten_fn(
        label,
        make_optimizer_flatten_fn(
            args.optimizer, args.max_iter, args.l_dist, args.l_nlarea, args.init
        ),
    )

    from autoflatten.flatten import FlattenConfig

    cfg = FlattenConfig()
    cfg.verbose = False

    paths.ensure_output_dirs()
    manifest = load_manifest()
    subset = 2 if args.dev else args.subset
    entries = select_entries(manifest, split=args.split, subset=subset)
    if not entries:
        print("No manifest entries selected.", file=sys.stderr)
        return 1

    record = new_record(
        kind="experiment",
        label=f"exp:{label}",
        manifest_id=manifest.get("created"),
        subjects=[
            {"subject": e["subject"], "hemi": e["hemi"], "split": e.get("split")}
            for e in entries
        ],
        method={
            "name": label,
            "optimizer": args.optimizer,
            "init": args.init,
            "max_iter": args.max_iter,
            "l_dist": args.l_dist,
            "l_nlarea": args.l_nlarea,
            "note": "energy-only optimizer swap; no multiscale schedule, no NAR cleanup",
        },
        repro_command="python -m benchmark.probe_optimizer " + " ".join(sys.argv[1:]),
    )
    record.decision = {
        "hypothesis": (
            f"Minimizing the same J_d+J_a energy with {args.optimizer} from the Tutte init "
            "may match the FreeSurfer-style schedule; or it may get stuck / flip without the "
            "multi-level smoothing continuation."
        )
    }

    print(
        f"Optimizer probe '{label}' on {len(entries)} hemispheres ({record.experiment_id})..."
    )
    t0 = time.time()
    save_dir = paths.RUNS_DIR / record.experiment_id if args.save else None
    result = evaluate(entries, cfg, method=label, save_dir=save_dir)
    record.per_subject = result["per_subject"]
    record.metrics = result["aggregate"]
    record.runtime_s = time.time() - t0
    if save_dir is not None:
        record.artifacts = [
            {
                "path": r["artifact"],
                "hash": file_hash(r["artifact"]),
                "kind": "flat_patch",
            }
            for r in result["per_subject"]
            if r.get("artifact")
        ]
    record.status = "ok" if result["aggregate"].get("n_failed", 0) == 0 else "partial"

    agg = result["aggregate"]
    base = Ledger().latest(kind="baseline")
    base_m = base["metrics"] if base else None
    if base_m and "mean_distortion" in agg:
        record.decision["conclusion"] = (
            f"{label}: dist {agg['mean_distortion']:.2f} vs base {base_m.get('mean_distortion'):.2f}; "
            f"flips {agg['total_flipped']} vs {base_m.get('total_flipped')}; "
            f"runtime {agg.get('mean_runtime_s', 0):.0f}s vs {base_m.get('mean_runtime_s', 0):.0f}s."
        )
    Ledger().append(record)

    print("\n=== aggregate ===")
    for k in (
        "n_patches",
        "n_failed",
        "mean_distortion",
        "total_flipped",
        "frac_patches_with_flips",
        "mean_runtime_s",
    ):
        if k in agg:
            print(f"  {k}: {agg[k]}")
    if base_m:
        print("\n=== vs baseline ===")
        for k in ("mean_distortion", "total_flipped", "mean_runtime_s"):
            print(f"  {k}: exp={agg.get(k)}  baseline={base_m.get(k)}")
    print(f"\nLogged {record.experiment_id} -> {Ledger().path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
