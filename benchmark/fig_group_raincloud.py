"""Raincloud figures for the group distortion-at-scale results.

Two figures from a group_flatten config CSV (true-geodesic distortion at optimal scale, the
reported metric -- local + global):

  1. ``fig_group_raincloud_paired``    -- paired LH-vs-RH rainclouds (vertical half-violin + box +
     jittered rain), with thin lines connecting the two hemispheres of each subject. Shows the
     hemispheric asymmetry and within-subject pairing.
  2. ``fig_group_raincloud_collapsed`` -- a single horizontal raincloud collapsing across all 100
     hemispheres (cloud above, box on the line, rain below). The headline "distortion at scale".

Both panel a metric pair: local (within-radius) and global distortion, each at the map's own
distance-optimal scale (see ``benchmark/truedist.py``; the k-ring metric is NOT used here).

Usage
-----
    python -m benchmark.fig_group_raincloud --run-dir <group_TS_dir> --ts <TS> [--config robust_fast]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 120,
    }
)

# metric key -> (column, axis label)
METRICS = {
    "local": ("true_local_at_optscale", "Local metric distortion (%)"),
    "global": ("true_global_at_optscale", "Global metric distortion (%)"),
}
HEMI_COLORS = {"lh": "#3B6EA5", "rh": "#C1532A"}
COLLAPSE_COLOR = "#4C8C6B"


def _load(
    run_dir: Path, config: str, ts: str, csv_path: Path | None = None
) -> list[dict]:
    if csv_path is None:
        csv_path = run_dir / "results" / f"group_{config}_{ts}.csv"
    rows = [r for r in csv.DictReader(open(csv_path)) if r.get("status") == "ok"]
    if not rows:
        raise SystemExit(f"no ok rows in {csv_path}")
    return rows


def _by_hemi(rows, col):
    out = {}
    for h in ("lh", "rh"):
        out[h] = {r["subject"]: float(r[col]) for r in rows if r["hemi"] == h}
    return out


def _half_violin(ax, vals, center, width, side, color, vmax=None):
    """Half-violin (KDE) anchored at ``center``; ``side`` in {'left','right','up','down'}.

    When ``vmax`` is given, the cloud represents the in-view bulk (values <= vmax): the KDE is
    fit on that subset so a far outlier can't inflate the bandwidth and flatten the density. The
    off-scale points are shown separately (carets) by the caller."""
    vals = np.asarray(vals, float)
    if vmax is not None:
        vals = vals[vals <= vmax]
    kde = gaussian_kde(vals)
    grid = np.linspace(vals.min(), vals.max(), 200)
    dens = kde(grid) / kde(grid).max() * width
    if side == "left":
        ax.fill_betweenx(grid, center - dens, center, color=color, alpha=0.35, lw=0)
    elif side == "right":
        ax.fill_betweenx(grid, center, center + dens, color=color, alpha=0.35, lw=0)
    elif side == "up":
        ax.fill_between(grid, center, center + dens, color=color, alpha=0.35, lw=0)
    elif side == "down":
        ax.fill_between(grid, center - dens, center, color=color, alpha=0.35, lw=0)


def _box(ax, vals, center, vert=True, width=0.06):
    bp = ax.boxplot(
        vals,
        positions=[center],
        widths=width,
        vert=vert,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", lw=1.2),
        boxprops=dict(facecolor="white", edgecolor="black", lw=0.8),
        whiskerprops=dict(color="black", lw=0.8),
        capprops=dict(color="black", lw=0.8),
    )
    return bp


def _rng(seed):
    # Seeded jitter for reproducible figures.
    return np.random.default_rng(seed)


def _far_fence(*arrays):
    """Tukey far-outlier fence (Q3 + 3*IQR) over the pooled values, or None if nothing exceeds
    it. Used to zoom the display so a few localized blow-ups don't squash the bulk."""
    v = np.concatenate([np.asarray(a, float) for a in arrays])
    q1, q3 = np.percentile(v, [25, 75])
    cap = q3 + 3.0 * (q3 - q1)
    return float(cap) if v.max() > cap else None


# ---------------------------------------------------------------------------------
# Figure 1: paired LH vs RH
# ---------------------------------------------------------------------------------
def fig_paired(rows, out_dir: Path, ts: str, config: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2))
    rng = _rng(0)
    for ax, (mkey, (col, label)) in zip(axes, METRICS.items()):
        hemi = _by_hemi(rows, col)
        subjects = sorted(set(hemi["lh"]) & set(hemi["rh"]))
        lh = np.array([hemi["lh"][s] for s in subjects])
        rh = np.array([hemi["rh"][s] for s in subjects])
        cap = _far_fence(lh, rh)

        # outward-facing clouds
        _half_violin(ax, lh, 0.0, 0.32, "left", HEMI_COLORS["lh"], vmax=cap)
        _half_violin(ax, rh, 1.0, 0.32, "right", HEMI_COLORS["rh"], vmax=cap)
        # boxes at category centers
        _box(ax, lh, 0.0)
        _box(ax, rh, 1.0)
        # rain in the middle, paired connectors (clipped by ylim when zoomed)
        xlh = 0.18 + rng.uniform(0, 0.10, len(lh))
        xrh = 0.82 - rng.uniform(0, 0.10, len(rh))
        for i in range(len(subjects)):
            ax.plot(
                [xlh[i], xrh[i]],
                [lh[i], rh[i]],
                color="0.6",
                lw=0.4,
                alpha=0.45,
                zorder=1,
            )
        ax.scatter(xlh, lh, s=12, color=HEMI_COLORS["lh"], alpha=0.8, lw=0, zorder=2)
        ax.scatter(xrh, rh, s=12, color=HEMI_COLORS["rh"], alpha=0.8, lw=0, zorder=2)

        # zoom past the far-outlier fence; mark off-scale points with carets + a note
        if cap is not None:
            ax.set_ylim(top=cap * 1.06)
            off = []
            for xs, vv in ((xlh, lh), (xrh, rh)):
                m = vv > cap
                if m.any():
                    ax.scatter(
                        xs[m],
                        np.full(m.sum(), cap),
                        marker="^",
                        s=22,
                        color="0.25",
                        zorder=3,
                        clip_on=False,
                    )
                    off += list(vv[m])
            ax.annotate(
                f"{len(off)} hemi off-scale (max {max(off):.0f}%)",
                xy=(0.5, cap),
                xytext=(0.5, cap * 0.99),
                ha="center",
                va="top",
                fontsize=6.5,
                color="0.3",
            )

        # frame the in-view bulk (avoid a wide empty margin from the long upper tail)
        pooled = np.concatenate([lh, rh])
        inv = pooled[pooled <= cap] if cap is not None else pooled
        rng_v = np.ptp(inv)
        top = (cap * 1.06) if cap is not None else inv.max() + 0.05 * rng_v
        ax.set_ylim(inv.min() - 0.08 * rng_v, top)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [f"LH\n(median {np.median(lh):.1f})", f"RH\n(median {np.median(rh):.1f})"]
        )
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(label)
        ax.set_title(mkey.capitalize(), fontsize=10)

    fig.suptitle(
        f"AutoFlatten group distortion ({config}) — paired by hemisphere, "
        f"{len(rows) // 2} subjects",
        fontsize=10,
    )
    fig.text(
        0.01, 0.01, f"fig_group_raincloud_paired  TS={ts}", fontsize=4.5, color="0.5"
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _save(fig, out_dir, "fig_group_raincloud_paired", ts, config)


# ---------------------------------------------------------------------------------
# Figure 2: collapsed across 100 hemispheres
# ---------------------------------------------------------------------------------
def fig_collapsed(rows, out_dir: Path, ts: str, config: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.4))
    rng = _rng(1)
    for ax, (mkey, (col, label)) in zip(axes, METRICS.items()):
        vals = np.array([float(r[col]) for r in rows])
        base = 0.0
        cap = _far_fence(vals)
        # cloud above, box on the line, rain below (classic horizontal raincloud)
        _half_violin(ax, vals, base, 0.30, "up", COLLAPSE_COLOR, vmax=cap)
        # horizontal boxplot on the baseline
        ax.boxplot(
            vals,
            positions=[base - 0.12],
            widths=0.10,
            vert=False,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black", lw=1.2),
            boxprops=dict(facecolor="white", edgecolor="black", lw=0.8),
            whiskerprops=dict(color="black", lw=0.8),
            capprops=dict(color="black", lw=0.8),
        )
        yr = base - 0.40 + rng.uniform(0, 0.16, len(vals))
        ax.scatter(vals, yr, s=12, color=COLLAPSE_COLOR, alpha=0.7, lw=0)
        med = float(np.median(vals))
        ax.axvline(
            med,
            color="black",
            lw=1.0,
            ls="--",
            label=f"median {med:.1f}%  (mean {vals.mean():.1f}%)",
        )
        ax.set_yticks([])
        ax.set_ylim(-0.65, 0.40)
        ax.set_xlabel(label)
        ax.set_title(f"{mkey.capitalize()} — {len(vals)} hemispheres", fontsize=10)
        ax.legend(loc="upper right", frameon=False, fontsize=8)

        # zoom past the far-outlier fence; carets + note for off-scale hemispheres
        if cap is not None:
            ax.set_xlim(right=cap * 1.04)
            off = vals[vals > cap]
            ax.scatter(
                np.full(len(off), cap),
                np.full(len(off), base - 0.40),
                marker=">",
                s=22,
                color="0.25",
                zorder=3,
                clip_on=False,
            )
            ax.annotate(
                f"{len(off)} off-scale (max {off.max():.0f}%)",
                xy=(cap, base - 0.40),
                xytext=(cap, base + 0.18),
                ha="right",
                va="center",
                fontsize=6.5,
                color="0.3",
            )

    fig.suptitle(
        f"AutoFlatten group distortion ({config}) — {len(rows)} hemispheres",
        fontsize=10,
    )
    fig.text(
        0.01, 0.01, f"fig_group_raincloud_collapsed  TS={ts}", fontsize=4.5, color="0.5"
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _save(fig, out_dir, "fig_group_raincloud_collapsed", ts, config)


def _load_both(run_dir: Path, ts: str, col: str):
    """Pair the two configs by (subject, hemi): {(subj,hemi): {config: value}}."""
    out: dict = {}
    for cfg in ("robust_fast", "tutte_default"):
        p = run_dir / "results" / f"group_{cfg}_{ts}.csv"
        for r in csv.DictReader(open(p)):
            if r.get("status") == "ok":
                out.setdefault((r["subject"], r["hemi"]), {})[cfg] = float(r[col])
    return out


def fig_config_paired(run_dir: Path, out_dir: Path, ts: str) -> None:
    """Collapsed vertical raincloud, robust_fast vs tutte_default paired per hemisphere, points
    colored by hemisphere (within-config pairing line connects each hemi's two configs)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.6))
    rng = _rng(2)
    for ax, (mkey, (col, label)) in zip(axes, METRICS.items()):
        data = _load_both(run_dir, ts, col)
        keys = [k for k in data if {"robust_fast", "tutte_default"} <= set(data[k])]
        rob = np.array([data[k]["robust_fast"] for k in keys])
        tut = np.array([data[k]["tutte_default"] for k in keys])
        hemis = np.array([k[1] for k in keys])
        cap = _far_fence(rob, tut)

        _half_violin(ax, rob, 0.0, 0.30, "left", "0.6", vmax=cap)
        _half_violin(ax, tut, 1.0, 0.30, "right", "0.6", vmax=cap)
        _box(ax, rob, 0.0)
        _box(ax, tut, 1.0)

        xr = 0.20 + rng.uniform(0, 0.10, len(keys))
        xt = 0.80 - rng.uniform(0, 0.10, len(keys))
        for i in range(len(keys)):
            ax.plot(
                [xr[i], xt[i]],
                [rob[i], tut[i]],
                color="0.75",
                lw=0.3,
                alpha=0.4,
                zorder=1,
            )
        for h in ("lh", "rh"):
            m = hemis == h
            ax.scatter(
                xr[m],
                rob[m],
                s=12,
                color=HEMI_COLORS[h],
                alpha=0.85,
                lw=0,
                zorder=2,
                label=h.upper() if ax is axes[0] else None,
            )
            ax.scatter(
                xt[m], tut[m], s=12, color=HEMI_COLORS[h], alpha=0.85, lw=0, zorder=2
            )

        if cap is not None:
            for xs, vv in ((xr, rob), (xt, tut)):
                off = vv > cap
                if off.any():
                    ax.scatter(
                        xs[off],
                        np.full(off.sum(), cap),
                        marker="^",
                        s=20,
                        color="0.25",
                        zorder=3,
                        clip_on=False,
                    )
            nbad = int((rob > cap).sum() + (tut > cap).sum())
            ax.annotate(
                f"{nbad} off-scale (max {max(rob.max(), tut.max()):.0f}%)",
                xy=(0.5, cap),
                xytext=(0.5, cap * 0.99),
                ha="center",
                va="top",
                fontsize=6.5,
                color="0.3",
            )
        pooled = np.concatenate([rob, tut])
        inv = pooled[pooled <= cap] if cap is not None else pooled
        rng_v = np.ptp(inv)
        top = (cap * 1.06) if cap is not None else inv.max() + 0.05 * rng_v
        ax.set_ylim(inv.min() - 0.08 * rng_v, top)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [
                f"robust_fast\n(median {np.median(rob):.1f})",
                f"tutte_default\n(median {np.median(tut):.1f})",
            ]
        )
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(label)
        ax.set_title(mkey.capitalize(), fontsize=10)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, title="hemisphere")
    fig.suptitle(
        f"AutoFlatten group distortion — robust_fast vs tutte_default, "
        f"{len(_load_both(run_dir, ts, METRICS['local'][0]))} hemispheres",
        fontsize=10,
    )
    fig.text(
        0.01, 0.01, f"fig_group_raincloud_configs  TS={ts}", fontsize=4.5, color="0.5"
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            out_dir / f"fig_group_raincloud_configs_{ts}.{ext}", bbox_inches="tight"
        )
    plt.close(fig)
    print(f"  wrote fig_group_raincloud_configs_{ts}.pdf / .png")


def _save(fig, out_dir: Path, name: str, ts: str, config: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}_{config}_{ts}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}_{config}_{ts}.pdf / .png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="the group_<TS> run dir")
    ap.add_argument("--ts", required=True)
    ap.add_argument("--config", default="robust_fast")
    ap.add_argument(
        "--csv",
        default=None,
        help="explicit results CSV (else auto: floored if present)",
    )
    ap.add_argument("--out-dir", default=None, help="default: <run-dir>/figures")
    ap.add_argument(
        "--mode",
        choices=["per-config", "config-paired"],
        default="per-config",
        help="per-config: paired-hemi + collapsed for --config; config-paired: robust vs tutte",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures"
    if args.mode == "config-paired":
        print(f"Rendering config-paired raincloud (robust vs tutte) -> {out_dir}")
        fig_config_paired(run_dir, out_dir, args.ts)
        return 0
    rows = _load(run_dir, args.config, args.ts, Path(args.csv) if args.csv else None)
    print(f"Rendering group rainclouds ({args.config}, {len(rows)} hemis) -> {out_dir}")
    fig_paired(rows, out_dir, args.ts, args.config)
    fig_collapsed(rows, out_dir, args.ts, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
