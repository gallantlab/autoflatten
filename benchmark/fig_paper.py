"""Nature-style aggregate figures for the paper benchmark (Deliverables 1-3).

Renders three separate, publication-ready figures (vector PDF + 300-dpi PNG) from the
timestamped benchmark CSVs:

  1. fig_runtime_e2e   -- end-to-end per-subject wall-clock @16 cores, stacked by stage.
  2. fig_core_scaling  -- flatten-only runtime vs cores {1,8,16}, log-log, FS6 overlaid.
  3. fig_distortion    -- true-geodesic local + global distortion at optimal scale, FS6 overlaid.

FreeSurfer6 numbers are reused from the existing comparison (never re-run): timings from
``speed_1core/8core.csv``, distortion from ``fs6_compare/true_comparison_20subj.csv``.

Usage
-----
    python -m benchmark.fig_paper --cores-dir <cores_TS_dir> --e2e-dir <e2e_TS_dir> \\
        --ts <TS> --out-dir <cores_TS_dir>/figures
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import paths  # noqa: E402

# ---- Nature-ish style -----------------------------------------------------------
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # editable text in vector PDF
        "font.family": "sans-serif",
    }
)

COL = {
    "robust_fast": "#2e6f95",  # blue
    "tutte_default": "#8a4f9e",  # purple
    "freesurfer6": "#d1495b",  # red
}
LABEL = {
    "robust_fast": "AutoFlatten (robust_fast)",
    "tutte_default": "AutoFlatten (tutte_default)",
    "freesurfer6": "FreeSurfer 6",
}
STAGE_COL = {"projection": "#9ecae1", "prep": "#fdae6b", "flatten": "#74c476"}


def _read_csv(path: Path) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _save(fig, out_dir: Path, name: str, ts: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.text(0.005, 0.005, f"{name}  TS={ts}", fontsize=4.5, color="0.5")
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}_{ts}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}_{ts}.pdf / .png")


# =================================================================================
# Figure 1: end-to-end per-subject runtime @16 cores
# =================================================================================
def fig_runtime_e2e(e2e_dir: Path, configs, out_dir: Path, ts: str) -> None:
    # per-subject per-config stage totals (sum over hemis)
    data = {}  # config -> subject -> {projection,prep,flatten}
    for cfg in configs:
        rows = _read_csv(e2e_dir / "e2e" / f"{cfg}_{ts}.csv")
        per_sub = defaultdict(lambda: dict(projection=0.0, prep=0.0, flatten=0.0))
        for r in rows:
            if r.get("status") != "ok":
                continue
            s = r["subject"]
            per_sub[s]["projection"] += _f(r["projection_s"], 0)
            per_sub[s]["prep"] += _f(r["prep_s"], 0)
            per_sub[s]["flatten"] += _f(r["flatten_s"], 0)
        data[cfg] = dict(per_sub)
    subjects = sorted({s for cfg in configs for s in data.get(cfg, {})})
    if not subjects:
        print("  [fig_runtime_e2e] no e2e data, skipping")
        return

    n_cfg = len(configs)
    fig, ax = plt.subplots(figsize=(min(7.0, 0.45 * len(subjects) * n_cfg + 1.5), 2.6))
    x = np.arange(len(subjects))
    width = 0.8 / n_cfg
    for j, cfg in enumerate(configs):
        off = (j - (n_cfg - 1) / 2) * width
        proj = np.array(
            [data[cfg].get(s, {}).get("projection", np.nan) for s in subjects]
        )
        prep = np.array([data[cfg].get(s, {}).get("prep", np.nan) for s in subjects])
        flat = np.array([data[cfg].get(s, {}).get("flatten", np.nan) for s in subjects])
        ax.bar(
            x + off,
            proj,
            width,
            color=STAGE_COL["projection"],
            edgecolor="white",
            linewidth=0.3,
            label="projection" if j == 0 else None,
        )
        ax.bar(
            x + off,
            prep,
            width,
            bottom=proj,
            color=STAGE_COL["prep"],
            edgecolor="white",
            linewidth=0.3,
            label="prep (k-ring)" if j == 0 else None,
        )
        ax.bar(
            x + off,
            flat,
            width,
            bottom=proj + prep,
            color=STAGE_COL["flatten"],
            edgecolor="white",
            linewidth=0.3,
            label="flatten" if j == 0 else None,
        )
        totals = proj + prep + flat
        med = np.nanmedian(totals)
        ax.axhline(med, color=COL[cfg], lw=0.8, ls="--", alpha=0.8)
        ax.text(
            len(subjects) - 0.4,
            med,
            f" {LABEL[cfg].split('(')[1].rstrip(')')}\n median {med:.0f}s",
            color=COL[cfg],
            fontsize=5.5,
            va="center",
            ha="left",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("sub-", "") for s in subjects], rotation=45, ha="right"
    )
    ax.set_xlabel("subject")
    ax.set_ylabel("per-brain wall-clock (s, 16 cores)")
    ax.set_title("End-to-end AutoFlatten runtime (both hemispheres, 16 cores)")
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=6)
    ax.margins(x=0.02)
    _save(fig, out_dir, "fig_runtime_e2e", ts)


# =================================================================================
# Figure 2: flatten-only core scaling
# =================================================================================
def _fs6_timing(core_counts):
    """Median FreeSurfer6 flatten runtime (s) at available core counts (1, 8)."""
    out = {}
    for n, fname in (
        (1, "speed_1core/speed_1core.csv"),
        (8, "speed_8core/speed_8core.csv"),
    ):
        rows = _read_csv(paths.DATA_ROOT / fname)
        vals = [_f(r["seconds"]) for r in rows if r.get("method") == "freesurfer6"]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[n] = float(np.median(vals))
    return out


def fig_core_scaling(
    cores_dir: Path, configs, core_counts, out_dir: Path, ts: str
) -> None:
    rows = _read_csv(cores_dir / "cores" / f"cores_{ts}.csv")
    rows = [r for r in rows if r.get("status") == "ok"]
    if not rows:
        print("  [fig_core_scaling] no cores data, skipping")
        return

    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    cores_sorted = sorted(core_counts)
    for cfg in configs:
        ys, lo, hi = [], [], []
        for n in cores_sorted:
            v = [
                _f(r["flatten_s"])
                for r in rows
                if r["config"] == cfg and int(r["n_cores"]) == n
            ]
            v = [x for x in v if np.isfinite(x)]
            ys.append(np.median(v) if v else np.nan)
            lo.append(np.percentile(v, 25) if v else np.nan)
            hi.append(np.percentile(v, 75) if v else np.nan)
        ys = np.array(ys)
        ax.fill_between(cores_sorted, lo, hi, color=COL[cfg], alpha=0.15, linewidth=0)
        ax.plot(cores_sorted, ys, "-o", color=COL[cfg], ms=4, lw=1.2, label=LABEL[cfg])
        if np.isfinite(ys[0]) and np.isfinite(ys[-1]):
            sp = ys[0] / ys[-1]
            ax.annotate(
                f"{sp:.1f}x",
                (cores_sorted[-1], ys[-1]),
                color=COL[cfg],
                fontsize=6,
                xytext=(3, 2),
                textcoords="offset points",
            )
        # ideal linear-scaling guide from the 1-core point
        if np.isfinite(ys[0]):
            ideal = ys[0] / np.array(cores_sorted, float)
            ax.plot(cores_sorted, ideal, ":", color=COL[cfg], lw=0.7, alpha=0.6)

    fs6 = _fs6_timing(core_counts)
    if fs6:
        ax.plot(
            list(fs6),
            list(fs6.values()),
            "s",
            color=COL["freesurfer6"],
            ms=5,
            label=LABEL["freesurfer6"],
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(cores_sorted)
    ax.set_xticklabels(cores_sorted)
    ax.set_xlabel("CPU cores (physical, no hyperthreads)")
    ax.set_ylabel("flatten-only runtime (s)")
    ax.set_title("Flatten-only core scaling")
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.02,
        0.02,
        "dotted = ideal linear scaling",
        transform=ax.transAxes,
        fontsize=5.5,
        color="0.5",
    )
    _save(fig, out_dir, "fig_core_scaling", ts)


# =================================================================================
# Figure 3: distortion (true-geodesic, optimal scale)
# =================================================================================
def _fs6_distortion(s_time_subjects=None):
    """FS6 true-geodesic distortion from the existing comparison CSV."""
    rows = _read_csv(paths.DATA_ROOT / "fs6_compare" / "true_comparison_20subj.csv")
    loc = {"robust_fast": [], "tutte_default": [], "freesurfer6": []}
    glob = {"robust_fast": [], "tutte_default": [], "freesurfer6": []}
    for r in rows:
        m = r.get("method")
        if m in loc:
            loc[m].append(_f(r.get("true_local_mean")))
            glob[m].append(_f(r.get("true_global_at_optscale")))
    return loc, glob


def fig_distortion(cores_dir, configs, out_dir: Path, ts: str) -> None:
    rows = _read_csv(cores_dir / "cores" / f"cores_{ts}.csv")
    rows = [r for r in rows if r.get("status") == "ok"]
    if not rows:
        print("  [fig_distortion] no cores data, skipping")
        return

    # use the 16-core rows (distortion is core-invariant; this avoids triple-counting)
    nmax = max(int(r["n_cores"]) for r in rows)
    sel = [r for r in rows if int(r["n_cores"]) == nmax]

    fs6_loc, fs6_glob = _fs6_distortion()
    methods = list(configs) + ["freesurfer6"]
    panels = [
        (
            "true_local_at_optscale",
            "local metric distortion @ opt scale (%)",
            {
                c: [_f(r["true_local_at_optscale"]) for r in sel if r["config"] == c]
                for c in configs
            },
            fs6_loc,
        ),
        (
            "true_global_at_optscale",
            "global metric distortion @ opt scale (%)",
            {
                c: [_f(r["true_global_at_optscale"]) for r in sel if r["config"] == c]
                for c in configs
            },
            fs6_glob,
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.7))
    for ax, (key, ylabel, mydata, fs6data) in zip(axes, panels):
        series = {
            **mydata,
            "freesurfer6": [v for v in fs6data["freesurfer6"] if np.isfinite(v)],
        }
        xs = np.arange(len(methods))
        for i, m in enumerate(methods):
            vals = [v for v in series.get(m, []) if np.isfinite(v)]
            if not vals:
                continue
            ax.boxplot(
                vals,
                positions=[i],
                widths=0.5,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor=COL[m], alpha=0.35, linewidth=0.6),
                medianprops=dict(color=COL[m], linewidth=1.2),
                whiskerprops=dict(linewidth=0.6),
                capprops=dict(linewidth=0.6),
            )
            jit = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.18
            ax.scatter(
                np.full(len(vals), i) + jit,
                vals,
                s=6,
                color=COL[m],
                alpha=0.7,
                linewidths=0,
                zorder=3,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [LABEL[m].replace("AutoFlatten ", "AF\n") for m in methods], fontsize=6
        )
        ax.set_ylabel(ylabel)
    axes[0].set_title("Local (Ju 2005)", fontsize=8)
    axes[1].set_title("Global (Ju 2005)", fontsize=8)
    fig.suptitle("Metric distortion (true geodesics, optimal scale)", fontsize=9)
    _save(fig, out_dir, "fig_distortion", ts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cores-dir", required=True, help="core-scaling run dir (<TS>)")
    ap.add_argument("--e2e-dir", default=None, help="end-to-end run dir (e2e_<TS>)")
    ap.add_argument("--ts", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--configs", nargs="*", default=["robust_fast", "tutte_default"])
    ap.add_argument("--core-counts", nargs="*", type=int, default=[1, 8, 16])
    args = ap.parse_args()

    cores_dir = Path(args.cores_dir)
    out_dir = Path(args.out_dir) if args.out_dir else cores_dir / "figures"
    print(f"Rendering figures -> {out_dir}")
    fig_core_scaling(cores_dir, args.configs, args.core_counts, out_dir, args.ts)
    fig_distortion(cores_dir, args.configs, out_dir, args.ts)
    if args.e2e_dir:
        fig_runtime_e2e(Path(args.e2e_dir), args.configs, out_dir, args.ts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
