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
        "figure.constrained_layout.use": True,  # auto-space titles/labels, no overlap
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
    # timestamp stamp in the top-left corner (empty in all these layouts)
    fig.text(
        0.006,
        0.995,
        f"{name}  TS={ts}",
        fontsize=4.5,
        color="0.6",
        ha="left",
        va="top",
    )
    for ext in ("pdf", "png"):
        fig.savefig(
            out_dir / f"{name}_{ts}.{ext}", bbox_inches="tight", pad_inches=0.12
        )
    plt.close(fig)
    print(f"  wrote {name}_{ts}.pdf / .png")


def _strip(ax, x, vals, color, half_width=0.16, seed=0, size=11):
    """Jittered strip of points centred on category position ``x``."""
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return
    jit = (np.random.default_rng(seed).random(len(vals)) - 0.5) * 2 * half_width
    ax.scatter(
        np.full(len(vals), x) + jit,
        vals,
        s=size,
        color=color,
        alpha=0.75,
        linewidths=0,
        zorder=3,
    )


def _box(ax, x, vals, color, width=0.5):
    """Translucent box (no fliers) for a category, coloured by method."""
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return
    ax.boxplot(
        [vals],
        positions=[x],
        widths=width,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.30, linewidth=0.6),
        medianprops=dict(color=color, linewidth=1.4),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
    )


def _fs6_e2e_perbrain(subjects=None):
    """FreeSurfer 6 per-brain mris_flatten time (min): lh+rh summed per subject.

    From the existing fs6_compare/timing.csv (OMP mris_flatten, never re-run). If ``subjects``
    is given, restrict to those (paired to the AutoFlatten subjects).
    """
    rows = _read_csv(paths.DATA_ROOT / "fs6_compare" / "timing.csv")
    by_sub = defaultdict(dict)
    for r in rows:
        by_sub[r["subject"]][r["hemi"]] = _f(r.get("runtime_s"))
    keep = set(subjects) if subjects else None
    out = []
    for s, hemis in by_sub.items():
        if keep is not None and s not in keep:
            continue
        if (
            "lh" in hemis
            and "rh" in hemis
            and np.isfinite(hemis["lh"])
            and np.isfinite(hemis["rh"])
        ):
            out.append((hemis["lh"] + hemis["rh"]) / 60.0)
    return out


# =================================================================================
# Figure 1: end-to-end runtime @16 cores (group-level)
# =================================================================================
def fig_runtime_e2e(e2e_dir: Path, configs, out_dir: Path, ts: str) -> None:
    # per-subject per-config stage totals (sum over both hemispheres)
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
    if not any(data.values()):
        print("  [fig_runtime_e2e] no e2e data, skipping")
        return

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 3.1))

    # --- Panel A: per-brain total runtime (min, log) -- AutoFlatten vs FreeSurfer 6 ---
    af_subjects = sorted({s for cfg in configs for s in data.get(cfg, {})})
    fs6_perbrain = _fs6_e2e_perbrain(af_subjects)  # min, paired to the same subjects
    cats = list(configs) + (["freesurfer6"] if fs6_perbrain else [])
    cat_vals = {
        **{
            cfg: [sum(v.values()) / 60.0 for v in data[cfg].values()] for cfg in configs
        },
        "freesurfer6": fs6_perbrain,
    }
    for i, cat in enumerate(cats):
        vals = cat_vals.get(cat, [])
        _box(axA, i, vals, COL[cat], width=0.5)
        _strip(axA, i, vals, COL[cat], seed=i)
        if vals:
            med = float(np.median(vals))
            top = float(np.max(vals))
            txt = f"{med:.0f} min" if med >= 30 else f"{med:.1f} min"
            axA.annotate(
                txt,
                (i, top),
                color=COL[cat],
                fontsize=6.5,
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
    axA.set_yscale("log")
    axA.set_xticks(range(len(cats)))
    axA.set_xticklabels(
        [
            LABEL[c]
            .replace("AutoFlatten ", "AF\n")
            .replace("FreeSurfer 6", "FreeSurfer 6")
            for c in cats
        ]
    )
    axA.set_xlim(-0.6, len(cats) - 0.4)
    axA.set_ylabel("per-brain wall-clock (min)")
    axA.set_title("Time to flatten one brain\n(both hemispheres, 16 cores)", fontsize=8)

    # --- Panel B: per-stage runtime distribution (log s), grouped by config ---
    stages = [
        ("projection", "projection"),
        ("prep", "prep (k-ring)"),
        ("flatten", "flatten"),
    ]
    gap = len(configs) + 0.8
    centers = []
    for si, (skey, _slabel) in enumerate(stages):
        base = si * gap
        centers.append(base + (len(configs) - 1) / 2)
        for ci, cfg in enumerate(configs):
            vals = [data[cfg][s][skey] for s in data[cfg]]
            _box(axB, base + ci, vals, COL[cfg], width=0.6)
            _strip(axB, base + ci, vals, COL[cfg], seed=si * 10 + ci, size=8)
    axB.set_yscale("log")
    axB.set_xticks(centers)
    axB.set_xticklabels([s[1] for s in stages])
    axB.set_ylabel("stage wall-clock (s)")
    axB.set_title("Where the time goes", fontsize=8)
    handles = [
        plt.Line2D([0], [0], color=COL[c], lw=3, alpha=0.6, label=LABEL[c])
        for c in configs
    ]
    axB.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.5)

    axB.set_title("Where the time goes (AutoFlatten)", fontsize=8)
    axA.text(
        0.5,
        -0.32,
        "FreeSurfer 6 = mris_flatten (flatten only, OMP)",
        transform=axA.transAxes,
        ha="center",
        va="top",
        fontsize=5.5,
        color="0.4",
    )
    fig.suptitle("End-to-end AutoFlatten runtime vs FreeSurfer 6", fontsize=9.5)
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
def _fs6_distortion(hemis=None):
    """FS6 true-geodesic distortion from the existing comparison CSV.

    If ``hemis`` (a set of ``(subject, hemi)``) is given, restrict to those so FS6 is scored
    on the *same* hemispheres as AutoFlatten (paired comparison).
    """
    rows = _read_csv(paths.DATA_ROOT / "fs6_compare" / "true_comparison_20subj.csv")
    loc = {"robust_fast": [], "tutte_default": [], "freesurfer6": []}
    glob = {"robust_fast": [], "tutte_default": [], "freesurfer6": []}
    for r in rows:
        m = r.get("method")
        if m not in loc:
            continue
        if hemis is not None and (r.get("subject"), r.get("hemi")) not in hemis:
            continue
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

    # pair FS6 to the exact hemispheres AutoFlatten was scored on
    af_hemis = {(r["subject"], r["hemi"]) for r in sel}
    fs6_loc, fs6_glob = _fs6_distortion(hemis=af_hemis)
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


# =================================================================================
# Figure 4: speed / accuracy trade-off (esp. for choosing the default)
# =================================================================================
def _fs6_runtime_perhemi(hemis):
    """FS6 mris_flatten per-hemi runtime (s), keyed on (subject, hemi)."""
    out = {}
    for r in _read_csv(paths.DATA_ROOT / "fs6_compare" / "timing.csv"):
        key = (r.get("subject"), r.get("hemi"))
        if hemis is None or key in hemis:
            out[key] = _f(r.get("runtime_s"))
    return out


def _fs6_local_perhemi(hemis):
    """FS6 true-local distortion (%) per (subject, hemi)."""
    out = {}
    for r in _read_csv(paths.DATA_ROOT / "fs6_compare" / "true_comparison_20subj.csv"):
        if r.get("method") != "freesurfer6":
            continue
        key = (r.get("subject"), r.get("hemi"))
        if hemis is None or key in hemis:
            out[key] = _f(r.get("true_local_mean"))
    return out


def fig_speed_accuracy(cores_dir, configs, out_dir: Path, ts: str) -> None:
    """Per-hemisphere flatten runtime (x) vs local metric distortion (y).

    Lower-left is better (fast + accurate). Makes the robust_fast / tutte_default trade-off
    explicit and shows both dominate FreeSurfer 6.
    """
    rows = [
        r
        for r in _read_csv(cores_dir / "cores" / f"cores_{ts}.csv")
        if r.get("status") == "ok"
    ]
    if not rows:
        print("  [fig_speed_accuracy] no cores data, skipping")
        return
    nmax = max(int(r["n_cores"]) for r in rows)
    sel = [r for r in rows if int(r["n_cores"]) == nmax]
    af_hemis = {(r["subject"], r["hemi"]) for r in sel}

    # method -> list of (runtime_s, local_distortion%)
    pts = {c: [] for c in configs}
    for r in sel:
        c = r["config"]
        if c in pts:
            pts[c].append((_f(r["flatten_s"]), _f(r["true_local_at_optscale"])))
    fs6_rt = _fs6_runtime_perhemi(af_hemis)
    fs6_loc = _fs6_local_perhemi(af_hemis)
    fs6_pts = [
        (fs6_rt[k], fs6_loc[k])
        for k in af_hemis
        if k in fs6_rt
        and k in fs6_loc
        and np.isfinite(fs6_rt[k])
        and np.isfinite(fs6_loc[k])
    ]

    methods = list(configs) + (["freesurfer6"] if fs6_pts else [])
    series = {**pts, "freesurfer6": fs6_pts}

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for m in methods:
        arr = np.array(
            [p for p in series[m] if np.isfinite(p[0]) and np.isfinite(p[1])],
            dtype=float,
        )
        if not len(arr):
            continue
        ax.scatter(
            arr[:, 0], arr[:, 1], s=10, color=COL[m], alpha=0.35, linewidths=0, zorder=2
        )
        mx, my = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        ax.errorbar(
            mx,
            my,
            xerr=[
                [mx - np.percentile(arr[:, 0], 25)],
                [np.percentile(arr[:, 0], 75) - mx],
            ],
            yerr=[
                [my - np.percentile(arr[:, 1], 25)],
                [np.percentile(arr[:, 1], 75) - my],
            ],
            fmt="o",
            color=COL[m],
            ms=8,
            mec="white",
            mew=0.8,
            lw=1.0,
            capsize=2,
            zorder=4,
            label=LABEL[m],
        )
        # stagger labels so the two close AutoFlatten points don't collide
        off = {
            "robust_fast": (-6, -10, "right", "top"),
            "tutte_default": (6, 9, "left", "bottom"),
            "freesurfer6": (-8, 10, "right", "bottom"),
        }.get(m, (6, -8, "left", "top"))
        ax.annotate(
            f"{mx:.0f} s, {my:.1f}%",
            (mx, my),
            color=COL[m],
            fontsize=6,
            xytext=off[:2],
            textcoords="offset points",
            ha=off[2],
            va=off[3],
        )

    ax.set_xscale("log")
    ax.set_xlabel("flatten runtime (s, 16 cores; FS6 = mris_flatten, OMP)")
    ax.set_ylabel("local metric distortion @ opt scale (%)")
    ax.set_title("Speed / accuracy trade-off")
    ax.legend(loc="upper center", frameon=False, fontsize=6.5)
    ax.annotate(
        "better",
        xy=(0.05, 0.07),
        xytext=(0.60, 0.45),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=7,
        color="0.45",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8),
    )
    _save(fig, out_dir, "fig_speed_accuracy", ts)


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
    fig_speed_accuracy(cores_dir, args.configs, out_dir, args.ts)
    if args.e2e_dir:
        fig_runtime_e2e(Path(args.e2e_dir), args.configs, out_dir, args.ts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
