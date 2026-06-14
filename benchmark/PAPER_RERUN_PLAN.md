# Paper Benchmark Re-run + Nature-style Figures — Plan

## Context

The flattening code changed: the perf + metric-fidelity improvements (PRs #69/#70) were merged
into `dev`. The previously recorded pyflatten benchmarks in `/data2/projects/autoflatten/` were
produced by the *old* code, so the paper needs **fresh** timing + distortion numbers on the
current `dev` package. The FreeSurfer `mris_flatten` comparison does **not** change (no FS code
moved) — its existing timings/metrics are reused as-is, never re-run.

Deliverables (all confirmed with the user):
1. **End-to-end** AutoFlatten timing at **16 cores**, reported **per subject** (both hemis run
   sequentially, each hemi using all 16 cores) — the "time to flatten a brain" headline.
2. **Flatten-only** timing at **1, 8, and 16 cores** on a fixed hemisphere set, **plus distortion
   metrics** at each core count (distortion should be core-count-invariant — recording it confirms
   determinism).
3. Headline method benchmarked as **two AutoFlatten variants**: `robust_fast` (shipping default)
   and `tutte_default` (quality-first).
4. **Separate Nature-style figures** (one per benchmark), vector PDF + PNG.
5. **Per-hemisphere qualitative figures (Fischl 1999 style)** — one image **per hemisphere**, with
   the **three methods** (`robust_fast`, `tutte_default`, `freesurfer6`) side by side; each method
   shows its flatmap colored by per-vertex distortion **plus a distortion histogram**.

Constraints (standing): public Narratives data only; **CPU-only** (don't attempt CUDA);
**all generated artifacts → `/data2/projects/autoflatten/`** (repo holds only code + markdown);
American spelling in code.

## Hardware / environment (verified)
- 32 logical cores = **16 physical × 2 hyperthreads** (Xeon Silver 4216), 250 GB RAM. **We cap at
  16 = the physical-core count** and never use hyperthreads, so the 1→8→16 scaling curve is free of
  HT confounds (and whether JAX even benefits across physical cores is part of what we measure).
- Run in the repo uv `.venv`; JAX = CPU backend. Core count set via
  `autoflatten/flatten/threading.py::configure_threading(n)` (sets `XLA_FLAGS` device/eigen-thread
  count + `OMP/MKL/OPENBLAS_NUM_THREADS` + `numba.set_num_threads`). **Must be called before JAX
  import** → each core-count run is a fresh **subprocess**.

## Subject sets (fixed, documented)
- **`S_time` (core-scaling, 10 hemis)** — reuse the prior `speed_1core` set exactly:
  `sub-055 lh`, `sub-055 rh`, `sub-066 rh`, `sub-190 rh`, `sub-201 rh`, `sub-264 rh`, `sub-268 rh`,
  `sub-271 lh`, `sub-296 rh`, `sub-303 lh`. Rationale: **FS6 timings already exist for these exact
  hemis at 1- and 8-core** (`/data2/.../speed_1core/`, `speed_8core/`), so the scaling figure
  overlays FreeSurfer with zero re-run.
- **`S_e2e` (end-to-end, ~10 subjects, both hemis)** — 10 subjects drawn from the available
  Narratives derivatives that have both `lh`/`rh` patch + base surface (prefer subjects already in
  the scaleup-20 set for continuity). Each subject → LH then RH at 16 cores.

## Branch strategy
New branch **`bench/paper-rerun-2026`** off `origin/dev` (current merged code), then overlay only
the benchmark tooling:
```
git checkout -b bench/paper-rerun-2026 origin/dev
git checkout origin/benchmark-autoresearch -- benchmark/
```
This guarantees the **package code == dev** (no merge, no conflicts) while pulling in the
self-contained `benchmark/` harness. First step after: `uv run python -c "import benchmark.harness"`
to confirm the harness imports cleanly against dev's package API.

## What we build / reuse

### Reuse (do not modify)
- `benchmark/harness.py` — `build_flattener(entry, config, use_cache)`, `load_manifest`,
  `select_entries`; k-ring cache pathing.
- `benchmark/metrics.py` — `per_patch_metrics(uv, flattener)` → mean/p90 distortion, n_flipped,
  frac_flipped, area_distortion. Method-agnostic, identical scoring for both configs.
- `benchmark/truedist.py` — `true_distortion_full`, `fischl_distortion` (energy-independent
  yardstick; use where a `{subject}_{hemi}.truegeo.npz` reference exists).
- `benchmark/scaleup.py` — `run_hemi(...)` does **end-to-end** (projection→prep→flatten) with
  `projection_s/prep_s/flatten_s/total_s` + distortion + **resumable CSV**, and the
  `--flatten-config {robust_fast,tutte_default,...}` factory (canonical source of the two named
  configs). Used directly for Deliverable 1.
- `benchmark/ledger.py` — append a record per run (commit SHA, env, config, metrics, runtime,
  repro command).
- `benchmark/paths.py` — canonical `/data2/projects/autoflatten/` locations.
- `benchmark/fig_*.py` (`fig_speed.py`, `fig_speed_accuracy.py`, `fig_group.py`) — Nature-style
  rcParams conventions (spines off, `font.size` 11–12, method color palette, `constrained_layout`).
- `autoflatten/viz.py` — `plot_flatmap()` for the **layout** of the Fischl-1999 three-panel visual
  and `compute_kring_distortion(...)` for the **fair per-vertex field** (fixed neighborhood + optimal
  scale, identical across methods); `autoflatten/freesurfer.py` patch readers. Basis for Deliverable 5.
- **Existing FS6 flat patches** under `/data2/.../fs6_compare/runs/<subject>_<hemi>/` (produced by
  the prior `freesurfer6` runs on `S_time`) — reused directly for the FS column; **mris_flatten is
  never re-run**.

### New code (in repo, committed on the branch)
- **`benchmark/time_cores.py`** — core-scaling driver for Deliverable 2.
  - *Worker mode* (`--worker`): args `(subject, hemi, patch, surface, config_name, n_cores, out_csv)`.
    Calls `configure_threading(n_cores)` **before** importing the flatten stack; builds the named
    config via scaleup's factory; runs flatten with **k-ring cache DISABLED** (so the Numba-parallel
    prep actually runs and scales — a cache hit would zero out `prep_s`); times `prep_s` (k-ring) and
    `flatten_s` (optimization) separately; computes `per_patch_metrics` (+ truedist if ref exists);
    appends one CSV row and one ledger record. On the **16-core** run it also **saves the flat
    patch** (`flattener.save_result`) into the hemi's run dir, so the per-hemi flatmaps
    (Deliverable 5) render without re-flattening.
  - *Driver mode*: iterates `S_time × {robust_fast, tutte_default} × {1, 8, 16}` and spawns each
    worker as a fresh subprocess (correct per-run XLA thread/device env). Resumable (skip rows
    already in CSV). Output → `/data2/projects/autoflatten/paper_bench_2026/cores/cores.csv`.
- **`benchmark/fig_paper.py`** — renders the three aggregate figures (Deliverables 1–3) from the CSVs.
  Reuses the existing rcParams/palette helpers from `fig_speed.py`.
- **`benchmark/fig_flatmaps.py`** — Deliverable 5. For each hemi in `S_time`, loads the three flat
  patches (robust_fast + tutte_default from our 16-core runs; freesurfer6 from the existing
  `fs6_compare/runs/`), computes **per-vertex distortion via `viz.compute_kring_distortion` with a
  fixed `(k, n)` + `optimal_scale=True` identical for all three methods** (dense, fair; see "Distortion
  metric"), and renders **one figure per hemisphere**: a 3-column panel (one column per method),
  each column = distortion-colored flatmap above its distortion histogram (Fischl-1999 layout), with
  a shared color scale / histogram x-axis and per-method summary stats (mean / p90 / flips). Skips a
  method if its flat patch is missing (logs which). Reuses `plot_flatmap` coloring/binning where
  practical.

### Deliverable 1 runner (end-to-end @16 cores)
Drive `scaleup.py` (or a thin wrapper) over `S_e2e` at 16 cores, once per config, writing to
`/data2/projects/autoflatten/paper_bench_2026/e2e/{robust_fast,tutte_default}.csv`. Per-subject
headline = `lh.total_s + rh.total_s` (sequential). Keeps scaleup's projection/prep/flatten
breakdown for the stacked-bar figure.

## Output layout (all under /data2 — nothing generated lands in the repo)
**Every run is timestamped** so we always know when it was produced. A single run stamp
`TS = YYYYMMDD-HHMMSS` (captured once at launch) names the run directory, and that same stamp is
embedded in every CSV/figure filename and stored in a `run_meta.json` (TS, git SHA, host, core
counts, subject sets). Per-row CSVs also carry a `timestamp` column (ISO 8601, per measurement).
```
/data2/projects/autoflatten/paper_bench_2026/<TS>/
  run_meta.json                            # TS, commit SHA, host, configs, S_time, S_e2e
  cores/cores_<TS>.csv                     # S_time × {2 configs} × {1,8,16}: + per-row `timestamp`
  e2e/robust_fast_<TS>.csv                 # S_e2e per-hemi end-to-end @16c (scaleup schema + timestamp)
  e2e/tutte_default_<TS>.csv
  figures/fig_runtime_e2e_<TS>.{pdf,png}   # TS also rendered as small caption text on each figure
  figures/fig_core_scaling_<TS>.{pdf,png}
  figures/fig_distortion_<TS>.{pdf,png}
  flatmaps/<subject>_<hemi>_3method_<TS>.{pdf,png}   # Deliverable 5: one per S_time hemisphere
```
Resume targets the existing `<TS>` dir (don't mint a new stamp on resume). Ledger records appended
to `/data2/projects/autoflatten/ledger/experiments.jsonl` (each already carries its own ISO
timestamp + repro command).

## Distortion metric — use the CORRECT local measure (do not use the k-ring metric)

The harness exposes several distortion numbers and they are **not interchangeable**. The k-ring
`mean_distortion` in `per_patch_metrics` is **biased by `n_neighbors_per_ring`** (robust_fast uses
n=6, tutte_default/baseline differ), so it is **not comparable across the methods** and must **not**
be the reported metric (the project's "§8 miscalibration trap"; see memory `[[autoflatten-distortion-metric]]`).

Reference: **Ju et al. 2005, NeuroImage 28(4):869–880** distinguishes a **local** from a **global**
metric-distortion measure (FreeSurfer best on global; LSCM best on local). For the paper we report
the **energy-independent, true-geodesic metric** from `benchmark/truedist.py`, computed against
heat-method geodesics and evaluated **at each map's own distance-optimal scale**, so it is
n_neighbors-independent and method-fair:
- **Local metric distortion** (the "local" measure, the one the user is flagging): relative (%)
  error of flattened vs. true-geodesic distance for pairs within a local radius R, at optimal scale
  → `truedist.true_distortion_full(...)["true_local_mean"]` (+ p90). This is the primary local number.
- **Global metric distortion**: `true_global_at_optscale` (co-primary, matches Ju's "global").
- **Fischl J_d** (`fischl_distortion`, RMS mm within 0.5 cm at opt scale) reported as the
  FreeSurfer-native cross-check, since our optimizer clones `mris_flatten`.

Two distinct uses, two correct choices (the bias only bites when configs are compared using each
one's **own** differing internal targets):
- **Aggregate distortion numbers** (`fig_distortion` box/strip): the **true-geodesic** measures
  above (`true_local_mean` + `true_global_at_optscale`), scored by the identical `truedist` routine
  for all three methods (FS6 from its flat patch). Unbiased, n_neighbors-independent.
- **Per-vertex flatmap coloring + histogram** (`fig_flatmaps`, Fischl 1999 Fig 9/11 visual): recompute
  per-vertex distortion from each saved flat patch + base surface with `viz.compute_kring_distortion`
  using a **fixed `(k, n)` + `optimal_scale=True`, identical for all three methods**. This is dense
  (100% vertex coverage, vs ~80% for source-sampled heat geodesics) and fair because the neighborhood
  is the same for every method — it is exactly the quantity Fischl plotted. NOT each config's own
  biased internal k-ring targets.

**Before plotting, confirm the exact Ju-2005 local-distortion definition against the full text**
(neighborhood, relative-vs-absolute, optimal-scale handling); cross-check against the FreeSurfer
metric in Fischl 1999 "Cortical Surface-Based Analysis II" (`recon2.pdf`). The worker records both
the true-geodesic numbers and the native k-ring diagnostic per run; the fair per-vertex field is
computed at figure time from the saved flat patches.

## Figures (Nature style: vector PDF + 300-dpi PNG, spines off, single-column width)
**Aggregate figures (Deliverables 1–3):**
1. **`fig_runtime_e2e`** — end-to-end per-subject wall-clock @16 cores. Stacked bars
   (projection / prep / flatten) per subject, faceted or grouped by the two configs; annotate
   median per-brain time. Optional FS6 reference line from existing `fs6_compare/timing.csv`.
2. **`fig_core_scaling`** — flatten-only runtime vs cores {1,8,16}, **log–log**, one line per config
   (mean ± spread over `S_time`), plus an **ideal linear-scaling** guide and the **FS6** points
   (1- and 8-core, from existing CSVs). Annotate measured speedups (16 cores = all physical cores,
   no HT). Secondary: report prep vs flatten scaling separately (k-ring scales well; JAX less so).
3. **`fig_distortion`** — distortion at convergence per config: box/strip of **true-local @
   opt-scale** (primary), true-global @ opt-scale, Fischl J_d, and flip counts, with FS6 from existing
   `fs6_compare/comparison.csv` as the comparator. Also serves as the core-count-invariance check
   (distortion at 1/8/16 overlaid — should coincide).

**Per-hemisphere qualitative figures (Deliverable 5, `fig_flatmaps.py`):** one image per `S_time`
hemisphere, 3 columns (robust_fast / tutte_default / freesurfer6), each = distortion-colored flatmap
+ distortion histogram with shared scales and mean/p90/flips annotations — the Fischl-1999 visual.

## Execution order
1. Create branch, overlay `benchmark/`, confirm harness imports against dev package.
2. **Smoke (1 hemi)**: run `time_cores.py` worker for `sub-055 lh`, robust_fast, n_cores=8 →
   verify CSV row + ledger record + plausible prep/flatten split and distortion.
3. **Confirm the local-distortion definition** (Ju 2005 full text + Fischl 1999 `recon2.pdf`) and
   align `truedist`'s local metric to it if needed; build any missing `truegeo` refs for `S_time`.
4. **Core-scaling run** (`S_time`, both configs, 1/8/16) — background, resumable. ~4–5 h (1-core
   dominates).
5. **End-to-end run** (`S_e2e`, both configs, 16 cores) via scaleup — background, resumable.
6. **Render aggregate figures** with `fig_paper.py`; eyeball; iterate styling.
7. **Render per-hemi flatmaps** with `fig_flatmaps.py` over `S_time` (uses the saved 16-core flat
   patches + existing FS6 flats).
8. Append ledger records; regenerate `NOTEBOOK.md` (`python -m benchmark.report`).
9. Commit code (`time_cores.py`, `fig_paper.py`, `fig_flatmaps.py`) + this plan to the branch. Send
   the user the figures (3 aggregate + per-hemi flatmaps).

## Verification
- **Import/parity:** `git diff --stat origin/dev -- autoflatten/` is empty (package untouched);
  `import benchmark.harness` succeeds in the `.venv`.
- **Smoke:** one-hemi worker produces a CSV row with `prep_s>0` (cache disabled), `flatten_s>0`,
  finite distortion, and a ledger record with the dev commit SHA.
- **Determinism / invariance:** for a given hemi+config, distortion at 1/8/16 cores agrees within
  tolerance (assert in `fig_distortion`); runtime monotonically decreases 1→8 (and ≥ no worse at 16).
- **Metric correctness:** the reported local distortion is the true-geodesic local-at-opt-scale
  measure (matches Ju 2005's "local"), **not** the k-ring metric. Sanity check its
  n_neighbors-independence: recompute on the same map at n=6 vs n=12 — true-local agrees, whereas the
  k-ring metric would not. All three methods scored by the identical `truedist` routine.
- **End-to-end:** scaleup CSV has `status==ok` for all `S_e2e` hemis; per-subject total = lh+rh.
- **Figures:** three aggregate PDFs render without error, axes/labels correct, FS6 overlays align on
  `S_time`; each figure carries the run stamp `TS` as caption text.
- **Per-hemi flatmaps:** one figure per `S_time` hemisphere with all three method columns populated
  (or a logged skip when an FS6 flat is genuinely missing); distortion color scale + histogram axes
  shared across the three methods within a hemi.
- **Timestamping:** the `<TS>` run dir, `run_meta.json`, every CSV/figure filename, and the per-row
  CSV `timestamp` column are all present and consistent.
- **Provenance:** every run appended to the ledger; nothing written outside
  `/data2/projects/autoflatten/`; `git status` in the repo shows only the new tracked code files
  (`time_cores.py`, `fig_paper.py`, `fig_flatmaps.py`) plus the overlaid `benchmark/` tree.
