# Benchmark outputs & data guide

Orientation for the paper benchmark: **what is produced, where it lives, what is stored, and
how to load it** to inspect or re-plot. Code lives in this repo (`benchmark/`); all generated
data lives under a separate **data root** (nothing generated is written into the repo).

- **Data root** `DATA_ROOT` = `/data2/projects/autoflatten` (override with env
  `AUTOFLATTEN_BENCH_ROOT`). Defined in [`benchmark/paths.py`](paths.py).
- **Read-only input** `NARRATIVES_FS` = the Narratives FreeSurfer derivatives
  `/data2/projects/idem/exps/narratives/datalad-narratives/derivatives/freesurfer`
  (override with `AUTOFLATTEN_NARRATIVES_FS`). Per-subject `surf/{lh,rh}.fiducial`, `.sphere.reg`,
  etc. Surface content is materialized for ~82 subjects (the rest are unfetched git-annex links).

Every run is timestamped `TS = YYYYMMDD-HHMMSS`; the stamp names the run directory and is embedded
in CSV/figure filenames and a `run_meta.json`.

---

## 1. The canonical runs (use these)

| Deliverable | Directory (`under DATA_ROOT/paper_bench_2026/`) | TS |
|---|---|---|
| **Core-scaling** (flatten-only, 1/8/16 cores) + per-hemi flatmaps | `20260617-123120/` | 20260617-123120 |
| **End-to-end timing** (projection+prep+flatten @16c, 10 subj) | `e2e_20260617-123120/` | 20260617-123120 |
| **Group quality** (50 subj / 100 hemi, both configs) | `group_20260624-060144/` | 20260624-060144 |
| **FreeSurfer-6 comparison** (40 hemi, same patches) | `DATA_ROOT/fs6_compare/` | — |

Two configs throughout: **`robust_fast`** (shipping default) and **`tutte_default`** (quality-first).

Older run dirs (`20260614-105856`, `20260615-084508`, `e2e_*`, `smoke*`, `*_fixtest`) are
**superseded** (earlier projection variants / pre-metric-fix) — kept for provenance, not for the
paper. `PAPER_BENCH_TS.txt` / `PAPER_BENCH_TS_CONTINUITY.txt` list run stamps.

> **Metric note (important):** all `true_*` distortion numbers in the canonical runs are on the
> **fiducial** surface with the geodesic≥½·chord sanitization (see §5). The `*.unfloored.csv` and
> `truegeo_inflated/` siblings are pre-fix backups — do **not** use them for results.

---

## 2. Per-run directory layout

### Core-scaling — `paper_bench_2026/20260617-123120/`
```
run_meta.json                      # TS, git_sha, host, configs, core_counts, s_time, projection, created
cores/cores_<TS>.csv               # the data: S_time x {2 configs} x {1,8,16 cores}  (see §3)
patches/<subj>_<hemi>.autoflatten.patch.3d     # input 3D patches (continuity-only projection), 10 hemis
flat/<subj>_<hemi>_<config>.flat.patch.3d      # flattened maps, 16-core run, 10 hemis x 2 configs = 20
truegeo/<subj>_<hemi>.truegeo.npz              # fiducial geodesic reference per hemi (see §4)
truegeo_inflated/                  # PRE-FIX backup (inflated surface) — ignore
figures/fig_*_<TS>.{pdf,png}       # fig_core_scaling, fig_distortion, fig_distortion_paired,
                                   #   fig_speed_accuracy, fig_runtime_e2e
flatmaps/<subj>_<hemi>_3method_<TS>.{pdf,png}  # per-hemi 3-method Fischl-style panels (10 hemis)
*.log
```
`S_time` = 10 hemis: sub-055 lh/rh, 066 rh, 190 rh, 201 rh, 264 rh, 268 rh, 271 lh, 296 rh, 303 lh.

### End-to-end — `paper_bench_2026/e2e_20260617-123120/`
```
run_meta.json
e2e/<config>_<TS>.csv              # per-hemi end-to-end timing+distortion (10 subj x 2 hemi x 2 cfg)
e2e/<config>_<TS>.unfloored.csv    # PRE-FIX backup — ignore
patches/  flat/  truegeo/  truegeo_inflated/  kring_cache/
```
`S_e2e` = 10 subjects (both hemis), no S_time overlap: sub-056/259/265/270/277/283/285/287/298/299.

### Group quality — `paper_bench_2026/group_20260624-060144/`
```
COHORT.md                          # the 50-subject cohort + reproducible selection rule
run_meta.json                      # configs, n_subjects, n_hemis, lanes, etc.
results/group_<config>_<TS>.csv    # the data: 50 subj x 2 hemi per config (see §3)
results/group_<config>_<TS>.unfloored.csv      # PRE-FIX backup — ignore
patches/  (100)  flat/ (200 = 100 hemi x 2 cfg)  truegeo/ (100)  kring_cache/  truegeo_inflated/
figures/fig_group_raincloud_{paired,collapsed}_<config>_<TS>.{pdf,png}
figures/fig_group_raincloud_configs_<TS>.{pdf,png}     # robust vs tutte, paired by hemi, colored by hemi
```

### FreeSurfer-6 comparison — `DATA_ROOT/fs6_compare/`
```
true_comparison_fiducial.csv       # ** USE THIS ** corrected FS6-vs-pyflatten (40 hemi; see §3)
true_comparison_20subj.csv         # OLD (inflated-surface) — superseded, do not use
comparison.csv                     # n_flipped per method (metric-independent; still valid)
timing.csv                         # FS6 mris_flatten per-hemi runtime (never re-run)
runs/<subj>.<hemi>/
    <hemi>.patch.3d                # input patch (identical to the continuity patch; verified)
    <hemi>.flat                    # FreeSurfer mris_flatten output (the FS flat map)
    <hemi>.smoothwm -> ...fiducial # base-surface symlink
    truegeo_fiducial.npz           # fiducial reference built by fs6_recompare
    mris_flatten.log
```

---

## 3. CSV schemas

All CSVs have one row per measurement, a `status` column (`ok`/`error`), and an `error` column.
**Filter `status == "ok"`.** Distortions are **percent**; the reported metric is at each map's
own distance-optimal scale.

**`cores/cores_<TS>.csv`** (core-scaling; distortion is core-count-invariant — present at all 3
core counts to confirm determinism, so filter to one `n_cores`, e.g. 16, before aggregating):
```
timestamp, subject, hemi, config, n_cores, n_cores_pinned, n_vertices, n_faces,
prep_s, flatten_s, total_s,
kring_mean_distortion, kring_p90_distortion,        # k-ring diagnostic only — NOT the reported metric
n_flipped, frac_flipped, area_distortion,
true_local_mean, true_local_at_optscale,            # <- reported LOCAL distortion (<=30mm, at opt scale)
true_global_mean, true_global_at_optscale,          # <- reported GLOBAL distortion (all pairs, at opt scale)
opt_scale, n_pairs_global, flat_path, status, error
```

**`e2e/<config>_<TS>.csv`**: like cores but with `projection_s`/`prep_s`/`flatten_s` (per hemi,
16 cores); per-brain time = lh+rh. Has `true_local_at_optscale`, `true_global_at_optscale`, `opt_scale`.

**`results/group_<config>_<TS>.csv`**: like cores (per hemi, 8 CPUs) with `lane_cpus`;
key columns `true_local_at_optscale`, `true_global_at_optscale`, `opt_scale`, `n_flipped`,
`frac_flipped`, `flat_path`.

**`fs6_compare/true_comparison_fiducial.csv`** (the corrected FS-vs-pyflatten):
```
subject, hemi, method,            # method in {freesurfer6, robust_fast}
true_local_at_optscale, true_global_at_optscale, opt_scale, n_pairs, status, error
```
Pair by `(subject, hemi)` across the two `method` rows to compare.

---

## 4. The true-geodesic reference (`truegeo/<subj>_<hemi>.truegeo.npz`)

Per-hemi heat-method geodesic fields on the **fiducial** surface; this is what distortion is scored
against. NumPy `.npz` with keys:

| key | shape | meaning |
|---|---|---|
| `srcs` | `(200,)` | indices of 200 sampled source vertices (into the patch vertex array) |
| `geo`  | `(200, V)` | geodesic distance (mm) from each source to all `V` patch vertices |
| `R`    | scalar | local radius = 30.0 mm |
| `surface` | str | `"fiducial"` (sanity tag; pre-fix backups say nothing/inflated) |
| `sanitized` | bool | `True` = gross heat-failures zeroed (geo<½·chord) |

```python
import numpy as np
d = np.load("truegeo/sub-022_lh.truegeo.npz")
srcs, geo, R = d["srcs"], d["geo"], float(d["R"])
```

---

## 5. How distortion is computed (and the metric definition)

`d2d` = Euclidean distance in the flat map; `d_geo` = fiducial geodesic from the reference.
Per source `s`, over targets `j` with `geo[s,j] > 1e-6`: relative error `|d2d - d_geo| / d_geo`.
- **local** = mean over pairs with `d_geo <= R` (30 mm), at the global optimal scale.
- **global** = mean over all pairs, at the optimal scale (single scalar `s*` minimizing it).
- Reference built on the **fiducial** surface (NOT the patch's stored inflated coords); gross
  heat-solver failures (`d_geo < 0.5*chord`, physically impossible) are zeroed. No denominator floor.

The k-ring `*_distortion` columns are a biased internal diagnostic — **not** comparable across
methods/configs; do not report them. See [`truedist.py`](truedist.py) and `MEMORY` notes.

---

## 6. Loading a flat map & scoring it (copy-paste recipe)

Run from the **repo root** (so `import benchmark` resolves) with the package installed
(`pip install -e .`). Paths below are relative to a run directory.

```python
import numpy as np
from autoflatten.freesurfer import read_patch          # binary FreeSurfer patch reader
from benchmark import truedist

# flat map: read_patch -> (vertices (N,3), original_indices (N,), is_border (N,)); flat z~0
uv = read_patch("flat/sub-022_lh_robust_fast.flat.patch.3d")[0][:, :2].astype(float)

# reference (same vertex order as the flat map)
d = np.load("truegeo/sub-022_lh.truegeo.npz")
ref = {"srcs": d["srcs"], "geo": d["geo"], "R": float(d["R"])}

full = truedist.true_distortion_full(uv, ref)            # global + opt_scale
opt  = full["opt_scale"]
loc  = truedist.true_distortion(uv * opt, ref)           # local at opt scale
print(loc["true_mean_distortion"], full["true_global_at_optscale"], opt)
```
`read_patch` also reads the **input** 3D patch (`patches/*.autoflatten.patch.3d`, `fs6_compare/.../*.flat`)
and the FreeSurfer `<hemi>.flat`. The 3D patch's stored coords are the FreeSurfer **inflated** surface;
the anatomical/scoring surface is the **fiducial** (`NARRATIVES_FS/<subj>/surf/<hemi>.fiducial`).

---

## 7. Scripts (in `benchmark/`)

**Runners (produce the data):**
- [`time_cores.py`](time_cores.py) — core-scaling sweep (`cores_<TS>.csv`). Driver + `--worker`; resumable.
- [`time_e2e.py`](time_e2e.py) — end-to-end timing (`e2e/<config>_<TS>.csv`).
- [`group_flatten.py`](group_flatten.py) — 50-subject group run (`results/`, `flat/`, `truegeo/`),
  4 hemis × 8 CPUs; cohort `S_GROUP` + `COHORT.md`.
- [`fs6_recompare.py`](fs6_recompare.py) — FS6-vs-pyflatten on the corrected metric
  (`fs6_compare/true_comparison_fiducial.csv`).

**Metric / reference:**
- [`truedist.py`](truedist.py) — `compute_truegeo` (build fiducial reference), `true_distortion_full`,
  `true_distortion`.
- [`rebuild_truegeo.py`](rebuild_truegeo.py) — regenerate references on fiducial (`--sanitize` masks
  existing refs without re-solving heat).
- [`recompute_truedist.py`](recompute_truedist.py) — rewrite `true_*` columns of a CSV from saved
  flat+truegeo, **no re-flattening** (keeps `.unfloored.csv` backup).
- [`metrics.py`](metrics.py) — k-ring diagnostic (`per_patch_metrics`). [`projection.py`](projection.py)
  — FS-free cut projection.

**Figures:**
- [`fig_paper.py`](fig_paper.py) — `fig_core_scaling`, `fig_distortion`, `fig_distortion_paired`
  (per-hemi FS6 vs pyflatten), `fig_speed_accuracy`, `fig_runtime_e2e`.
  Run: `python -m benchmark.fig_paper --cores-dir <run> --e2e-dir <e2e_run> --ts <TS> --out-dir <run>/figures`
- [`fig_group_raincloud.py`](fig_group_raincloud.py) — group rainclouds.
  `--mode per-config` (paired LH/RH + collapsed, per `--config`); `--mode config-paired` (robust vs tutte).
- [`fig_flatmaps.py`](fig_flatmaps.py) — per-hemi 3-method flatmap panels.
- [`fig_pipeline.py`](fig_pipeline.py), [`fig_pipeline_detail.py`](fig_pipeline_detail.py) — schematics.

**Support:** `paths.py` (locations), `harness.py`/`metrics.py`, `ledger.py` (per-run provenance
records → `DATA_ROOT/ledger/`), `probe_*.py` / `validate_*.py` (exploratory, not paper outputs).

---

## 8. Headline numbers (corrected metric, for sanity-checking a reload)

- **Group** (100 hemi): robust_fast local median **14.3%** / global **12.2%**; tutte 14.2% / 12.1%.
  Configs ~tied; LH/RH symmetric; max local ~22% (genuine, not artifact).
- **FS6 vs pyflatten** (40 hemi, same patch): pyflatten better **local** 38/40 (14.85 vs 15.21% mean),
  FreeSurfer better **global** 38/40 (12.18 vs 12.45%) — within ~0.3 pp (local-vs-global trade-off).
- **Per-brain time @16c**: robust_fast ~9.5 min, tutte_default ~14.8 min.
