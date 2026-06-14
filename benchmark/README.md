# AutoFlatten benchmark / autoresearch harness

Reproducible, provenance-tracked benchmark for optimizing the **pyflatten flattening
stage**. The full design and rationale live in [`PLAN.md`](PLAN.md); this README is the
operational quick-start. Session-recovery instructions for an agent live in
[`.claude/skills/autoflatten-autoresearch/SKILL.md`](../.claude/skills/autoflatten-autoresearch/SKILL.md).

## Layout (where things go)

- **Code + docs → this repo** (`benchmark/*.py`, `PLAN.md`, `README.md`).
- **Every generated artifact → `/data2/projects/autoflatten/`** (override with
  `AUTOFLATTEN_BENCH_ROOT`): the JSONL ledger, `NOTEBOOK.md`, `manifest.json`, candidate
  configs, flat patches, k-ring caches, and the Optuna DB. *Nothing generated lands in
  the repo.* See [`paths.py`](paths.py).

## Data

Public **Narratives** dataset (OpenNeuro `ds002345`). 82 subjects already have a
projection patch (`{hemi}.autoflatten.patch.3d`) + a base surface (`{hemi}.fiducial`)
under the FreeSurfer derivatives, so the benchmark reuses those and **needs no FreeSurfer
and no fetching**. The private lab set (`/data2/freesurfer_subjects/all-subjects/`) is
*not* used — it isn't publicly shareable.

## Compute

**CPU only.** The machine's GPUs are blocked by an old driver (440 / CUDA 10.2), too old
for any JAX `autoflatten` supports. Don't attempt CUDA unless a sysadmin upgrades the
driver to ≥525. The biggest speed win (Tutte flip-free init) is algorithmic anyway.

## Quick start

```bash
uv sync --extra bench

# 1. Build the benchmark manifest (deterministic train/holdout split)
python -m benchmark.build_dataset                # ~18 subjects
python -m benchmark.build_dataset --dev          # tiny: 2 subjects (dev loop)

# 2. Baseline (current FreeSurfer-clone defaults) + determinism check
python -m benchmark.run_baseline --dev --check-determinism

# 3. Render the lab notebook from the ledger
python -m benchmark.report
```

## How the harness is structured

- [`paths.py`](paths.py) — the two pinned roots.
- [`ledger.py`](ledger.py) — append-only JSONL provenance ledger (commit SHA, env, seeds,
  metrics, artifacts, repro command, decision trace). **Every run appends a record.**
- [`metrics.py`](metrics.py) — uniform per-patch metrics (mean/p90 % distance error,
  flipped triangles, area distortion) computed from `uv` independent of method, plus
  cross-subject aggregation into the multi-objective vector.
- [`harness.py`](harness.py) — owns geometry (loads patch+surface, computes/caches k-ring
  targets), runs a `flatten_fn(flattener) -> uv`, and scores it. The current pipeline is
  one `flatten_fn`; alternatives register via `register_flatten_fn`.
- [`build_dataset.py`](build_dataset.py), [`run_baseline.py`](run_baseline.py),
  [`report.py`](report.py) — the Phase-A commands above.
- [`probe_tutte_init.py`](probe_tutte_init.py) — the flip-free (Tutte/LSCM) init probe
  (Phase B primary experiment): injects a guaranteed-injective embedding and disables the
  initial NAR, reusing the existing refinement.
- [`experiment.py`](experiment.py) — **general autoresearch runner**. Parametrizes the init
  method (`--init projection|tutte|lscm`) and refinement toggles
  (`--skip-initial-nar`/`--skip-final-nar`/`--skip-spring`/`--skip-epoch`, `--k-ring`),
  logging each variant to the ledger with a decision trace. This is how optimization ideas
  are fanned out.
- [`plot.py`](plot.py) — fast flatmap renderer for visual verification of any logged
  experiment (`python -m benchmark.plot <experiment_id>`; `--full` for the slow 3-panel).
- `optimize.py` — optional Optuna HPO (Phase C, not yet built).

## Running an optimization experiment

```bash
# e.g. test whether a flip-free start makes the final NAR phase removable:
python -m benchmark.experiment --init tutte --skip-final-nar --subset 1 --save \
    --label tutte_nofinalnar --hypothesis "flip-free start keeps the map near-injective"
python -m benchmark.plot --latest experiment      # eyeball the result
python -m benchmark.report                          # refresh NOTEBOOK.md
```

## Full public replication path

We skip projection locally (reusing cached patches), but the whole pipeline is publicly
reproducible: get the surfaces from OpenNeuro/datalad `ds002345`, install FreeSurfer +
`autoflatten`, run `autoflatten project` then `autoflatten flatten`. The manifest records
the dataset DOI and datalad commit.
