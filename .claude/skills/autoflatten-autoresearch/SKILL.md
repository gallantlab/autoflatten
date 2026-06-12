---
name: autoflatten-autoresearch
description: >-
  Resume the AutoFlatten flattening-optimization autoresearch effort. Use when the user
  says things like "resume the autoflatten autoresearch", "continue optimizing the
  flattening", "what have we tried so far", "run the flatten benchmark", or asks to
  propose/test a new flattening method or config. Orients a fresh session to the plan,
  the provenance ledger, the data, and the conventions, and drives the experiment loop.
---

# AutoFlatten autoresearch

A benchmark-driven effort to improve the **pyflatten flattening stage** (and ultimately
replace its FreeSurfer-clone optimizer with a principled method), with every experiment
logged so the *process* is shareable in a paper.

## 0. Recover state FIRST (before proposing anything)

1. Read the design + status: `benchmark/PLAN.md` and `benchmark/README.md` (in this repo).
2. Read the ledger to see everything tried so far, the current baseline, and the last
   planned step:
   - Ledger: `/data2/projects/autoflatten/ledger/experiments.jsonl` (append-only JSONL).
   - Rendered notebook: `/data2/projects/autoflatten/NOTEBOOK.md`
     (regenerate with `python -m benchmark.report`).
   - Look at the most recent records' `metrics` (the multi-objective vector) and any
     `decision.next_step`. **Do not re-run experiments already in the ledger.**
3. Confirm the manifest exists: `/data2/projects/autoflatten/manifest.json`
   (else `python -m benchmark.build_dataset`).

## 1. Environment & constraints

- Use the repo's **uv** venv: `uv sync --extra bench`, then `.venv/bin/python`.
- **CPU only.** GPUs are blocked by driver 440 / CUDA 10.2 — do **not** install/attempt
  CUDA jax unless the user says the driver was upgraded (≥525).
- Dev on a **tiny subset** (`--dev`, 2 subjects / a few hemispheres); scale to the full
  82 only once a change looks good. A single hemisphere takes ~8–15 min on CPU.
- **Output locations are pinned:** code + docs in the repo; *all generated artifacts* go
  to `/data2/projects/autoflatten/` (see `benchmark/paths.py`,
  override `AUTOFLATTEN_BENCH_ROOT`). Never write generated files into the repo.
- Data is the public **Narratives** set (OpenNeuro `ds002345`); never the private
  `all-subjects/` lab data.

## 2. Core objects

- `benchmark/harness.py` — `evaluate(entries, config, method=...)`; geometry +
  k-ring caching; `register_flatten_fn(name, fn)` to add a method. A `flatten_fn` has
  signature `flatten_fn(flattener) -> uv` (shape `(V, 2)`), scored uniformly.
- `benchmark/metrics.py` — `per_patch_metrics(uv, flattener)` and `aggregate(...)`.
  Objective stays **geodesic distance distortion** (+ flips, robustness, runtime) — never
  switch to pure conformality.
- `benchmark/ledger.py` — `new_record(kind, label, ...)` + `Ledger().append(record)`.

## 3. The experiment loop

For each idea:

1. **State a hypothesis** (what you expect to improve and why).
2. **Implement** a `flatten_fn` (or a `FlattenConfig` change). The primary open lead is a
   **Tutte/LSCM flip-free init** that removes the ~4-min initial NAR phase — see
   `benchmark/probe_tutte_init.py` (Phase B in the plan).
3. **Evaluate on the dev subset**, then promote to the full train split if promising:
   ```bash
   python -m benchmark.run_baseline --dev          # reference
   # ... your probe/optimize script, same evaluate() harness ...
   ```
4. **Append a ledger record** with `kind`, `label`, `method`, `metrics`, `per_subject`,
   `repro_command`, and a **decision trace** in `record.decision`:
   `{hypothesis, rationale, conclusion, next_step}`.
5. **Commit** the code change in this repo (the ledger pins the commit SHA), then
   `python -m benchmark.report` to refresh `NOTEBOOK.md`.
6. If it **Pareto-beats** the baseline (lower distortion and/or fewer flips and/or faster,
   none worse), promote to the **full 82-subject** run and validate on the **holdout**
   split before claiming a win.

## 4. Guardrails

- Every experiment is pinned to a git commit; commit before/after running.
- Determinism is assumed (one run/experiment) but **re-assert it** when you change the
  optimizer (`run_baseline.py --check-determinism`).
- Never silently drop subjects — `evaluate` records per-subject `status="error"`; surface
  failures in the conclusion.
- Keep the ledger append-only; to revise a result, append a new record.
