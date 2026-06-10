# Findings

Curated conclusions from the autoresearch loop. The full, append-only record (every run with
provenance) is the ledger at `/data2/projects/autoflatten/ledger/experiments.jsonl`, rendered to
`NOTEBOOK.md`. All numbers below are CPU-only on the public Narratives benchmark.

## 1. Flip-free (Tutte) init is a validated win: equal quality, ~37% faster

Replacing the FreeSurfer-style normal-projection + **initial** negative-area-removal (NAR) with a
**Tutte embedding** (libigl harmonic map, boundary pinned to a circle — injective by Tutte's
theorem) and feeding it into the existing refinement. The Tutte map starts at **0 flipped
triangles** (vs ~141k for the projection), so the ~4-min initial NAR phase is unnecessary.

Verified across **4 hemispheres / 2 subjects** (`sub-022`, `sub-026`):

| | mean distortion | total flips | mean runtime |
|---|---|---|---|
| baseline (FS-clone) | 14.73% | 121 | 702 s |
| **Tutte init** | 14.75% | 121 | **442 s (−37%)** |

Distortion identical (+0.03pp), total flips identical (121→121), runtime down 37%. Maps are
visually clean (single blob, smooth boundary, no folds) on both subjects.

## 2. The *final* NAR and spring smoothing are NOT removable

Tempting because dropping them is faster and even *lowers* the distortion number — but the number
lies:

| variant (sub-022 lh) | distortion | flipped | runtime |
|---|---|---|---|
| Tutte init (full refinement) | 15.25% | **24** | 454 s |
| Tutte, skip final NAR | 13.38% | **19458** | 256 s |
| Tutte, skip final NAR + spring ("lean") | **13.21%** | **21481** | 263 s |
| Tutte, skip spring only | 15.50% | 75 | 473 s |

Skipping the final NAR gives the lowest distortion in the whole study (13.21%) but **~21000
flipped triangles** — a folded, invalid flatmap (visible as dark fold streaks). The final NAR and
spring phases are doing real work cleaning up flips introduced by the metric epochs. **Only the
initial NAR is removable.**

## 3. LSCM init ≈ Tutte (no advantage)

LSCM (conformal) init: 15.26% distortion, 34 flips, 444 s on `sub-022 lh` — essentially tied with
Tutte (15.25%, 24, 454 s), slightly more flips. Tutte is preferred (flip-free *guarantee*).

## 4. Swapping the optimization *algorithm* does not help — the energy is the issue

Tested replacing the FreeSurfer-style line-search GD with off-the-shelf optimizers on the *same*
metric+area energy, from the flip-free Tutte init (`benchmark/probe_optimizer.py`):

- **L-BFGS / CG on the soft energy fold.** Full L-BFGS reaches **10.31% distortion in 83 s (≈8×
  faster)** but **58025 flipped triangles** — it minimizes distance error by folding the mesh.
- **Pareto-worse than the baseline.** Sweeping the area weight `l_nlarea` only trades one for the
  other: to get flips down to baseline (~24) L-BFGS needs ~25–29% distortion (vs baseline 15% @ 24
  flips). The FreeSurfer optimizer sits on a strictly better frontier.
- **The epoch weight schedule doesn't rescue it.** Running L-BFGS through the area-dominant →
  distance-dominant schedule still folds (71269 flips at the distance-dominant stage).
- **A flip barrier keeps it injective but stalls.** Adding a one-sided area barrier (active only as
  area→0) holds flips at ~0, but L-BFGS then can't reduce distortion at all (stuck at the init's
  33.7%): from the Tutte init, the *local* distance-reducing moves all push triangles toward
  folding, which the barrier blocks.

**Conclusion.** The FreeSurfer-style optimizer's value is **not** the line-search algorithm — it is
the **multi-level gradient smoothing** (spatially-coherent, coarse-to-fine moves that reduce
distortion *without* folding) plus the NAR passes. A generic first-order solver on per-vertex
gradients makes high-frequency moves that fold. So a faster/better optimizer alone is a dead end on
this energy; the productive directions are **multiscale / hierarchical (geometric-multigrid)
optimization** or a proper **SLIM** local-global solver (symmetric-Dirichlet energy with a
flip-preventing line search), not a drop-in optimizer swap. (`optax`/`jaxopt` are installed for
this.)

## Method note

Determinism confirmed bit-identical across reruns, so a single run per experiment is sound.
Compute is CPU-only (the box's GPUs are blocked by driver 440 / CUDA 10.2).

## Next ideas (untested)

- Reduce `k_ring` (7 → 5): attacks the dominant cost (k-ring geodesic computation, ~4 min + 237 MB
  cache/hemi). Changes the cache key, so needs recompute.
- Fewer epoch iterations from the better Tutte start.
- Combine: Tutte init as the new default `initial_projection`, initial NAR off by default.
