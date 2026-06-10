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

## Method note

Determinism confirmed bit-identical across reruns, so a single run per experiment is sound.
Compute is CPU-only (the box's GPUs are blocked by driver 440 / CUDA 10.2).

## Next ideas (untested)

- Reduce `k_ring` (7 → 5): attacks the dominant cost (k-ring geodesic computation, ~4 min + 237 MB
  cache/hemi). Changes the cache key, so needs recompute.
- Fewer epoch iterations from the better Tutte start.
- Combine: Tutte init as the new default `initial_projection`, initial NAR off by default.
