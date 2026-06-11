# Findings

Curated conclusions from the autoresearch loop. The full, append-only record (every run with
provenance) is the ledger at `/data2/projects/autoflatten/ledger/experiments.jsonl`, rendered to
`NOTEBOOK.md`. All numbers below are CPU-only on the public Narratives benchmark.

## Executive summary

- **Speed (validated, shippable):** a Tutte flip-free init removes the ~4-min initial NAR (§1), and
  stacked config levers (lean line search, sparser k-ring, fewer iters, capped smoothing) give a
  combined **~3.4× speedup** that is **not overfit** — confirmed across **9 hemispheres / 7 subjects**,
  with the 5 held-out subjects matching the tuned one (§7, §12). On the *faithful* (global true-geodesic)
  metric the fast config is in fact **slightly better** than baseline (−0.73pp), not worse; the earlier
  "+0.37pp" was an artifact of the miscalibrated k-ring metric (§12). This is the main practical win.
- **Quality / distance error:** the FreeSurfer-style multiscale line-search optimizer is **hard to
  beat** — optimizer swaps fold (§4), Adam can't span the multiscale step range (§6), spectral
  multigrid is elegant but not faster (§5).
- **Metric caveat (important for the paper):** the k-ring energy metric is miscalibrated and a purely
  **local** distance metric is *gameable* — a conformal Tutte disk wins it while being a degenerate
  flatmap (§9c). Score distance distortion **globally** (all-pairs geodesics), not locally.
- **Objective (grounded in Fischl 1999, but the code diverges from the paper):** the goal is **metric
  (distance) distortion**; area is a *dependent byproduct*, not a separate objective (§9, §10). The
  implemented area term is a pure fold barrier.
- **Where the implementation drifts from that objective:** the `Dijkstra/1.207` target correction is
  slightly too compact and `scale_to_area` is not distance-optimal (§9d) — but recalibrating the
  correction is **not a robust default** (subject-specific optimum, unpredictable from local geometry;
  §10–11). The only consistently-safe tweak is a distance-optimal output scale (small).
- **Net:** the pipeline is well-tuned; config-lever gains on distance error are small and non-robust.
  The one remaining path to a *larger robust* reduction is **long-range geodesic anchors in the energy**
  (the paper's own argument) — a real energy change, left as future work.

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

## 5. Spectral (manifold-harmonic) multigrid — elegant, flip-free, but not faster

`benchmark/probe_multigrid.py`. Parametrize the flatmap in the lowest-`K` cotangent-Laplacian
eigenvectors and optimize coarse-to-fine (add modes progressively). A band-limited map *cannot*
make local folds, so this is **flip-free by construction**:

| modes | distortion | flips | cumulative time |
|---|---|---|---|
| 20 | 26.89% | 12 | 13s |
| 50 | 24.96% | 13 | 28s |
| 100 | 23.78% | 13 | 44s |
| 200 | 22.35% | 19 | 66s |

So the coarse spectral solve is a genuinely nice **fast, flip-free approximate flattener** (66s, ~0
flips) — contrast L-BFGS's 58025 flips. But it is **band-limited**: 200 smooth modes can't represent
fine detail, so distortion floors at ~22%.

Used as the coarse leg of a multigrid V-cycle (spectral init → full-resolution refine):
- spectral coarse (22.35%) → full refine → **15.32% / 26 flips / 430s refine**.
- Even with a *shortened* refine (half iters): 15.37% / 28 flips / 347s.

It **matches** baseline quality but is **not faster**: the spectral overhead (eigsh ~75s one-time +
coarse solve ~66s) plus the refine (~350–430s) totals ≈ 490s vs Tutte+pipeline's 454s. The better
init (22% vs Tutte's 33%) doesn't shorten the refine enough to pay for the overhead, because the
refinement runs a largely fixed schedule. A true win would need a **mesh-decimation** multigrid
(each level full-DOF but few vertices) rather than a band-limited basis.

## 6. Adam (and fixed-step methods) can't replace the line search

Adam (optax) driving the same multi-level gradient smoothing as the baseline:
- **Fixed lr**: lr=0.001 is stable but crawls (33.7%→31.5%, nowhere near 15%); lr≥0.01 reduces
  distortion but folds, and **diverges at the finest smoothing level** (up to 175855 flips).
- **Per-level lr schedule** (large→small with the smoothing level): still folds (76552 flips).

Root cause: the optimal step size spans *orders of magnitude* across the smoothing schedule (large at
coarse `n_avg`, tiny at fine). A fixed or simply-scheduled step is either too slow at coarse scales
or unstable at fine scales. The baseline's **per-iteration line search** is precisely what adapts the
step across scales — that, plus the gradient smoothing, is the load-bearing machinery.

## Overall conclusion on the optimizer

Across L-BFGS, CG, Adam, flip barriers, and a spectral multigrid, **nothing beats the FreeSurfer-style
multiscale line-search GD** on the speed/quality/flip Pareto for this energy. Strong optimizers fold
without the smoothing; fixed-step optimizers can't span the multiscale step range; the eigenbasis is
flip-free but band-limited. The genuinely better directions remaining are a **mesh-decimation
geometric multigrid** or a proper **SLIM** local-global solver — real reimplementations, not drop-in
swaps. The one validated free win remains the **Tutte flip-free init** (§1).

## 7. Speeding up the FreeSurfer-style optimizer (the productive direction)

Rather than replace the optimizer, **accelerate it**. Profiling one iteration (193k-vertex
hemisphere, CPU):

| component | cost / iter | notes |
|---|---|---|
| gradient (J_d + J_a) | 177 ms | iterates over ~16M k-ring edges |
| line search (15 pts, vmap) | 272 ms | also iterates over k-ring edges |
| `smooth_gradient`, n_avg=1 | 1 ms | fine levels |
| `smooth_gradient`, n_avg=256 | 243 ms | |
| `smooth_gradient`, n_avg=1024 | **902 ms** | 1024 sequential neighbor-averaging passes |

So per iteration ≈ gradient + line search (~450 ms, constant) + smoothing (1→902 ms by level).
The levers, each **measured on sub-022 lh** (Tutte init; reference Tutte+full = 15.25% / 24 / 454s):

| lever | distortion | flips | runtime | vs 454s |
|---|---|---|---|---|
| line search 15 → 7 points | 15.28% | 28 | 319s | **−30%, ~free** |
| k-ring neighbors 12 → 6 | 15.37% | 30 | 303s | −33%, +0.1pp |
| k_ring 7 → 5 | 15.73% | 24 | 340s | −25%, +0.5pp |
| iters/level 40 → 20 | 15.37% | 28 | 362s | −20%, +0.1pp |
| drop n_avg=1024 level | 15.35% | 35 | 422s | −7% (few coarse iters) |
| **combo** (ls7 + iters25 + smooth256) | 15.51% | 28 | **262s** | **−42%** |

These levers are largely **independent and compounding**, and stack on top of the Tutte init
(which removed the ~4-min initial NAR). The combined "fast" config reaches **262s vs the original
658s baseline — ~60% faster** — at +0.31pp distortion and comparable flips. The biggest near-free
win is the **line search point count** (15→7, no measurable quality cost); the biggest single lever
is **k-ring density** (fewer neighbors cuts gradient + line search + the one-time k-ring computation
+ the 237MB cache together). Capping the coarse smoothing helps less than expected because few
iterations actually run at the coarsest level.

**Validated stacked "fast" config** (Tutte init + n_neighbors 6 + line-search 7 + iters/level 25 +
smoothing cap 256), across **4 hemispheres / 2 subjects**:

| | mean distortion | total flips | mean runtime |
|---|---|---|---|
| baseline (n=4) | 14.73% | 121 | 702 s |
| **fast_ultimate (n=4)** | 15.10% | 237 | **194 s (−72%, 3.6×)** |

+0.37pp distortion, flips still 0.06% of faces, and visually clean maps on both subjects.
(**Update — see §12:** that +0.37pp is on the *miscalibrated* k-ring metric; on the faithful global
true-geodesic metric the fast config is actually −0.73pp *better*, validated on 9 hemispheres.)

**Takeaway:** the practical, low-risk path to a faster pipeline is not a new optimizer but
(1) Tutte init, (2) a leaner line search, (3) a sparser k-ring, (4) fewer iters/level — each a small
config change, together a **~3.6× speedup** at near-baseline quality. (A spectral/low-rank
acceleration of the coarse `smooth_gradient` is a deeper exact win if the coarse levels ever dominate.)

## 8. Improving the energy (quality at the cost of speed)

Built a **true geodesic distortion** yardstick (libigl heat method, 200 sampled sources, local
≤30mm pairs) to score quality independently of the k-ring targets. It immediately showed the k-ring
metric is **miscalibrated as an absolute number**: the Tutte init reads 33.7% on the k-ring metric
but ~16% in true geodesic terms, and the metric even *mis-ranks* maps (Tutte better than the spectral
map in true terms, opposite of the k-ring ranking). Calibration: the k-ring targets (Dijkstra ×
1.207) are ~6–9% larger than true geodesics at the very local scale (mean ratio 1.094, median 1.059).

**But recalibrating the correction does NOT improve quality — it hurts.** Optimizing from the Tutte
init with correction ∈ {1.207, 1.10, 1.00} and scoring true distortion:

| correction | k-ring metric | true geodesic | flips |
|---|---|---|---|
| **1.207 (default)** | 15.26% | **18.15%** | 30 |
| 1.10 | 15.24% | 21.66% | 26 |
| 1.00 | 15.26% | 26.88% | 35 |

True distortion gets monotonically *worse* as the correction shrinks. Interpretation: the local
Dijkstra/geodesic ratio (~1.09) does not represent the full range, and — more importantly — a
flatmap intrinsically **compresses** medium-range distances (curvature; Gauss). The 1.207 factor acts
as a useful global **pre-stretch** that compensates for that compression over the 0–30mm range, so a
"locally accurate" smaller correction leaves the map too compact and worse overall. **The default
1.207 is empirically near-optimal; recalibration is a dead end.** (The k-ring metric is still worth
recalibrating for *interpretation*, since it overstates absolute distortion — but not for the search.)

Also notable: the optimized maps (~18% true) are not better than the raw Tutte init (~16% true) on
this medium-range metric — local k-ring fitting trades some medium-range geodesic accuracy. Which map
is "best" depends on the distortion metric (local vs medium-range), a point worth making in the paper.

**Remaining genuine lever (untested, heavier):** replace Dijkstra × constant with **actual per-edge
geodesic targets** (heat/exact) in the energy — a constant can't capture the spatially-varying
Dijkstra/geodesic ratio. That is the real "better energy at the cost of a slight slowdown," but it
needs per-vertex geodesic computation (expensive) and is left as the next step.

## 9. Local vs global distance metric, and a validated free win: distance-optimal output scale

Pushing on "reduce distance error at a slight slowdown" (all on `sub-022 lh`, scored with the
heat-method true-geodesic yardstick; `benchmark/probe_truedist.py`, `benchmark/truedist.py`):

**a) Direct measurement of the Dijkstra/geodesic relationship.** Using the 200 saved heat-geodesic
source fields vs raw graph Dijkstra: raw Dijkstra overestimates the true geodesic by only **~7.8%**
(ratio 1.078), and that ratio is **nearly flat across 0–30 mm** (1.12→1.07). The code computes
`target = Dijkstra / 1.207`, so the **targets are ~11% *smaller* than true geodesics** (target/true =
0.893). Two consequences: (i) a *distance-dependent* correction is pointless (no distance structure to
exploit), and (ii) §8's "shrink the correction" sweep only ever made targets *larger*; the optimum is
at/above 1.207, confirming 1.207 is near-optimal. The targets are a **confirmed dead end**.

**b) `k_ring` 7→11 does not help.** True distortion 18.84%→18.66% (−0.18pp) but flips 24→444 and
runtime 445→649 s (+46%). More constraints with the same flawed targets just add folding pressure.

**c) The local metric is gameable — a conformal disk beats a real flatmap on it.** Skipping all metric
epochs (just flip-cleaning the Tutte init) gives the *lowest* local (≤30 mm) true distortion in the
study — **16.26%, 0 flips, 6 s** — but the map is a **featureless disk** (Tutte pins the boundary to a
circle, destroying all anatomical shape). Lighter-touch variants (skip epoch_3 / epoch_2+3) are *worse*
(18.9–19.0%), not better. So the metric epochs are not "degrading" quality; the **local ≤30 mm metric
simply doesn't see the global/boundary area distortion** and mis-ranks a useless disk above a correct
flatmap. Lesson for the paper: report distance distortion **globally**, not just locally.

| map | local ≤30 mm | **global (all pairs)** | flips | shape |
|---|---|---|---|---|
| clean_init (Tutte disk) | **16.26%** | 13.04% | 0 | degenerate disk |
| full pipeline (fast_ultimate) | 17.93% | **11.35%** | 33 | correct flatmap |

On the **global** metric (all geodesic pairs, no 30 mm cap) the full pipeline is **best** and the disk is
**worst** — the correct ranking. The pipeline is actually better at long range (11.35%) than short range
(17.93%); its only real weakness is a slight **global scale** bias.

**d) Output scale is not metric-optimal — but the picture is subtle (grounded in Fischl 1999).**
Fischl, Sereno & Dale (1999), §2.1–2.3: the flattening energy is `J = λ_d·J_d + λ_a·J_a` where
`J_d = (1/2V) Σ_i Σ_{n∈N(i)} (d_in^t − d_in^0)^2` is **metric (distance) distortion** — *the* objective —
and `J_a = (1/2T) Σ_i P(A_i)(A_i − A_i^0)^2` with `P(A_i)=1 iff A_i ≤ 0`. The area term is **gated to
folded (negative-area) triangles only**; valid triangles' area magnitude is unconstrained. So FreeSurfer
has **no area-preservation objective** — `J_a` only removes folds, and area is preserved only as a
*byproduct* of distance preservation (an isometry preserves both; distance is the stronger property).
The final `scale_to_area` (`s = √(orig_area/total_area)`) is therefore a **display convention**, not part
of the objective.

Refitting a single global scale `s` on the optimized map exposes that the area-matched scale is not
distortion-optimal — and the two distance metrics disagree on direction:

| objective | optimal scale | note |
|---|---|---|
| **true geodesic** (heat, the paper's *intent*) | `s ≈ 1.04` (expand) | area-matched map is ~4% too small vs truth; rescale cuts true distortion 11.35%→10.88% (−0.48pp), reproduced on held-out sources (11.31%→10.74%) |
| **k-ring surrogate `J_d`** (the energy actually minimized, `Dijkstra/1.207` targets) | `s ≈ 0.98` (shrink) | the ~11% too-compact targets pull the surrogate optimum the *wrong* way |

So this is **not** a clean free win: the true-geodesic gain is real (it improves the paper's actual
objective) but it is a **symptom of the target-compaction bias (a)**, and the implemented surrogate energy
prefers the opposite scale. Shippable form: after optimization, choose the output scale that minimizes
distortion against a **true-geodesic sample**, rather than matching total area (and ideally fix the target
calibration upstream). Validated on one hemisphere; confirm multi-hemi before changing a default.

**Area as an objective?** Per Fischl 1999 it deliberately is *not* one. Adding an area-preservation term
would be redundant in the isometric limit and would *fight* `J_d` in the real (non-isometric) regime,
trading distance fidelity for area fidelity. Only worth it if the scientific use is reading cortical
*area* off the map (an equiareal/authalic objective) — a different goal from Fischl's metric-distance one.

## 10. Target correction, re-judged on the GLOBAL metric (reverses §8, but only partly)

§8 swept the graph-distance correction and concluded "1.207 is near-optimal, recalibration is a dead
end" — but that was scored on the **local ≤30 mm** metric, which §9(c) showed is gameable. Re-running
the sweep scored on the faithful **global** metric (`benchmark/probe_truedist.py --target-scale`, which
rescales the cached targets, equivalent to changing the correction without a cache rebuild; eff.
correction = 1.207 / target_scale) flips the conclusion — *larger* targets (smaller correction, toward
the directly-measured Dijkstra/geodesic ratio ~1.08) **reduce** global true distortion, but the size of
the win is **subject-dependent**.

Global true distortion at each map's distance-optimal scale (removes the global-zoom confound):

| target_scale (eff. correction) | sub-022 lh | sub-022 rh | sub-026 lh | mean |
|---|---|---|---|---|
| 1.00 (1.207, default) | 11.65% | 11.30% | 11.78% | 11.58% |
| 1.05 (1.15) | 11.34% | 11.20% | 11.84% | 11.46% |
| 1.10 (1.10) | **11.13%** | 11.20% | 11.82% | 11.38% |

- On **sub-022** (both hemispheres) calibrating toward ~1.10 helps clearly (−0.1 to −0.5pp; lh is
  monotonic out to 1.10). At ts=1.10 the map beats even the *best post-hoc rescale of the baseline*
  (11.13 vs 11.65 on lh), so calibrated targets reshape the map, not just rezoom it.
- On **sub-026 lh** it is neutral-to-slightly-worse (+0.04–0.06pp). The optimal correction is
  subject-dependent, and ts=1.10 *overshoots* (its post-hoc s* drops below 1.0).
- Flips stay comparable throughout (24–40), so calibration is not trading validity for distance.

**Takeaways.** (i) §8's "dead end" was a metric artifact — on the right (global) objective the correction
*does* matter and the FreeSurfer 1.207 is **slightly too large** (too-compact targets). (ii) But the gain
is small (~0.2pp mean) and not universal, so the shippable change is modest: nudge the effective
correction toward ~1.10–1.15 (the measured ratio), expect a small global-distortion reduction on most
subjects, and pair it with a distance-optimal output scale (§9d) rather than `scale_to_area`. (iii) The
consistent, always-safe component is the output-scale fix (`s*≈1.02` on all three hemispheres). A larger,
robust reduction would need **long-range geodesic anchors** in the energy (the paper's own point that
long-range distances are required to unfold) — a real energy change with weight-tuning, not a config
knob; left as the next step pending a decision on scope.

## 11. Broadened validation: the correction optimum is subject-specific and unpredictable

Per the §10 caveat, the correction sweep was broadened to **8 hemispheres across 7 subjects** (added
sub-041/052/059/066/075 lh to the benchmark, each with its own heat-geodesic reference; scored on global
true distortion at each map's own optimal scale).

| subject | Dijkstra/geo ratio | ts=1.0 (1.207) | ts=1.05 (1.15) | ts=1.10 (1.10) | Δ(1.0→1.10) |
|---|---|---|---|---|---|
| sub-041 lh | 1.087 | 12.33% | — | 11.78% | **−0.55** |
| sub-022 lh | 1.078 | 11.65% | 11.34% | 11.13% | **−0.52** |
| sub-052 lh | 1.089 | 12.02% | — | 11.69% | **−0.33** |
| sub-059 lh | 1.093 | 12.50% | — | 12.21% | **−0.29** |
| sub-022 rh | 1.082 | 11.30% | — | 11.20% | −0.10 |
| sub-026 lh | 1.084 | 11.78% | — | 11.82% | +0.04 |
| sub-066 lh | 1.085 | 11.59% | 11.64% | 11.63% | +0.05 |
| sub-075 lh | 1.076 | 12.28% | 12.64% | 12.50% | +0.22 |

**Mean Δ = −0.19pp; 5/8 helped, 2 neutral, 1 (sub-075) hurt.** Two robust negatives emerged:

1. **The local Dijkstra/geodesic ratio is nearly subject-invariant (1.076–1.093)** — so all subjects'
   targets are ~8% too compact in the *same* way. It therefore **cannot predict** the optimum: sub-075
   has the *lowest* ratio yet is *hurt* by a smaller correction; sub-059 has the *highest* and is
   *helped*. Per-subject **auto-calibration to the measured ratio would not work** (it would apply ~the
   same correction to everyone).
2. **No correction change is universally safe.** A milder ts=1.05 does not rescue the non-helpers — on
   sub-075 it is *worse* than ts=1.10 (12.64 vs 12.50 vs 12.28 baseline); sub-066 also degrades at any
   increase. 5 subjects want ~1.10, 3 genuinely prefer the original 1.207, and nothing local separates
   the groups (the difference is global geometry / optimization dynamics).

**Conclusion.** Changing the default correction is **not a robust win**: ~−0.2pp on average but with
real per-subject downside (up to +0.35pp). FreeSurfer's 1.207 is defensible as a robust compromise even
though it is slightly too large for the average subject on the global metric. The config-lever avenue is
now **exhausted**: the only consistently non-harmful operation is the distance-optimal **output scale**
(a 1-parameter minimization, ≥0 by construction, but small — `s*` ranges 0.995–1.020 across subjects), and
the only remaining path to a *larger, robust* reduction is **long-range geodesic anchors in the energy**
(a real energy change, not a config knob).

## 12. Full-benchmark validation of the fast config — not overfit (corrects §7's quality claim)

The §7 speed levers were tuned on a **single hemisphere** (`sub-022 lh`) and the stacked
`fast_ultimate` config had only been confirmed on 4 hemispheres / 2 subjects — a real overfit risk.
Re-validated across **all 9 manifest hemispheres** (`benchmark/validate_speed.py`), running baseline
and fast back-to-back per hemisphere (so the speedup ratio is internally valid), and grouping by
whether the hemisphere was used to tune the levers: **tuned** (`sub-022 lh`), **seen** (in the n=4
stack), **held-out** (`sub-041/052/059/066/075 lh`, never used to tune anything).

Quality is scored with the **energy-independent global true-geodesic** metric at each map's
distance-optimal scale — *not* raw k-ring distortion, which is incomparable across n12 vs n6 (the §8
trap). `dq` = fast − baseline (negative = fast better).

| subject | group | base s | fast s | speedup | base q% | fast q% | dq (pp) | flips b/f |
|---|---|---|---|---|---|---|---|---|
| sub-022 lh | tuned | 641 | 187 | 3.42 | 11.76 | 10.88 | −0.89 | 25/33 |
| sub-022 rh | seen | 667 | 185 | 3.60 | 11.46 | 10.94 | −0.52 | 37/129 |
| sub-026 lh | seen | 626 | 195 | 3.21 | 11.96 | 11.20 | −0.76 | 36/41 |
| sub-026 rh | seen | 696 | 211 | 3.31 | — | — | — | 23/34 |
| sub-041 lh | held-out | 758 | 221 | 3.43 | 12.25 | 11.37 | −0.88 | 81/120 |
| sub-052 lh | held-out | 561 | 180 | 3.12 | 12.22 | 11.20 | −1.02 | 39/44 |
| sub-059 lh | held-out | 759 | 222 | 3.42 | 12.60 | 11.77 | −0.83 | 353/88 |
| sub-066 lh | held-out | 699 | 208 | 3.37 | 11.71 | 11.27 | −0.44 | 54/55 |
| sub-075 lh | held-out | 615 | 183 | 3.35 | 12.45 | 11.97 | −0.48 | 60/65 |

**By group:** tuned 3.42× (dq −0.89); seen 3.37× [3.21–3.60] (dq −0.64); **held-out 3.34× [3.12–3.43]
(dq −0.73)**. **Overall 9 hemis: 3.36× [3.12–3.60], dq −0.73pp.**

**Takeaways.**
1. **Not overfit.** The held-out group matches the tuned hemisphere on *both* axes — speedup 3.34×
   vs 3.42×, quality −0.73 vs −0.89pp. The win generalizes across subjects.
2. **§7's quality claim was a metric artifact, now corrected.** §7 reported fast as **+0.37pp worse**,
   but that was the raw k-ring metric comparing n6 targets to n12 targets (incomparable; §8). On the
   faithful global true-geodesic metric the fast config is **−0.73pp better** on every scored
   hemisphere (8/8). The sparser k-ring/leaner schedule does not cost quality — if anything the Tutte
   init + distance-optimal scoring helps. The honest headline is **~3.4× faster at equal-or-better
   distance quality**.
3. **Flips stay negligible.** Fast has slightly more flipped triangles on most hemispheres (max 129 on
   `sub-022 rh`) but fewer on `sub-059` (353→88); all are <0.1% of faces either way — visually clean.

This is the strongest single result for the paper: a 3.4× speedup with no quality cost, validated on
held-out subjects. Logged as 10 ledger records (`exp:validate_speed:*`).

## 13. K-ring cache build parallelized (~26×) — removes the first-run penalty

The one-time k-ring cache build (the "Sampling neighbors" pass, ~4 min/hemi) was a **serial
Python loop** over all ~200–400k vertices while 31 of 32 cores idled. Folded the per-vertex work
(tangent-plane projection → per-ring angular sampling → limited Dijkstra) into a single
`@njit(parallel=True)` `prange` kernel (`_angular_kring_kernel` + an njit port of the angular
sampler). **Output is bit-identical** to the previous numba path — verified against the existing
`sub-022 lh` cache (0/193174 neighbor-set mismatches, 0 distance mismatches) and pinned by a
regression test. Steady-state runtime on `sub-022 lh` dropped **~230 s → 8.7 s (~26×)** (≈17 s with
cold JIT). This is pure performance with no numerical change, so it needs no re-validation of any
flatmap. It mostly eliminates the first-run penalty for a new subject (the 180–220 s steady-state
flatten is JAX and unaffected); the warm-cache flatten is unchanged.

**Authoritative end-to-end measurement (4 hemispheres, cold cache).** Timed the full per-hemisphere
pipeline with the shippable fast config — geometry I/O → cold k-ring cache build (parallel kernel) →
fast JAX flatten — into a temp cache so the validated caches were untouched:

| subject·hemi | I/O (s) | cache build (s) | flatten (s) | total (s) |
|---|---|---|---|---|
| sub-022 lh | 2.0 | 10.4 | 188.5 | 200.9 |
| sub-022 rh | 1.8 | 10.3 | 184.6 | 196.8 |
| sub-026 lh | 1.9 | 11.1 | 188.6 | 201.6 |
| sub-026 rh | 1.9 | 11.5 | 207.4 | 220.8 |
| **mean** | 1.9 | **10.8** | **192.3** | **205.0 (3.4 min)** |

Against the §12 baseline on the *same* 4 hemispheres (flatten mean 657.5 s) plus the serial cache
build (~230 s), end-to-end per hemisphere goes from **~887 s (~14.8 min) → 205 s (~3.4 min), ~4.3×**
cold-cache (~3.4× warm). The measured total matches the component-sum estimate to the second. Note the
cache build read 10.8 s here (not the ~17 s cold-JIT figure) because numba's compiled kernel was
already cached to disk — the realistic steady state after a machine's first-ever run; the one-time
cold-JIT adds ~7 s once per machine. Flips and quality are the validated fast-config values (the cache
build is bit-identical and the flatten config unchanged), so no re-scoring was needed.

## Method note

Determinism confirmed bit-identical across reruns, so a single run per experiment is sound.
Compute is CPU-only (the box's GPUs are blocked by driver 440 / CUDA 10.2).

## Next ideas

Remaining, in priority order (config levers are exhausted — see §10–11):

- **Long-range geodesic anchors in the energy** (the genuine open lever). Add a sparse set of
  true-geodesic long-range distance constraints, up-weighted vs the local k-ring, to directly constrain
  the global metric (Fischl 1999's own point that long-range distances are needed to unfold). Requires
  modifying the energy/optimizer (not a config knob) and weight-tuning; validate with train/test-split
  geodesic sources across subjects. Uncertain but the only path to a larger robust reduction.
- **Distance-optimal output scale** (small, safe): replace `scale_to_area` with the scale that minimizes
  distance distortion (a 1-parameter minimization, ≥0 by construction; `s*`≈0.995–1.02). Cheap to ship.
- **Ship the validated speed defaults**: Tutte init as default `initial_projection` with initial NAR off
  (§1), plus the §7 lean line-search / sparse k-ring levers.
