# Example Output

This page shows the output of a basic autoflatten run and explains how to interpret the results.

## Running autoflatten

A minimal run with default parameters:

```bash
autoflatten /path/to/subjects/sub-01 --output-dir /path/to/output
```

## Output Files

For each hemisphere, autoflatten generates the following files:

| File | Description |
|------|-------------|
| `{hemi}.autoflatten.patch.3d` | 3D patch file with cuts applied |
| `{hemi}.autoflatten.flat.patch.3d` | 2D flattened surface |
| `{hemi}.autoflatten.flat.patch.3d.log` | Optimization log with metrics |
| `{hemi}.autoflatten.flat.patch.png` | Visualization plot |
| `{hemi}.autoflatten.projection.log` | Projection phase log |

## Visualization Output

The visualization plot (`{hemi}.autoflatten.flat.patch.png`) shows a three-panel figure:

<!-- TODO: Add example image here -->
<!-- ![Example flatmap](example-lh-flatmap.png) -->

**Left panel: Flatmap**

- The flattened cortical surface mesh
- Red triangles indicate flipped (negative area) triangles — ideally there should be none
- Yellow dots mark the centroids of flipped triangles for visibility when zoomed out
- Blue dots show boundary vertices

**Center panel: Metric Distortion**

- Per-vertex distortion map showing the percentage error between 2D and 3D geodesic distances
- Uses a viridis colormap (0-100% range)
- Lower values (darker colors) indicate better preservation of distances
- This metric matches the approach in Fischl et al., 1999

**Right panel: Distortion Distribution**

- Histogram of per-vertex distortion values
- Bars colored by distortion value using the same colormap
- Black dashed line shows the mean distortion
- Gray dotted line shows the median distortion

**Title metrics:**

- **Vertex and face count**: Size of the flattened mesh
- **% error**: Mean percentage distance error (lower is better)
- **Flipped count**: Number of flipped triangles (0 is ideal)

## Log File Contents

The log file (`{hemi}.autoflatten.flat.patch.3d.log`) contains detailed optimization progress:

```
=== pyflatten v1.0.0 ===
Input patch: lh.autoflatten.patch.3d
Base surface: lh.fiducial
Vertices: 120000, Faces: 240000

--- K-ring distance computation ---
k_ring: 7, n_neighbors_per_ring: 12
Computing distances... done (15.2s)

--- Optimization ---
Phase: initial_nar
  Step 0: E=1.234e+05, |grad|=5.67e+02, dt=0.001
  ...
  Converged after 50 steps

Phase: epoch_1 (l_nlarea=1.0, l_dist=0.1)
  ...

--- Final metrics ---
Mean distance error: 2.34%
Negative triangles: 0
Total time: 45.6s
```

### Key sections:

1. **Header**: Version, input files, mesh statistics
2. **K-ring computation**: Neighborhood size and timing
3. **Optimization phases**:
   - `initial_nar`: Initial negative area removal
   - `epoch_1`, `epoch_2`, `epoch_3`: Main optimization epochs
   - `final_nar`: Final cleanup
4. **Final metrics**: Summary of flattening quality
