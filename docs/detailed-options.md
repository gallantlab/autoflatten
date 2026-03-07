# Detailed Options

This page describes all available CLI commands and options.

autoflatten provides a CLI with six commands:

| Command | Description |
|---------|-------------|
| `autoflatten /path/to/subject` | Full pipeline: projection + flattening |
| `autoflatten project /path/to/subject` | Projection only: create patch file |
| `autoflatten flatten PATCH_FILE` | Flattening only: flatten existing patch |
| `autoflatten plot-projection PATCH` | Plot 3D surface with cuts highlighted |
| `autoflatten plot-flatmap FLAT_PATCH` | Plot 2D flatmap with distortion metrics |
| `autoflatten render-snapshots NPZ` | Render animation frames from optimization snapshots |

## Full Pipeline

Process a subject through the complete pipeline (projection + flattening):

```bash
# Basic usage with default pyflatten backend
autoflatten /path/to/subjects/sub-01 --output-dir /path/to/output

# Use FreeSurfer backend instead
autoflatten /path/to/subjects/sub-01 --backend freesurfer

# Process both hemispheres in parallel
autoflatten /path/to/subjects/sub-01 --parallel

# Process only left hemisphere
autoflatten /path/to/subjects/sub-01 --hemispheres lh
```

## Projection Only

Create patch files without running the flattening step:

```bash
autoflatten project /path/to/subjects/sub-01 --output-dir /path/to/output
```

This creates `{hemi}.autoflatten.patch.3d` files that can be flattened later.

## Flattening Only

Flatten an existing patch file:

```bash
# Basic usage
autoflatten flatten lh.autoflatten.patch.3d

# Specify base surface explicitly
autoflatten flatten lh.autoflatten.patch.3d --base-surface /path/to/lh.smoothwm

# Customize pyflatten parameters
autoflatten flatten lh.autoflatten.patch.3d --k-ring 25 --n-neighbors 40
```

## Visualization

### Plot 3D Projection

Plot the 3D surface with cuts highlighted from multiple views:

```bash
# Basic usage (auto-detects subject directory from patch location)
autoflatten plot-projection lh.autoflatten.patch.3d

# Specify subject directory explicitly
autoflatten plot-projection lh.autoflatten.patch.3d --subject-dir /path/to/subject/surf

# Custom output path
autoflatten plot-projection lh.autoflatten.patch.3d --output /path/to/output.png
```

This generates a three-panel view (medial, ventral, frontal) showing the surface with cut vertices highlighted in red.

### Plot 2D Flatmap

Plot a flattened surface with quality metrics:

```bash
# Basic usage (auto-detects base surface)
autoflatten plot-flatmap lh.autoflatten.flat.patch.3d

# Specify subject directory for base surface lookup
autoflatten plot-flatmap lh.autoflatten.flat.patch.3d --subject-dir /path/to/subject/surf

# Custom output path
autoflatten plot-flatmap lh.autoflatten.flat.patch.3d --output /path/to/output.png
```

This generates a three-panel visualization showing:

- The flattened mesh with flipped triangles highlighted
- Per-vertex metric distortion map
- Distortion distribution histogram

## Animation

You can capture the optimization process as a video to visualize how the surface is flattened over time. This is a two-step process: first capture snapshots during flattening, then render them as frames.

### Step 1: Capture Snapshots

Add `--save-snapshots` when running `flatten` or the full pipeline:

```bash
# During flattening
autoflatten flatten lh.autoflatten.patch.3d --save-snapshots snapshots.npz

# During full pipeline
autoflatten /path/to/subjects/sub-01 --save-snapshots snapshots.npz

# Control snapshot frequency (default: every 10 iterations)
autoflatten flatten lh.autoflatten.patch.3d --save-snapshots snapshots.npz --snapshot-every 5
```

When processing both hemispheres, snapshot paths are automatically suffixed (e.g., `snapshots_lh.npz`, `snapshots_rh.npz`).
### Step 2: Render Frames

```bash
# Basic rendering with curvature shading
autoflatten render-snapshots snapshots.npz --subject-dir /path/to/subject

# Show area distortion instead of curvature
autoflatten render-snapshots snapshots.npz --color-mode distortion

# Customize output
autoflatten render-snapshots snapshots.npz \
    --output-dir my_frames \
    --n-frames 60 \
    --fps 10 \
    --dpi 200
```

### Step 3: Assemble Video

Use ffmpeg to combine the frames into a video:

```bash
ffmpeg -r 15 -i flatten_frames/frame_%04d.png \
    -c:v libx264 -pix_fmt yuv420p flatten.mp4
```

### Render Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output-dir` | `flatten_frames/` | Directory for output PNGs |
| `--n-frames` | 120 | Number of frames to render |
| `--fps` | 15 | Frames per second (used in suggested ffmpeg command) |
| `--color-mode` | `curvature` | Face coloring: `curvature` or `distortion` |
| `--subject-dir` | - | FreeSurfer subject directory (auto-detect curvature) |
| `--curv-path` | - | Path to curvature file (e.g., `lh.curv`) |
| `--figsize` | 6.0 | Figure size in inches |
| `--dpi` | 150 | Resolution in DPI |
| `--overwrite` | False | Overwrite existing frame files |

The renderer automatically adds hold frames at the start, end, and phase transitions to create natural pacing in the video.

## Output Files

For each processed hemisphere, the pipeline creates:

| File | Description |
|------|-------------|
| `{hemi}.autoflatten.patch.3d` | 3D patch file with cuts |
| `{hemi}.autoflatten.flat.patch.3d` | 2D flattened surface |
| `{hemi}.autoflatten.flat.patch.3d.log` | Optimization log (pyflatten) |
| `{hemi}.autoflatten.patch.png` | 3D projection visualization |
| `{hemi}.autoflatten.flat.patch.png` | 2D flatmap visualization |
| `{hemi}.autoflatten.projection.log` | Projection phase log |
| `snapshots.npz` | Optimization snapshots (with `--save-snapshots`) |

## Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | subject's surf/ | Directory to save output files |
| `--hemispheres` | both | Hemispheres to process (lh, rh, or both) |
| `--parallel` | False | Process hemispheres in parallel |
| `--overwrite` | False | Overwrite existing files |

## pyflatten Options

| Option | Default | Description |
|--------|---------|-------------|
| `--k-ring` | 7 | K-ring neighborhood size |
| `--n-neighbors` | 12 | Neighbors per ring (angular sampling) |
| `--n-cores` | -1 | CPU cores (-1 = all) |
| `--skip-phase` | - | Skip specific optimization phases |
| `--skip-spring-smoothing` | False | Skip final smoothing |
| `--save-snapshots` | - | Save optimization snapshots to `.npz` file |
| `--snapshot-every` | 10 | Save snapshot every N iterations |

## FreeSurfer Backend Options

| Option | Default | Description |
|--------|---------|-------------|
| `--seed` | random | Random seed for mris_flatten |
| `--nthreads` | 1 | Number of threads |
| `--distances` | 15 80 | Distance parameters |
| `--n-iterations` | 200 | Maximum iterations |
| `--tol` | 0.005 | Flatness tolerance |

## Custom Templates

Use a custom template with the `--template-file` option:

```bash
autoflatten /path/to/subjects/sub-01 --template-file /path/to/template.json
```

The default template (`fsaverage_cuts_template.json`) contains the medial wall + 5 anatomical cuts.
