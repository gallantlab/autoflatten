# Detailed Options

This page describes all available CLI commands and options.

autoflatten provides a CLI with four commands:

| Command | Description |
|---------|-------------|
| `autoflatten /path/to/subject` | Full pipeline: projection + flattening |
| `autoflatten project /path/to/subject` | Projection only: create patch file |
| `autoflatten flatten PATCH_FILE` | Flattening only: flatten existing patch |
| `autoflatten plot FLAT_PATCH` | Plot a flattened surface |

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

Plot a flattened surface with quality metrics:

```bash
autoflatten plot lh.autoflatten.flat.patch.3d --subject sub-01
```

This generates a visualization showing:

- The flattened mesh
- Area distortion distribution
- Mean % distance error
- Number of negative (flipped) triangles

## Output Files

For each processed hemisphere, the pipeline creates:

| File | Description |
|------|-------------|
| `{hemi}.autoflatten.patch.3d` | 3D patch file with cuts |
| `{hemi}.autoflatten.flat.patch.3d` | 2D flattened surface |
| `{hemi}.autoflatten.flat.patch.3d.log` | Optimization log (pyflatten) |
| `{hemi}.autoflatten.flat.patch.png` | Visualization plot |
| `{hemi}.autoflatten.projection.log` | Projection phase log |

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
