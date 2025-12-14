# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoFlatten is a Python pipeline for automatically creating flattened versions of FreeSurfer cortical surfaces. The pipeline maps template cuts from fsaverage to individual subjects using FreeSurfer's surface-based registration, then creates patch files and runs surface flattening to produce 2D flat representations.

**Key feature**: AutoFlatten supports two flattening backends:
- **pyflatten** (default): JAX-accelerated Python implementation with vectorized optimization
- **freesurfer**: Traditional FreeSurfer mris_flatten wrapper

## Development Commands

### Installation
```bash
pip install -e .              # Install package in development mode
pip install -e ".[test]"      # Install with test dependencies
```

### Testing

**Run tests:**
```bash
pytest                        # Run all tests
pytest --cov=autoflatten      # Run tests with coverage report
pytest -k test_name           # Run specific test by name
pytest autoflatten/tests/test_core.py  # Run specific test file
```

### Code Quality
```bash
pre-commit install            # Install pre-commit hooks
pre-commit run --all-files    # Run all pre-commit checks manually
ruff format .                 # Format code (used by pre-commit)
```

The project uses pre-commit hooks for:
- trailing-whitespace, end-of-file-fixer
- YAML/JSON/TOML validation
- ruff formatting (Python)
- codespell (spell checking)

### Git Workflow

**IMPORTANT**: If pre-commit hooks fail and modify files (e.g., ruff formatting, trailing whitespace fixes):

- **NEVER amend the commit** - this can cause issues with git history
- Instead, create a new commit with the pre-commit fixes
- This ensures clean, traceable git history

### Running the CLI
```bash
# Full pipeline (projection + flattening):
autoflatten /path/to/subjects/sub-01

# Projection only (create patch file):
autoflatten project /path/to/subjects/sub-01

# Flatten an existing patch file:
autoflatten flatten lh.autoflatten.patch.3d

# Use FreeSurfer backend instead of pyflatten:
autoflatten /path/to/subjects/sub-01 --backend freesurfer

# Plot a flattened surface:
autoflatten plot lh.autoflatten.flat.patch.3d --subject sub-01
```

## Code Style

- **Docstrings**: NumPy format (see `.github/copilot-instructions.md`)
- **Testing**: pytest framework
- **Python compatibility**: 3.10+

## Architecture Overview

### Module Structure

```
autoflatten/
  cli.py                # CLI with project/flatten/plot subcommands
  config.py             # Configuration and template paths
  core.py               # Graph algorithms for cut processing
  freesurfer.py         # FreeSurfer I/O and command wrappers
  template.py           # Template creation utilities
  utils.py              # JSON I/O with NumPy handling
  viz.py                # Matplotlib-based visualization

  flatten/              # JAX-accelerated flattening algorithm
    __init__.py         # Exports: SurfaceFlattener, FlattenConfig, etc.
    algorithm.py        # SurfaceFlattener class and optimization functions
    config.py           # Configuration dataclasses
    distance.py         # K-ring geodesic distance computation
    energy.py           # Energy functions and gradients

  backends/             # Backend abstraction
    __init__.py         # get_backend(), available_backends()
    base.py             # Abstract FlattenBackend class
    freesurfer.py       # FreeSurfer mris_flatten wrapper
    pyflatten.py        # pyflatten wrapper
```

### Pipeline Flow

The CLI supports three modes:

1. **Full Pipeline** (`autoflatten /path/to/subject`): Projection + Flattening
2. **Projection Only** (`autoflatten project /path/to/subject`): Creates patch file
3. **Flatten Only** (`autoflatten flatten PATCH_FILE`): Flattens existing patch

#### Projection Phase

1. **Template Loading** ([config.py](autoflatten/config.py), [template.py](autoflatten/template.py))
   - Default: `fsaverage_cuts_template.json` contains medial wall + 5 anatomical cuts
   - Templates stored in `autoflatten/default_templates/`

2. **Cut Mapping** ([core.py](autoflatten/core.py):`map_cuts_to_subject`)
   - Uses FreeSurfer's `mri_label2label` to map template cuts to target subject
   - Leverages FreeSurfer's surface-based registration (sphere.reg files)

3. **Cut Continuity** ([core.py](autoflatten/core.py):`ensure_continuous_cuts`)
   - Fixes disconnected cuts after mapping using NetworkX graphs
   - Finds connected components and connects them via shortest paths

4. **Geodesic Refinement** ([core.py](autoflatten/core.py):`refine_cuts_with_geodesic`)
   - Replaces mapped cuts with geodesic shortest paths between endpoints
   - **Enabled by default** (disable with `--no-refine-geodesic`)

5. **Patch File Creation** ([freesurfer.py](autoflatten/freesurfer.py):`create_patch_file`)
   - Generates FreeSurfer-compatible patch file
   - Output: `{hemi}.autoflatten.patch.3d`

#### Flattening Phase

**pyflatten backend** (default):
- JAX-accelerated gradient descent with vectorized line search
- FreeSurfer-style 3-epoch optimization: negative area removal → epoch_1 → epoch_2 → epoch_3 → final NAR
- FreeSurfer-style convergence criteria
- Final spring smoothing for visual quality
- Output: `{hemi}.autoflatten.flat.patch.3d` + log file

**freesurfer backend**:
- Wraps FreeSurfer's `mris_flatten` command
- Uses temporary directory isolation
- Parameters: seed, distances, iterations, dilations, passes, tolerance

### Key Modules

- **[cli.py](autoflatten/cli.py)**: Command-line interface
  - `cmd_run_full_pipeline()`: Full pipeline
  - `cmd_project()`: Projection only
  - `cmd_flatten()`: Flattening only
  - `cmd_plot()`: Visualization
  - Supports parallel hemisphere processing with `--parallel`

- **[flatten/](autoflatten/flatten/)**: JAX-accelerated flattening
  - `SurfaceFlattener`: Main class orchestrating optimization
  - `FlattenConfig`: Configuration with phases, k-ring params, convergence settings
  - Energy functions: metric distortion (J_d) and area energy (J_a)
  - K-ring geodesic distance computation with Numba acceleration

- **[backends/](autoflatten/backends/)**: Backend abstraction
  - `FlattenBackend`: Abstract base class
  - `PyflattenBackend`: JAX-accelerated implementation
  - `FreeSurferBackend`: mris_flatten wrapper
  - `get_backend()`, `available_backends()`: Registry functions

- **[viz.py](autoflatten/viz.py)**: Matplotlib-based visualization
  - `plot_flatmap()`: Two-panel plot (mesh + area distribution)
  - Shows flipped triangles in red, boundary vertices
  - Parses log file for optimization results

- **[freesurfer.py](autoflatten/freesurfer.py)**: FreeSurfer interface
  - Surface and patch file I/O (binary format)
  - `mris_flatten` wrapper with temporary directory isolation

### FreeSurfer Integration

**Requirements (for projection phase):**
- FreeSurfer 7.0+ must be installed
- `FREESURFER_HOME` and `SUBJECTS_DIR` environment variables must be set
- FreeSurfer binaries must be in PATH (`mri_label2label`, etc.)

**Note**: The pyflatten backend does NOT require FreeSurfer for flattening,
only for the projection phase (cut mapping).

### Pyflatten Algorithm

The pyflatten backend implements FreeSurfer-style optimization with JAX:

1. **K-ring Distance Computation**: Geodesic distances to k-hop neighbors
   - Numba-accelerated Dijkstra's algorithm
   - Angular sampling for efficient memory usage

2. **FreeSurfer-style 3-Epoch Optimization**:
   - Initial negative area removal (l_nlarea=1.0, varying l_dist)
   - Epoch 1: Area-dominant (l_nlarea=1.0, l_dist=0.1)
   - Epoch 2: Balanced (l_nlarea=1.0, l_dist=1.0)
   - Epoch 3: Distance-dominant (l_nlarea=0.1, l_dist=1.0)
   - Final negative area removal (tighter tolerance)

3. **Energy Functions**:
   - J_d: Metric distortion (preserve geodesic distances)
   - J_a: Area energy (prevent flipped triangles)

4. **Vectorized Line Search**: Log-spaced step sizes with quadratic refinement

5. **Final Spring Smoothing**: Laplacian smoothing for visual quality

### Configuration

```python
from autoflatten.flatten import FlattenConfig, SurfaceFlattener

config = FlattenConfig()
config.kring.k_ring = 7  # Neighborhood size
config.kring.n_neighbors_per_ring = 12  # Angular sampling

flattener = SurfaceFlattener(config)
flattener.load_data("lh.patch.3d", "lh.smoothwm")  # or lh.fiducial
flattener.compute_kring_distances()
flattener.prepare_optimization()
uv = flattener.run()
flattener.save_result(uv, "lh.flat.patch.3d")
```

## Testing Notes

- Tests require FreeSurfer installation (many are skipped if FreeSurfer unavailable)
- Test files use fixtures and mocking to avoid requiring real subject data
- Tests in [test_freesurfer.py](autoflatten/tests/test_freesurfer.py) validate:
  - Binary format reading/writing (labels, patches)
  - Temporary directory isolation behavior
- Tests in [test_core.py](autoflatten/tests/test_core.py) validate graph algorithms
- Tests in [test_template.py](autoflatten/tests/test_template.py) validate template operations

## Important Implementation Details

1. **Graph weights use fiducial surface**: Pathfinding uses fiducial surface for anatomical accuracy.

2. **Hemisphere processing can be parallel**: Use `--parallel` flag to process LH and RH simultaneously.

3. **pyflatten is the default backend**: Installed by default with all dependencies. Falls back to FreeSurfer if JAX unavailable.

4. **Output location flexibility**: `--output-dir` recommended to keep outputs separate from subject data.

5. **Log files**: pyflatten creates detailed log files (`*.flat.patch.3d.log`) with optimization progress and final metrics.

6. **Base surface auto-detection**: For the full pipeline, automatically uses `{hemi}.fiducial` (falls back to `{hemi}.smoothwm`) from the subject's surf/ directory. For `flatten` subcommand, use `--base-surface` to specify a custom path.
