# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoFlatten is a Python pipeline for automatically creating flattened versions of FreeSurfer cortical surfaces. The pipeline maps template cuts from fsaverage to individual subjects using FreeSurfer's surface-based registration, then creates patch files and runs `mris_flatten` to produce 2D flat representations.

## Development Commands

### Installation
```bash
pip install -e .              # Install package in development mode
pip install -e ".[test]"      # Install with test dependencies
```

### Testing

**Setup conda environment:**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sosem  # or stories_atl
```

**Run tests:**
```bash
pytest                        # Run all tests
pytest --cov=autoflatten      # Run tests with coverage report
pytest -k test_name           # Run specific test by name
pytest autoflatten/tests/test_core.py  # Run specific test file
```

**Or in a single command:**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sosem && pytest
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
autoflatten run SUBJECT_ID [options]    # Run flattening pipeline
autoflatten plot FLAT_PATCH_FILE [options]  # Plot results
```

## Code Style

- **Docstrings**: NumPy format (see `.github/copilot-instructions.md`)
- **Testing**: pytest framework
- **Python compatibility**: 3.9+

## Architecture Overview

### Pipeline Flow

The core pipeline in [cli.py](autoflatten/cli.py) follows these steps:

1. **Template Loading** ([config.py](autoflatten/config.py), [template.py](autoflatten/template.py))
   - Default: `fsaverage_cuts_template.json` contains medial wall + 5 anatomical cuts
   - Templates stored in `autoflatten/default_templates/`
   - Alternative templates: `SUBJ_A_cut_template.json`, `REDACTED_SUBJ_cut_template.json`

2. **Cut Mapping** ([core.py](autoflatten/core.py):`map_cuts_to_subject`)
   - Uses FreeSurfer's `mri_label2label` to map template cuts to target subject
   - Leverages FreeSurfer's surface-based registration (sphere.reg files)

3. **Cut Continuity** ([core.py](autoflatten/core.py):`ensure_continuous_cuts`)
   - **Critical component**: Fixes disconnected cuts after mapping
   - Uses NetworkX graphs built from surface triangulation
   - Finds connected components and connects them via shortest paths
   - Uses inflated surface for Euclidean distance, fiducial for path weights

4. **Geodesic Refinement** ([core.py](autoflatten/core.py):`refine_cuts_with_geodesic`)
   - Replaces mapped cuts with geodesic shortest paths between endpoints
   - Reduces wiggling from registration and flattening distortion
   - **Enabled by default** (disable with `--no-refine-geodesic` flag)

5. **Patch File Creation** ([freesurfer.py](autoflatten/freesurfer.py):`create_patch_file`)
   - Generates FreeSurfer-compatible patch file
   - Border vertices (adjacent to cuts) get positive indices
   - Interior vertices get negative indices
   - Output: `{hemi}.autoflatten.patch.3d`

6. **Surface Flattening** ([freesurfer.py](autoflatten/freesurfer.py):`run_mris_flatten`)
   - Calls FreeSurfer's `mris_flatten` with configurable parameters
   - **Key implementation detail**: Uses temporary directory isolation (since Nov 2025)
     - Copies necessary files to temp directory to work around FreeSurfer's path requirements
     - Moves output back to desired location
     - Prevents FreeSurfer from forcing outputs into subject's surf directory
   - Parameters: seed, distances, iterations, dilations, passes, tolerance
   - Output: `{hemi}.autoflatten_{params}_seed{seed}.flat.patch.3d`

### Key Modules

- **[cli.py](autoflatten/cli.py)**: Command-line interface with `run` and `plot` subcommands
  - `process_hemisphere()`: Main processing function for a single hemisphere
  - `run_flattening()`: Handles the 'run' subcommand
  - `run_plotting()`: Handles the 'plot' subcommand
  - Supports parallel hemisphere processing with `--parallel` flag

- **[core.py](autoflatten/core.py)**: Core graph algorithms for cut processing
  - Graph-based pathfinding using NetworkX
  - Surface topology analysis
  - All algorithms work on inflated surface for visualization and fiducial for accuracy

- **[freesurfer.py](autoflatten/freesurfer.py)**: FreeSurfer interface
  - Surface loading (with fallback: PyCortex DB → FreeSurfer SUBJECTS_DIR)
  - Label file I/O (FreeSurfer binary format)
  - Patch file creation (FreeSurfer binary format)
  - `mris_flatten` wrapper with temporary directory isolation
  - Binary format handling using `struct` module

- **[template.py](autoflatten/template.py)**: Template creation utilities
  - Functions to derive new templates from flattened surfaces
  - Surface component identification
  - Used by scripts in `scripts/` directory

- **[viz.py](autoflatten/viz.py)**: Visualization using matplotlib
  - Plot flat patches with proper aspect ratio
  - Overlay cuts and borders on inflated surface

- **[utils.py](autoflatten/utils.py)**: JSON I/O with NumPy array handling

### FreeSurfer Integration

**Requirements:**
- FreeSurfer 7.0+ must be installed
- `FREESURFER_HOME` environment variable must be set
- `SUBJECTS_DIR` environment variable must be set
- FreeSurfer binaries must be in PATH (`mris_flatten`, `mri_label2label`, etc.)

**Critical Implementation Detail - Temporary Directory Isolation:**
The `run_mris_flatten` function (added Nov 2025) uses a temporary directory workaround:
- FreeSurfer's `mris_flatten` has rigid requirements about file locations
- Cannot directly output to arbitrary directories
- Solution: Copy necessary files to temp dir, run `mris_flatten`, move outputs back
- Files copied: inflated surface, smoothwm, sphere.reg
- Controlled by `debug` flag - if True, temp dir is preserved for inspection

**Key FreeSurfer Commands Used:**
- `mri_label2label`: Map labels/cuts between subjects using sphere.reg registration
- `mris_flatten`: Flatten a patch to 2D while minimizing distortion
- `mri_info`: Check FreeSurfer version

### Surface Graph Construction

The graph-based algorithms in [core.py](autoflatten/core.py) build NetworkX graphs from surface triangulations:
```python
G = nx.Graph()
for triangle in polys:
    # Add edges between triangle vertices with Euclidean weights
    # Used for shortest path finding between cut segments
```

This is essential for:
- Finding connected components in cuts
- Computing geodesic paths on the surface
- Ensuring cuts are continuous after registration

### Template System

Templates are JSON files with hemisphere-specific vertex lists:
```json
{
  "lh_mwall": [vertex_indices],
  "lh_cut1": [vertex_indices],  // calcarine
  "lh_cut2": [vertex_indices],  // medial 1
  "lh_cut3": [vertex_indices],  // medial 2
  "lh_cut4": [vertex_indices],  // medial 3
  "lh_cut5": [vertex_indices],  // temporal
  "rh_mwall": [...],
  "rh_cut1": [...],
  ...
}
```

Create custom templates using `scripts/make_fsaverage_cut_template.py` or `scripts/make_subject_cut_template.py`.

## Testing Notes

- Tests require FreeSurfer installation (many are skipped if FreeSurfer unavailable)
- Test files use fixtures and mocking to avoid requiring real subject data
- Tests in [test_freesurfer.py](autoflatten/tests/test_freesurfer.py) validate:
  - Binary format reading/writing (labels, patches)
  - Temporary directory isolation behavior
  - FreeSurfer command execution
- Tests in [test_core.py](autoflatten/tests/test_core.py) validate graph algorithms
- Tests in [test_template.py](autoflatten/tests/test_template.py) validate template operations

## Important Implementation Details

1. **Graph weights use fiducial surface**: While Euclidean distances may use inflated surface for speed, actual pathfinding weights use fiducial (or computed fiducial from smoothwm + pial) for anatomical accuracy.

2. **Hemisphere processing can be parallel**: Use `--parallel` flag to process LH and RH simultaneously (threads are distributed between hemispheres).

3. **Deterministic patch files**: Patch file names don't include seed (deterministic). Flat file names include all parameters including seed for reproducibility.

4. **FreeSurfer version check**: CLI checks for FreeSurfer 7.0+ at startup.

5. **Output location flexibility**: `--output-dir` recommended for testing to keep outputs separate from subject data. Without it, outputs go to `$SUBJECTS_DIR/subject/surf` (with warning).
