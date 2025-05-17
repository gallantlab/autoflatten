# AutoFlatten: Automatic Flattening Pipeline for FreeSurfer Surfaces

This repository contains a Python pipeline for automatically creating flattened versions
of FreeSurfer surfaces. Template cuts on fsaverage are mapped to a new surface and
automatically fixed to generate a FreeSurfer patch file. Then, `mris_flatten` is called
to run the flattening process.

## Requirements

- Python 3.9+
- FreeSurfer 6.0 (properly installed with environment variables set)
- PyCortex
- NetworkX
- NumPy
- Nibabel

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/gallantlab/autoflatten.git
   cd autoflatten
   ```

2. Install the package:
   ```bash
   pip install .
   ```
   This will install the `autoflatten` command and all required dependencies.

3. Ensure FreeSurfer is properly installed and configured:
   - FREESURFER_HOME environment variable must be set
   - SUBJECTS_DIR environment variable must be set
   - FreeSurfer binaries must be in your PATH

## Usage

The `autoflatten` command has two subcommands:
- `run`: Run the flattening pipeline
- `plot`: Plot existing autoflatten results

Basic usage:

```bash
autoflatten run SUBJECT_ID [options]
autoflatten plot SUBJECT_ID [options]
```

### Arguments

- `SUBJECT_ID`: FreeSurfer subject identifier (required)

### Options for 'run' subcommand

- `--output-dir PATH`: Directory to save output files (default: subject's FreeSurfer surf directory)
- `--no-flatten`: Skip running mris_flatten after creating patch files (flattening is enabled by default)
- `--template-file PATH`: Path to a custom JSON template file defining cuts (default: uses built-in fsaverage template)
- `--parallel`: Process hemispheres in parallel (when processing both hemispheres)
- `--hemispheres {lh,rh,both}`: Hemispheres to process (default: both)
- `--overwrite`: Overwrite existing files
- `--seed INT`: Random seed for mris_flatten (if not provided, a random seed will be generated)
- `--nthreads INT`: Number of threads to use for mris_flatten (default: 1)
- `--distances DIST1 DIST2`: Distance parameters for mris_flatten as two integers (default: 15 80)
- `--n-iterations INT`: Maximum number of iterations for mris_flatten (default: 200)
- `--dilate INT`: Number of dilations for mris_flatten (default: 1)
- `--passes INT`: Number of passes for mris_flatten (default: 1)
- `--tol FLOAT`: Tolerance for mris_flatten flatness (default: 0.005)
- `--flatten-extra STR`: Additional parameters for mris_flatten in format 'key1=value1,key2=value2'

### Options for 'plot' subcommand

- `--output-dir PATH`: Directory containing the autoflatten output files (default: $SUBJECTS_DIR/subject/surf)

## Examples

Create patch files and flatten both hemispheres of a subject:
```bash
autoflatten run sub-01
```

Create patch files for the left hemisphere without flattening:
```bash
autoflatten run sub-01 --hemispheres lh --no-flatten
```

Process both hemispheres in parallel using a custom template:
```bash
autoflatten run sub-01 --parallel --template-file /path/to/my_template.json
```

Save output to a custom directory:
```bash
autoflatten run sub-01 --output-dir /path/to/output
```

Force regeneration of existing files:
```bash
autoflatten run sub-01 --overwrite
```

Use specific flattening parameters:
```bash
autoflatten run sub-01 --seed 42 --nthreads 4 --passes 2 --tol 0.001
```

Plot existing autoflatten results:
```bash
autoflatten plot sub-01
```

## How It Works

The pipeline follows these steps for each hemisphere:

1. **Load cuts template**:
   - By default, uses the built-in fsaverage cut template
   - The template contains vertex indices for the medial wall and five anatomically defined cuts:
     - Calcarine cut: along the calcarine sulcus
     - Medial cuts (1-3): cuts along the medial surface
     - Temporal cut: cut in the temporal lobe
   - Can use a custom template provided via JSON file

2. **Map cuts to target subject**:
   - Uses FreeSurfer's `mri_label2label` to map cuts from the template subject (fsaverage) to the individual subject's cortical surface
   - This mapping uses FreeSurfer's surface-based registration

3. **Ensure cuts are continuous**:
   - Processes the mapped cuts to ensure they form continuous lines on the surface
   - Uses graph-based algorithms to find and connect disconnected components
   - Computes shortest paths between disconnected segments to create continuous cuts
   - Handles cases where mappings create disconnected components

4. **Create patch file**:
   - Identifies vertices to exclude (medial wall and cuts)
   - Marks border vertices (adjacent to cuts) with positive indices
   - Marks interior vertices with negative indices
   - Generates a FreeSurfer-compatible patch file with the optimized cut pattern

5. **Surface flattening (default)**:
   - Runs FreeSurfer's `mris_flatten` on the patch file to create a flat surface
   - Uses parameters like seed, distances, iterations, dilations, passes, and tolerance to control the flattening process
   - The flattening process minimizes distortion while creating a 2D representation

## Available Templates

The package includes several built-in templates:

1. **fsaverage_cuts_template.json** (default):
   - Standard template based on the fsaverage subject
   - Created by Mark Lescroart and Natalia Bilenko
   - Provides anatomically consistent cuts across subjects

2. **SUBJ_A_cut_template.json**:
   - Alternative template with different cut patterns
   - May be useful for specific visualization needs

3. **REDACTED_SUBJ_cut_template.json**:
   - Another alternative template with different cut patterns
   - May be useful for specific visualization needs

You can also create custom templates using the scripts provided in the `scripts/` directory:
- `make_fsaverage_cut_template.py`: Create a template based on fsaverage
- `make_subject_cut_template.py`: Create a template based on a specific subject

## Output Files

For each processed hemisphere, the pipeline creates:

1. **Patch Files**:
   - `{hemi}.autoflatten.patch.3d`: The patch file with cuts
   - Contains the 3D coordinates of vertices with cuts removed
   - Used as input to the flattening process

2. **Flat Surface Files** (unless `--no-flatten` is used):
   - `{hemi}.autoflatten_{distances}_n{n}_dilate{dilate}_passes{passes}_seed{seed}.flat.patch.3d`
   - The naming includes the parameters used for flattening:
     - `distances`: The distance parameters used (e.g., distances1580 for 15 and 80)
     - `n`: Maximum number of iterations
     - `dilate`: Number of dilations
     - `passes`: Number of passes (if more than 1)
     - `seed`: Random seed used for flattening

3. **Log Files**:
   - `{hemi}.autoflatten_{parameters}.log`: Contains the output from the flattening process
   - Useful for debugging if flattening fails

4. **Output Files**:
   - `{hemi}.autoflatten_{parameters}.flat.patch.3d.out`: Additional output from mris_flatten

## License

This project is licensed under the BSD 2-Claude License - see the LICENSE file for details.

## Acknowledgments

- The default fsaverage template cuts were created by [Mark Lescroart and Natalia Bilenko](https://figshare.com/articles/dataset/fsaverage_subject_for_pycortex/)
- This pipeline builds on FreeSurfer and PyCortex functionalities
