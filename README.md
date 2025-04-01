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

Basic usage:

```bash
autoflatten SUBJECT_ID [options]
```

### Arguments

- `SUBJECT_ID`: FreeSurfer subject identifier (required)

### Options

- `--output-dir PATH`: Directory to save output files (default: subject's FreeSurfer surf directory)
- `--no-flatten`: Skip running mris_flatten after creating patch files (flattening is enabled by default)
- `--template-file PATH`: Path to a custom JSON template file defining cuts (default: uses built-in fsaverage template)
- `--parallel`: Process hemispheres in parallel (when processing both hemispheres)
- `--hemispheres {lh,rh,both}`: Hemispheres to process (default: both)
- `--overwrite`: Overwrite existing files

## Examples

Create patch files and flatten both hemispheres of a subject:
```bash
autoflatten sub-01
```

Create patch files for the left hemisphere without flattening:
```bash
autoflatten sub-01 --hemispheres lh --no-flatten
```

Process both hemispheres in parallel using a custom template:
```bash
autoflatten sub-01 --parallel --template-file /path/to/my_template.json
```

Save output to a custom directory:
```bash
autoflatten sub-01 --output-dir /path/to/output
```

Force regeneration of existing files:
```bash
autoflatten sub-01 --overwrite
```

## How It Works

The pipeline follows these steps for each hemisphere:

1. **Load cuts template**:
   - By default, uses the built-in fsaverage cut template
   - Can use a custom template provided via JSON file

2. **Map cuts to target subject**:
   - Uses FreeSurfer's registration to map cuts to the individual subject's cortical surface

3. **Ensure cuts are continuous**:
   - Processes the mapped cuts to ensure they form continuous lines on the surface
   - Handles cases where mappings create disconnected components

4. **Create patch file**:
   - Generates a FreeSurfer-compatible patch file with the optimized cut pattern

5. **Surface flattening (default)**:
   - Runs mris_flatten on the patch file to create a flat surface

## Output Files

For each processed hemisphere, the pipeline creates:
- `{hemi}.autoflatten.patch.3d`: The patch file with cuts
- `{hemi}.autoflatten.flat.patch.3d`: The flattened surface (unless `--no-flatten` is used)

## License

This project is licensed under the BSD 2-Claude License - see the LICENSE file for details.

## Acknowledgments

- The default fsaverage template cuts were created by [Mark Lescroart and Natalia Bilenko](https://figshare.com/articles/dataset/fsaverage_subject_for_pycortex/)
- This pipeline builds on FreeSurfer and PyCortex functionalities
