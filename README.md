# AutoFlatten: Automatic Flattening Pipeline for FreeSurfer Surfaces

This repository contains a Python pipeline for automatically creating flattened cortical surfaces from FreeSurfer subject data. It maps established cutting patterns from fsaverage to new subjects and ensures continuous cuts, making the process of cortical flattening consistent and reproducible.

## Features

- Automatic mapping of standardized fsaverage cutting patterns to any FreeSurfer subject
- Ensures continuous cuts in the target subject surface
- Creates anatomically consistent patch files for visualization and analysis
- Optional surface flattening using FreeSurfer's mris_flatten
- Support for parallel processing of hemispheres
- Customizable iteration parameters for flattening

## Requirements

- Python 3.6+
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

2. Install the required Python dependencies:
   ```bash
   pip install numpy networkx nibabel pycortex
   ```

3. Ensure FreeSurfer is properly installed and configured:
   - FREESURFER_HOME environment variable must be set
   - SUBJECTS_DIR environment variable must be set
   - FreeSurfer binaries must be in your PATH

## Usage

Basic usage:

```bash
python auto_flatten.py SUBJECT_ID [options]
```

### Arguments

- `SUBJECT_ID`: FreeSurfer subject identifier (required)

### Options

- `--output-dir PATH`: Directory to save output files (default: subject's FreeSurfer surf directory)
- `--flatten`: Run mris_flatten after creating patch files
- `--parallel`: Process hemispheres in parallel (when processing both hemispheres)
- `--iterations N`: Number of iterations for mris_flatten (if not specified, uses mris_flatten default)
- `--hemispheres {lh,rh,both}`: Hemispheres to process (default: both)
- `--overwrite`: Overwrite existing files

## Examples

Create patch files for both hemispheres of a subject:
```bash
python auto_flatten.py sub-01
```

Create patch files and run flattening for the left hemisphere only:
```bash
python auto_flatten.py sub-01 --flatten --hemispheres lh
```

Process both hemispheres in parallel with custom flattening iterations:
```bash
python auto_flatten.py sub-01 --flatten --parallel --iterations 15
```

Save output to a custom directory:
```bash
python auto_flatten.py sub-01 --output-dir /path/to/output --flatten
```

Force regeneration of existing files:
```bash
python auto_flatten.py sub-01 --overwrite
```

## How It Works

The pipeline follows these steps for each hemisphere:

1. **Get medial wall and cuts from fsaverage**:
   - Retrieves standardized cut patterns from the fsaverage template from PyCortex

2. **Map cuts to target subject**:
   - Uses FreeSurfer's registration to map cuts to the individual subject's cortical surface

3. **Ensure cuts are continuous**:
   - Processes the mapped cuts to ensure they form continuous lines on the surface
   - Handles cases where mappings create disconnected components

4. **Create patch file**:
   - Generates a FreeSurfer-compatible patch file with the optimized cut pattern

5. **Optional flattening**:
   - Runs mris_flatten on the patch file to create a flattened representation

## Output Files

For each processed hemisphere, the pipeline creates:
- `{hemi}.autoflatten.patch.3d`: The patch file with cuts
- `{hemi}.autoflatten.flat.patch.3d`: The flattened surface (if `--flatten` is used)

## License

This project is licensed under the BSD 2-Claude License - see the LICENSE file for details.

## Acknowledgments

- The cutting pattern is derived from Mark Lescroart's fsaverage flatmap
- This pipeline builds on FreeSurfer and PyCortex functionalities
