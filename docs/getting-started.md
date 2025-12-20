# Getting Started

## Requirements

- **Python 3.10+**
- **FreeSurfer 6.0+** (for the projection phase)
    - `FREESURFER_HOME` and `SUBJECTS_DIR` environment variables must be set
    - FreeSurfer binaries must be in your PATH

!!! note "FreeSurfer is only needed for projection"
    The pyflatten backend does NOT require FreeSurfer for the flattening step itself - only for the initial projection phase where cuts are mapped from the template to your subject.

## Installation

### From PyPI (recommended)

```bash
pip install autoflatten
```

### From source (for development)

```bash
git clone https://github.com/gallantlab/autoflatten.git
cd autoflatten

# Using uv (recommended)
uv pip install -e ".[test]"

# Using pip
pip install -e ".[test]"
```

## FreeSurfer Setup

Make sure FreeSurfer is properly configured:

```bash
# Set environment variables (add to your ~/.bashrc or ~/.zshrc)
export FREESURFER_HOME=/path/to/freesurfer
export SUBJECTS_DIR=/path/to/your/subjects
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```

Verify your setup:

```bash
# Check FreeSurfer is available
which mri_label2label

# Check environment variables
echo $FREESURFER_HOME
echo $SUBJECTS_DIR
```

## Verify Installation

```bash
# Check autoflatten is installed
autoflatten --help

# Check version
autoflatten --version
```

## Next Steps

See the [Example Output](example-output.md) page to understand the output files, or the [Detailed Options](detailed-options.md) page for all CLI options.
