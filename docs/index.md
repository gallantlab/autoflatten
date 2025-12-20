# autoflatten

<p align="center">
  <img src="autoflatten-logo.png" alt="autoflatten logo" width="600">
</p>

**Automatically create cortical flatmaps from FreeSurfer surfaces**

---

**TL;DR:** run `autoflatten /path/to/your/freesurfer/subject` and you're done.

## Features

- **Automatic cut mapping** from a template to individual subjects
- **Two backends for the flattening process**:
    - **pyflatten** (default): JAX-accelerated Python implementation - fast and reliable
    - **freesurfer**: FreeSurfer `mris_flatten` wrapper
- **Visualization** of 3D projections and 2D flatmaps with distortion metrics

## Quick Install

```bash
pip install autoflatten
```

## Quick Start

```bash
# Full pipeline (projection + flattening)
autoflatten /path/to/subjects/sub-01

# With output to a separate directory
autoflatten /path/to/subjects/sub-01 --output-dir /path/to/output

# Process both hemispheres in parallel
autoflatten /path/to/subjects/sub-01 --parallel

# Visualize the 3D surface with cuts
autoflatten plot-projection lh.autoflatten.patch.3d

# Visualize the 2D flatmap with distortion metrics
autoflatten plot-flatmap lh.autoflatten.flat.patch.3d
```

## Requirements

- Python 3.10+
- FreeSurfer 6.0+ (for projection phase only)

See the [Getting Started](getting-started.md) guide for detailed installation instructions.

## Links

- [GitHub Repository](https://github.com/gallantlab/autoflatten)
- [PyPI Package](https://pypi.org/project/autoflatten/)

## Citation

If you use autoflatten in your research, please cite:

> Visconti di Oleggio Castello, M., & Gallant, J. L. (2025). autoflatten: automatically create cortical flatmaps from FreeSurfer surfaces. Zenodo. [https://doi.org/10.5281/zenodo.17933205](https://doi.org/10.5281/zenodo.17933205)

If you use the flattening procedure, please also cite the original paper describing the algorithm:

> Fischl, B., Sereno, M. I., & Dale, A. M. (1999). Cortical surface-based analysis II: Inflation, flattening, and a surface-based coordinate system. *NeuroImage*, 9(2), 195-207. [https://doi.org/10.1006/nimg.1998.0396](https://doi.org/10.1006/nimg.1998.0396)
